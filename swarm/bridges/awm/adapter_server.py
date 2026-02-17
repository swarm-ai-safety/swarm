"""AWM Adapter Server — bridges AWM-1K FastAPI apps to SWARM's MCP endpoints.

Loads an AWM scenario's ``full_code`` from ``gen_envs.jsonl``, ``exec()``s it to
obtain the upstream FastAPI ``app``, and re-exposes it through the endpoints that
``AWMMCPSyncClient`` expects (``/tools``, ``/tools/call``, ``/reset``, ``/verify``,
``/task``, ``/transaction/*``).

Usage::

    python -m swarm.bridges.awm.adapter_server \
        --scenario content_platform_1 \
        --envs-jsonl data/awm-1k/gen_envs.jsonl \
        --db-dir external/awm-envs/outputs/databases \
        --host 127.0.0.1 --port 9100 \
        --data-path data/awm-1k
"""

import argparse
import inspect
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _load_scenario_code(envs_jsonl: Path, scenario: str) -> str:
    """Return the ``full_code`` string for *scenario* from *envs_jsonl*."""
    with open(envs_jsonl) as fh:
        for line in fh:
            entry = json.loads(line)
            if entry["scenario"] == scenario:
                return str(entry["full_code"])
    raise ValueError(f"Scenario {scenario!r} not found in {envs_jsonl}")


def _load_tasks(tasks_jsonl: Path, scenario: str) -> List[str]:
    """Return task descriptions for *scenario* from ``gen_tasks.jsonl``."""
    with open(tasks_jsonl) as fh:
        for line in fh:
            entry = json.loads(line)
            if entry["scenario"] == scenario:
                return list(entry["tasks"])
    return []


def _load_verifier_code(verifier_jsonl: Path, scenario: str, task_idx: int) -> Optional[str]:
    """Return the verifier ``code`` string for *scenario* / *task_idx*."""
    with open(verifier_jsonl) as fh:
        for line in fh:
            entry = json.loads(line)
            if entry["scenario"] == scenario and entry.get("task_idx") == task_idx:
                return str(entry["verification"]["code"])
    return None


def _exec_awm_app(code: str, db_url: str) -> Any:
    """``exec()`` the AWM full_code and return the FastAPI ``app`` object.

    We override ``DATABASE_PATH`` env-var so the AWM code's
    ``os.getenv("DATABASE_PATH", ...)`` picks up our working DB.

    A synthetic module is created so that Pydantic v2 can resolve
    forward-reference type annotations (``List``, ``Optional``, etc.)
    inside the exec'd namespace.
    """
    import types

    os.environ["DATABASE_PATH"] = db_url

    # Create a real module so Pydantic's model_rebuild() can locate types
    mod = types.ModuleType(f"_awm_env_{id(code)}")
    mod.__file__ = "<awm-env>"
    sys.modules[mod.__name__] = mod

    exec(code, mod.__dict__)  # noqa: S102 — trusted AWM-1K code

    app = getattr(mod, "app", None)
    if app is None:
        raise RuntimeError("AWM full_code did not define an `app` variable")

    # Rebuild any Pydantic models so forward refs resolve against the module
    from pydantic import BaseModel as _BM

    for obj in mod.__dict__.values():
        if isinstance(obj, type) and issubclass(obj, _BM) and obj is not _BM:
            try:
                obj.model_rebuild(_types_namespace=mod.__dict__)  # type: ignore[attr-defined]
            except Exception:
                pass

    return app


def _extract_tools_from_app(awm_app: Any) -> List[Dict[str, Any]]:
    """Introspect an AWM FastAPI app's routes and return MCP-style tool dicts."""
    # Built-in FastAPI routes to skip
    _SKIP_NAMES = {"openapi", "swagger_ui_html", "swagger_ui_redirect", "redoc_html"}

    tools: List[Dict[str, Any]] = []
    for route in awm_app.routes:
        if not hasattr(route, "endpoint"):
            continue
        endpoint = route.endpoint
        name = getattr(route, "operation_id", None) or endpoint.__name__
        if name in _SKIP_NAMES:
            continue
        description = getattr(route, "description", "") or ""
        summary = getattr(route, "summary", "") or ""

        # Build a minimal inputSchema from the endpoint's signature
        sig = inspect.signature(endpoint)
        properties: Dict[str, Any] = {}
        required: List[str] = []
        for pname, param in sig.parameters.items():
            if pname in ("self", "db"):
                continue
            prop: Dict[str, Any] = {"type": "string"}
            if param.annotation is not inspect.Parameter.empty:
                # Try to get a more specific type
                ann = param.annotation
                if ann is int:
                    prop = {"type": "integer"}
                elif ann is float:
                    prop = {"type": "number"}
                elif ann is bool:
                    prop = {"type": "boolean"}
            if param.default is inspect.Parameter.empty:
                required.append(pname)
            properties[pname] = prop

        methods = list(getattr(route, "methods", {"GET"}))
        tools.append(
            {
                "name": name,
                "description": summary or description,
                "inputSchema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
                "method": methods[0] if methods else "GET",
                "path": getattr(route, "path", "/"),
            }
        )
    return tools


def build_adapter(
    *,
    scenario: str,
    envs_jsonl: Path,
    db_dir: Path,
    data_path: Path,
    task_idx: int = 0,
) -> Any:
    """Build and return the adapter FastAPI application."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    # ── Resolve file paths ───────────────────────────────────────────
    tasks_jsonl = data_path / "gen_tasks.jsonl"
    verifier_jsonl = data_path / "gen_verifier.pure_code.jsonl"

    source_db = db_dir / f"{scenario}.db"
    if not source_db.exists():
        raise FileNotFoundError(f"Database not found: {source_db}")

    # Working copy so /reset can restore from the original
    work_dir = Path(tempfile.mkdtemp(prefix=f"awm-{scenario}-"))
    initial_db = work_dir / "initial.db"
    working_db = work_dir / "working.db"
    shutil.copy2(source_db, initial_db)
    shutil.copy2(source_db, working_db)

    db_url = f"sqlite:///{working_db}"

    # ── Load AWM app ─────────────────────────────────────────────────
    awm_code = _load_scenario_code(envs_jsonl, scenario)
    awm_app = _exec_awm_app(awm_code, db_url)
    tools_list = _extract_tools_from_app(awm_app)

    # Build lookup: tool name → (method, path)
    _SKIP = {"openapi", "swagger_ui_html", "swagger_ui_redirect", "redoc_html"}
    tool_meta: Dict[str, Dict[str, str]] = {}
    for route in awm_app.routes:
        if not hasattr(route, "endpoint"):
            continue
        op_id = getattr(route, "operation_id", None) or route.endpoint.__name__
        if op_id in _SKIP:
            continue
        methods = list(getattr(route, "methods", {"GET"}))
        tool_meta[op_id] = {
            "method": methods[0] if methods else "GET",
            "path": getattr(route, "path", "/"),
        }

    # Load tasks & current task index
    tasks = _load_tasks(tasks_jsonl, scenario) if tasks_jsonl.exists() else []
    current_task_idx = task_idx

    # ── Adapter app ──────────────────────────────────────────────────
    adapter = FastAPI(title=f"AWM Adapter — {scenario}")

    @adapter.get("/tools")
    async def list_tools() -> JSONResponse:
        return JSONResponse({"tools": tools_list})

    @adapter.post("/tools/call")
    async def call_tool(request: Request) -> JSONResponse:
        import httpx

        body = await request.json()
        tool_name = body.get("name", "")
        arguments = body.get("arguments", {})

        if tool_name not in tool_meta:
            return JSONResponse(
                {"isError": True, "result": f"Unknown tool: {tool_name}"},
                status_code=404,
            )

        meta = tool_meta[tool_name]
        method = meta["method"]
        path = meta["path"]

        # Substitute path parameters from arguments
        rendered_path = path
        path_params_used: set = set()
        def _validate_path_param_value(name: str, value: Any) -> str:
            """
            Validate a path parameter value to avoid unsafe characters that could
            alter the intended route (for example, path traversal or injection of
            additional segments). Returns the stringified value if valid, otherwise
            raises a ValueError.
            """
            s = str(value)
            # Disallow path separators and control characters
            if "/" in s or "\\" in s:
                raise ValueError(f"Invalid path parameter '{name}': unexpected '/' or '\\\\'")
            if "\n" in s or "\r" in s or "\t" in s:
                raise ValueError(f"Invalid path parameter '{name}': unexpected whitespace")
            # Basic traversal protection
            if s.startswith(".") or ".." in s:
                raise ValueError(f"Invalid path parameter '{name}': potentially unsafe value")
            return s


        def _validate_path_param_value(name: str, value: Any) -> str:
            """
                try:
                    safe_value = _validate_path_param_value(param_name, arguments[param_name])
                except ValueError as exc:
                    return JSONResponse(
                        {"isError": True, "result": str(exc)},
                        status_code=400,
                    )
                rendered_path = rendered_path.replace(f"{{{param_name}}}", safe_value)
            """
            value_str = str(value)
        # Ensure the rendered path is safe and not an absolute URL
        def _validate_rendered_path(p: str) -> Optional[str]:
            """
            Validate that the rendered path cannot be used to perform SSRF.

            - Must not contain a scheme (no "http://", "https://", etc.).
            - Must not start with '//' (scheme-relative URL).
            - Must start with '/' to be a normal path on the fixed base_url.
            - Must not contain path traversal components that escape the API root.
            """
            parsed = urlparse(p)
            if parsed.scheme or parsed.netloc:
                return "Absolute URLs are not allowed in tool paths."
            if p.startswith("//"):
                return "Scheme-relative URLs are not allowed in tool paths."
            if not p.startswith("/"):
                return "Tool paths must start with '/'."
            # Basic traversal check; adjust if more flexibility is needed.
            if "/../" in p or p.endswith("/.."):
                return "Path traversal components are not allowed in tool paths."
            return None

        path_error = _validate_rendered_path(rendered_path)
        if path_error is not None:
            return JSONResponse(
                {"isError": True, "result": path_error},
                status_code=400,
            )

            if "/" in value_str or value_str.startswith("."):
                raise ValueError(f"Invalid value for path parameter '{name}'")
            return value_str

        for match in re.finditer(r"\{(\w+)\}", path):
            param_name = match.group(1)
            if param_name in arguments:
                try:
                    safe_value = _validate_path_param_value(param_name, arguments[param_name])
                except ValueError as exc:
                    return JSONResponse(
                        {"isError": True, "result": str(exc)},
                        status_code=400,
                    )
                rendered_path = rendered_path.replace(f"{{{param_name}}}", safe_value)
                path_params_used.add(param_name)

        remaining = {k: v for k, v in arguments.items() if k not in path_params_used}

        # Dispatch via httpx ASGITransport directly to the AWM app
        transport = httpx.ASGITransport(app=awm_app)  # type: ignore[arg-type]
        async with httpx.AsyncClient(transport=transport, base_url="http://awm") as client:
            try:
                if method.upper() in ("POST", "PUT", "PATCH"):
                    if method.upper() == "PUT":
                        resp = await client.put(rendered_path, json=remaining)
                    elif method.upper() == "PATCH":
                        resp = await client.patch(rendered_path, json=remaining)
                    else:
                        resp = await client.post(rendered_path, json=remaining)
                elif method.upper() == "DELETE":
                    resp = await client.delete(rendered_path, params=remaining)
                else:
                    resp = await client.get(rendered_path, params=remaining)
            except Exception as exc:
                return JSONResponse(
                    {"isError": True, "result": str(exc)},
                    status_code=500,
                )

        if resp.status_code >= 400:
            return JSONResponse(
                {"isError": True, "result": resp.text},
                status_code=resp.status_code,
            )

        try:
            result = resp.json()
        except Exception:
            result = resp.text

        return JSONResponse({"result": result})

    @adapter.post("/reset")
    async def reset_environment() -> JSONResponse:
        nonlocal awm_app
        shutil.copy2(initial_db, working_db)
        # Re-exec the AWM app so SQLAlchemy picks up the fresh DB
        awm_app = _exec_awm_app(awm_code, db_url)
        # Rebuild tool metadata (should be identical)
        nonlocal tool_meta
        tool_meta.clear()
        for route in awm_app.routes:
            if not hasattr(route, "endpoint"):
                continue
            op_id = getattr(route, "operation_id", None) or route.endpoint.__name__
            if op_id in _SKIP:
                continue
            methods = list(getattr(route, "methods", {"GET"}))
            tool_meta[op_id] = {
                "method": methods[0] if methods else "GET",
                "path": getattr(route, "path", "/"),
            }
        return JSONResponse({"status": "reset"})

    @adapter.post("/verify")
    async def verify_task() -> JSONResponse:
        nonlocal current_task_idx
        if not verifier_jsonl.exists():
            return JSONResponse({"passed": False, "error": "no verifier data"})

        code = _load_verifier_code(verifier_jsonl, scenario, current_task_idx)
        if code is None:
            return JSONResponse({"passed": False, "error": "verifier not found for task"})

        ns: Dict[str, Any] = {}
        exec(code, ns)  # noqa: S102 — trusted AWM-1K verifier code
        verify_fn = ns.get("verify_task_completion")
        if verify_fn is None:
            return JSONResponse({"passed": False, "error": "verify_task_completion not found"})

        try:
            result = verify_fn(str(initial_db), str(working_db))
        except Exception as exc:
            return JSONResponse({"passed": False, "error": str(exc)})

        passed = result.get("result") == "complete"
        return JSONResponse({"passed": passed, "confidence": 0.8, "details": result})

    @adapter.get("/task")
    async def get_task() -> JSONResponse:
        nonlocal current_task_idx
        if not tasks or current_task_idx >= len(tasks):
            return JSONResponse(
                {"task_id": scenario, "description": f"Complete tasks for {scenario}"}
            )
        return JSONResponse(
            {
                "task_id": f"{scenario}_{current_task_idx}",
                "description": tasks[current_task_idx],
            }
        )

    @adapter.post("/transaction/begin")
    async def transaction_begin() -> JSONResponse:
        # SQLite WAL-mode transactions are implicit; this is a no-op stub
        return JSONResponse({"status": "begun"})

    @adapter.post("/transaction/end")
    async def transaction_end(request: Request) -> JSONResponse:
        # Accept {"commit": true/false} but SQLite auto-commits
        return JSONResponse({"status": "ended"})

    return adapter


def main() -> None:
    parser = argparse.ArgumentParser(description="AWM Adapter Server")
    parser.add_argument("--scenario", required=True, help="AWM scenario name")
    parser.add_argument(
        "--envs-jsonl",
        type=Path,
        default=Path("data/awm-1k/gen_envs.jsonl"),
        help="Path to gen_envs.jsonl",
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=Path("external/awm-envs/outputs/databases"),
        help="Directory containing .db files",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/awm-1k"),
        help="Path to AWM-1K data directory",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--task-idx", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    app = build_adapter(
        scenario=args.scenario,
        envs_jsonl=args.envs_jsonl,
        db_dir=args.db_dir,
        data_path=args.data_path,
        task_idx=args.task_idx,
    )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
