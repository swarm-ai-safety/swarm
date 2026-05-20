"""Gitlawb GraphQL client with query and WebSocket subscription support."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine, Optional

from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport
from gql.transport.websockets import WebsocketsTransport

from swarm.bridges.gitlawb.config import GitlawbConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GraphQL documents
# ---------------------------------------------------------------------------

SUB_REF_UPDATES = gql(
    """
    subscription OnRefUpdate($repo: String) {
        refUpdates(repo: $repo) {
            repo
            refName
            oldSha
            newSha
            pusherDid
            nodeDid
            timestamp
        }
    }
    """
)

SUB_TASK_EVENTS = gql(
    """
    subscription OnTaskEvent($taskId: String) {
        taskEvents(taskId: $taskId) {
            taskId
            oldStatus
            newStatus
            byDid
            at
        }
    }
    """
)

QUERY_TASKS = gql(
    """
    query GetTasks($status: String, $assigneeDid: String, $limit: Int) {
        tasks(status: $status, assigneeDid: $assigneeDid, limit: $limit) {
            id repoId kind status delegatorDid assigneeDid
            capability ucanToken payload result createdAt updatedAt deadline
        }
    }
    """
)

QUERY_TASK = gql(
    """
    query GetTask($id: String!) {
        task(id: $id) {
            id repoId kind status delegatorDid assigneeDid
            capability ucanToken payload result createdAt updatedAt deadline
        }
    }
    """
)

QUERY_REPOS = gql(
    """
    query GetRepos {
        repos {
            name
            ownerDid
            description
            defaultBranch
            createdAt
        }
    }
    """
)

QUERY_REF_UPDATES = gql(
    """
    query GetRefUpdates($repo: String, $limit: Int) {
        refUpdates(repo: $repo, limit: $limit) {
            repo refName oldSha newSha pusherDid nodeDid timestamp
        }
    }
    """
)

# Type alias for async event callbacks
EventCallback = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class GitlawbClient:
    """Encapsulates all Gitlawb GraphQL communication."""

    def __init__(self, config: GitlawbConfig) -> None:
        self._config = config
        self._http_url = f"{config.node_url}/graphql"
        self._ws_url = config.ws_url
        self._active_subscriptions: dict[str, asyncio.Task] = {}
        self._http_client: Optional[Client] = None

    async def _get_http_client(self) -> Client:
        """Lazily create the HTTP GraphQL client."""
        if self._http_client is None:
            transport = AIOHTTPTransport(url=self._http_url)
            self._http_client = Client(
                transport=transport,
                fetch_schema_from_transport=False,
            )
        return self._http_client

    async def execute_query(
        self,
        document: Any,
        variables: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Execute a GraphQL query/mutation via HTTP."""
        client = await self._get_http_client()
        async with client as session:
            result = await session.execute(document, variable_values=variables)
            return result

    async def subscribe_ref_updates(
        self,
        repo: Optional[str],
        callback: EventCallback,
    ) -> asyncio.Task:
        """Subscribe to ref update events. Returns the background task."""
        task = asyncio.create_task(
            self._subscribe_with_retry(
                document=SUB_REF_UPDATES,
                variables={"repo": repo},
                callback=callback,
                sub_name=f"refUpdates:{repo or '*'}",
            )
        )
        self._active_subscriptions[f"refUpdates:{repo or '*'}"] = task
        return task

    async def subscribe_task_events(
        self,
        task_id: Optional[str],
        callback: EventCallback,
    ) -> asyncio.Task:
        """Subscribe to task events. Returns the background task."""
        task = asyncio.create_task(
            self._subscribe_with_retry(
                document=SUB_TASK_EVENTS,
                variables={"taskId": task_id},
                callback=callback,
                sub_name=f"taskEvents:{task_id or '*'}",
            )
        )
        self._active_subscriptions[f"taskEvents:{task_id or '*'}"] = task
        return task

    async def _subscribe_with_retry(
        self,
        document: Any,
        variables: Optional[dict[str, Any]],
        callback: EventCallback,
        sub_name: str,
    ) -> None:
        """Subscribe with automatic reconnection on disconnect."""
        attempt = 0
        while True:
            try:
                transport = WebsocketsTransport(url=self._ws_url)
                async with Client(
                    transport=transport,
                    fetch_schema_from_transport=False,
                ) as session:
                    logger.info("Subscription %s connected", sub_name)
                    async for event in session.subscribe(
                        document, variable_values=variables
                    ):
                        await callback(event)
                    logger.info("Subscription %s ended normally", sub_name)
                    break
            except asyncio.CancelledError:
                logger.info("Subscription %s cancelled", sub_name)
                raise
            except Exception as exc:
                attempt += 1
                if (
                    self._config.max_reconnect_attempts > 0
                    and attempt >= self._config.max_reconnect_attempts
                ):
                    logger.error(
                        "Max reconnect attempts reached for %s: %s",
                        sub_name,
                        exc,
                    )
                    raise
                delay = min(
                    self._config.reconnect_delay_sec * (2 ** attempt),
                    60.0,
                )
                logger.warning(
                    "Subscription %s disconnected (%s). Reconnecting in %.1fs",
                    sub_name,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

    async def fetch_tasks(
        self,
        status: Optional[str] = None,
        assignee_did: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch tasks via HTTP query."""
        result = await self.execute_query(
            QUERY_TASKS,
            {"status": status, "assigneeDid": assignee_did, "limit": limit},
        )
        return result.get("tasks", [])

    async def fetch_task(self, task_id: str) -> Optional[dict[str, Any]]:
        """Fetch a single task by ID."""
        result = await self.execute_query(QUERY_TASK, {"id": task_id})
        return result.get("task")

    async def fetch_repos(self) -> list[dict[str, Any]]:
        """List all repos on the node."""
        result = await self.execute_query(QUERY_REPOS)
        return result.get("repos", [])

    async def fetch_historical_ref_updates(
        self,
        repo: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch historical ref updates for backfill."""
        result = await self.execute_query(
            QUERY_REF_UPDATES,
            {"repo": repo, "limit": limit},
        )
        return result.get("refUpdates", [])

    async def close(self) -> None:
        """Cancel all subscriptions and close connections."""
        for name, task in self._active_subscriptions.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            logger.info("Closed subscription %s", name)
        self._active_subscriptions.clear()
        if self._http_client:
            await self._http_client.close_async()
            self._http_client = None
