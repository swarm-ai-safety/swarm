"""
SEO API client library for swarm-ai.org.

Wraps Keywords Everywhere, DataForSEO, and GitHub CMS APIs.
All clients are lazy — they only fail when you call a method, not on import.
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Keywords Everywhere
# ---------------------------------------------------------------------------

@dataclass
class KeywordsEverywhereClient:
    """Thin wrapper around Keywords Everywhere v1 API."""

    api_key: str = field(default_factory=lambda: os.environ.get("KEYWORDS_EVERYWHERE_API_KEY", ""))
    base_url: str = "https://api.keywordseverywhere.com/v1"
    _session: requests.Session = field(default_factory=requests.Session, repr=False)

    def __post_init__(self) -> None:
        self._session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def _require_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "KEYWORDS_EVERYWHERE_API_KEY not set. "
                "Add it to .env or pass api_key= to the constructor."
            )

    # -- endpoints --

    def get_volume(self, keywords: list[str], country: str = "us", currency: str = "USD") -> dict:
        """Bulk keyword volume + CPC + competition."""
        self._require_key()
        resp = self._session.post(
            f"{self.base_url}/get_keyword_data",
            json={"kw": keywords, "country": country, "currency": currency},
        )
        resp.raise_for_status()
        return resp.json()

    def related_keywords(self, keyword: str, country: str = "us") -> dict:
        """Related keywords for a seed."""
        self._require_key()
        resp = self._session.post(
            f"{self.base_url}/get_related_keywords",
            json={"kw": [keyword], "country": country},
        )
        resp.raise_for_status()
        return resp.json()

    def people_also_search(self, keyword: str, country: str = "us") -> dict:
        """People Also Search For (PASF) expansion."""
        self._require_key()
        resp = self._session.post(
            f"{self.base_url}/get_pasf_keywords",
            json={"kw": [keyword], "country": country},
        )
        resp.raise_for_status()
        return resp.json()

    def expand_seeds(
        self, seeds: list[str], country: str = "us", max_per_seed: int = 50
    ) -> list[dict]:
        """Expand a list of seed keywords via related + PASF, deduplicated."""
        self._require_key()
        seen: set[str] = set()
        results: list[dict] = []
        for seed in seeds:
            for method in (self.related_keywords, self.people_also_search):
                data = method(seed, country=country)
                for kw_obj in data.get("data", [])[:max_per_seed]:
                    kw = kw_obj.get("keyword", kw_obj.get("kw", "")).lower().strip()
                    if kw and kw not in seen:
                        seen.add(kw)
                        results.append(kw_obj)
                time.sleep(0.3)  # respect rate limits
        return results


# ---------------------------------------------------------------------------
# DataForSEO
# ---------------------------------------------------------------------------

@dataclass
class DataForSEOClient:
    """Wrapper around DataForSEO REST API (SERP, Backlinks, Domain Intersection)."""

    login: str = field(default_factory=lambda: os.environ.get("DATAFORSEO_LOGIN", ""))
    password: str = field(default_factory=lambda: os.environ.get("DATAFORSEO_PASSWORD", ""))
    base_url: str = "https://api.dataforseo.com/v3"
    _session: requests.Session = field(default_factory=requests.Session, repr=False)

    def __post_init__(self) -> None:
        if self.login and self.password:
            creds = base64.b64encode(f"{self.login}:{self.password}".encode()).decode()
            self._session.headers.update({"Authorization": f"Basic {creds}"})

    def _require_creds(self) -> None:
        if not self.login or not self.password:
            raise RuntimeError(
                "DATAFORSEO_LOGIN / DATAFORSEO_PASSWORD not set. "
                "Add them to .env or pass login=/password= to the constructor."
            )

    def _post(self, endpoint: str, payload: list[dict]) -> dict:
        self._require_creds()
        resp = self._session.post(f"{self.base_url}/{endpoint}", json=payload)
        resp.raise_for_status()
        return resp.json()

    # -- SERP --

    def serp_google(
        self,
        keywords: list[str],
        location: str = "United States",
        language: str = "en",
        depth: int = 10,
    ) -> dict:
        """Google organic SERP for a list of keywords (live mode)."""
        tasks = [
            {
                "keyword": kw,
                "location_name": location,
                "language_name": language,
                "depth": depth,
            }
            for kw in keywords
        ]
        return self._post("serp/google/organic/live/advanced", tasks)

    # -- Backlinks --

    def backlinks_summary(self, target: str) -> dict:
        """Backlink summary for a domain."""
        return self._post("backlinks/summary/live", [{"target": target}])

    def backlinks_referring_domains(self, target: str, limit: int = 100) -> dict:
        """List referring domains for a target."""
        return self._post(
            "backlinks/referring_domains/live",
            [{"target": target, "limit": limit}],
        )

    def domain_intersection(
        self, targets: list[str], exclude: str, limit: int = 100
    ) -> dict:
        """
        Find domains linking to any target but NOT to `exclude`.
        Useful for competitor link gap analysis.
        """
        tasks = [{
            "targets": targets,
            "exclude_targets": [exclude],
            "limit": limit,
        }]
        return self._post("backlinks/domain_intersection/live", tasks)

    # -- Keywords Data --

    def keyword_suggestions(
        self, keyword: str, location: str = "United States", language: str = "en"
    ) -> dict:
        """Keyword suggestions from DataForSEO."""
        return self._post(
            "keywords_data/google_ads/keywords_for_keywords/live",
            [{"keywords": [keyword], "location_name": location, "language_name": language}],
        )


# ---------------------------------------------------------------------------
# GitHub CMS (MkDocs publish via git)
# ---------------------------------------------------------------------------

@dataclass
class GitHubCMSClient:
    """Publish MkDocs content by committing Markdown to the swarm repo."""

    token: str = field(default_factory=lambda: os.environ.get("GITHUB_TOKEN", ""))
    owner: str = field(default_factory=lambda: os.environ.get("GITHUB_REPO_OWNER", "swarm-ai-safety"))
    repo: str = field(default_factory=lambda: os.environ.get("GITHUB_REPO_NAME", "swarm"))
    branch: str = field(default_factory=lambda: os.environ.get("GITHUB_BRANCH", "main"))
    docs_path: str = field(default_factory=lambda: os.environ.get("MKDOCS_DOCS_PATH", "docs/"))
    _session: requests.Session = field(default_factory=requests.Session, repr=False)

    def __post_init__(self) -> None:
        if self.token:
            self._session.headers.update({
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json",
            })

    @property
    def api_base(self) -> str:
        return f"https://api.github.com/repos/{self.owner}/{self.repo}"

    def get_file(self, path: str) -> dict | None:
        """Get file content + SHA (needed for updates)."""
        resp = self._session.get(f"{self.api_base}/contents/{path}", params={"ref": self.branch})
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def upsert_file(self, path: str, content: str, message: str) -> dict:
        """Create or update a file in the repo."""
        existing = self.get_file(path)
        payload: dict[str, Any] = {
            "message": message,
            "content": base64.b64encode(content.encode()).decode(),
            "branch": self.branch,
        }
        if existing:
            payload["sha"] = existing["sha"]
        resp = self._session.put(f"{self.api_base}/contents/{path}", json=payload)
        resp.raise_for_status()
        return resp.json()

    def publish_page(self, rel_path: str, markdown: str, commit_msg: str | None = None) -> dict:
        """
        Publish a Markdown page to the docs site.
        rel_path is relative to docs/ (e.g. "guides/metrics/toxicity.md").
        """
        full_path = f"{self.docs_path.rstrip('/')}/{rel_path}"
        msg = commit_msg or f"seo: publish {rel_path}"
        return self.upsert_file(full_path, markdown, msg)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def get_clients() -> dict[str, Any]:
    """Return all clients. Safe to call even without API keys — clients fail lazily."""
    return {
        "ke": KeywordsEverywhereClient(),
        "dseo": DataForSEOClient(),
        "cms": GitHubCMSClient(),
    }
