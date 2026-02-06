"""GitHub App webhook handler for swarm-ai-bot."""

import hashlib
import hmac
import json
import os
import time

import jwt
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)
app.url_map.strict_slashes = False

# GitHub App credentials (set these in Railway environment variables)
GITHUB_APP_ID = os.environ.get("GITHUB_APP_ID")
# Handle various newline formats in private key
_raw_key = os.environ.get("GITHUB_PRIVATE_KEY", "")
_base64_key = os.environ.get("GITHUB_PRIVATE_KEY_BASE64", "")

if _base64_key:
    import base64
    GITHUB_PRIVATE_KEY = base64.b64decode(_base64_key).decode("utf-8")
else:
    GITHUB_PRIVATE_KEY = _raw_key.replace("\\n", "\n").replace("\\r", "").strip()
GITHUB_WEBHOOK_SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET", "")


def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify GitHub webhook signature."""
    if not signature or not secret:
        return False
    expected = "sha256=" + hmac.new(
        secret.encode(), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def get_jwt_token() -> str:
    """Generate a JWT for GitHub App authentication."""
    if not GITHUB_APP_ID:
        raise ValueError("GITHUB_APP_ID not set")
    if not GITHUB_PRIVATE_KEY:
        raise ValueError("GITHUB_PRIVATE_KEY not set")

    # Debug: log key format (first 50 chars only for security)
    print(f"DEBUG: App ID = {GITHUB_APP_ID}")
    print(f"DEBUG: Key starts with = {GITHUB_PRIVATE_KEY[:50]}...")
    print(f"DEBUG: Key length = {len(GITHUB_PRIVATE_KEY)}")

    now = int(time.time())
    payload = {
        "iat": now - 60,  # Issued 60 seconds ago
        "exp": now + (10 * 60),  # Expires in 10 minutes
        "iss": GITHUB_APP_ID,
    }
    return jwt.encode(payload, GITHUB_PRIVATE_KEY, algorithm="RS256")


def get_installation_token(installation_id: int) -> str:
    """Get an installation access token for a specific installation."""
    jwt_token = get_jwt_token()
    response = requests.post(
        f"https://api.github.com/app/installations/{installation_id}/access_tokens",
        headers={
            "Authorization": f"Bearer {jwt_token}",
            "Accept": "application/vnd.github+json",
        },
    )
    response.raise_for_status()
    return response.json()["token"]


def post_comment(repo_full_name: str, issue_number: int, body: str, installation_id: int) -> dict:
    """Post a comment on an issue or pull request."""
    token = get_installation_token(installation_id)
    response = requests.post(
        f"https://api.github.com/repos/{repo_full_name}/issues/{issue_number}/comments",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
        json={"body": body},
    )
    response.raise_for_status()
    return response.json()


@app.route("/", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "swarm-ai-bot"})


@app.route("/api/github-webhook", methods=["GET", "POST"])
def webhook():
    """Handle GitHub webhook requests."""
    # Health check
    if request.method == "GET":
        return jsonify({"status": "ok", "service": "swarm-ai-bot webhook"})

    # Webhook POST
    payload = request.get_data()

    # Verify signature
    signature = request.headers.get("X-Hub-Signature-256", "")

    if GITHUB_WEBHOOK_SECRET and not verify_signature(payload, signature, GITHUB_WEBHOOK_SECRET):
        return jsonify({"error": "Invalid signature"}), 401

    # Parse payload
    data = json.loads(payload)
    event_type = request.headers.get("X-GitHub-Event", "")

    # Handle ping event
    if event_type == "ping":
        return jsonify({"message": "pong"})

    # Handle issue events
    if event_type == "issues":
        action = data.get("action")
        issue = data.get("issue", {})
        repo = data.get("repository", {})
        installation = data.get("installation", {})

        # Only respond to newly opened issues with specific labels or keywords
        if action == "opened":
            issue_title = issue.get("title", "").lower()
            issue_body = issue.get("body", "") or ""

            # Check if this is a feature request or needs bot response
            if "feature request" in issue_title or "[bot]" in issue_body.lower():
                try:
                    comment_body = (
                        "Thanks for opening this issue! 🤖\n\n"
                        "I'm the swarm-ai-bot. A maintainer will review this soon.\n\n"
                        "---\n"
                        "*This is an automated response.*"
                    )
                    post_comment(
                        repo.get("full_name"),
                        issue.get("number"),
                        comment_body,
                        installation.get("id"),
                    )
                except Exception as e:
                    return jsonify({"error": str(e)}), 500

        return jsonify({"event": "issues", "action": action, "status": "processed"})

    # Handle issue comment events
    if event_type == "issue_comment":
        action = data.get("action")
        comment = data.get("comment", {})
        issue = data.get("issue", {})
        repo = data.get("repository", {})
        installation = data.get("installation", {})

        # Respond to comments that mention the bot
        if action == "created":
            comment_body = comment.get("body", "") or ""
            commenter = comment.get("user", {}).get("login", "")

            # Don't respond to our own comments
            if "[bot]" in commenter:
                return jsonify({"status": "ignored", "reason": "bot comment"})

            # Check if bot is mentioned
            if "@swarm-ai-bot" in comment_body.lower():
                try:
                    reply = (
                        f"Hi @{commenter}! 👋\n\n"
                        "I'm here to help. A maintainer will follow up shortly.\n\n"
                        "---\n"
                        "*This is an automated response from swarm-ai-bot.*"
                    )
                    post_comment(
                        repo.get("full_name"),
                        issue.get("number"),
                        reply,
                        installation.get("id"),
                    )
                except Exception as e:
                    import traceback
                    print(f"ERROR: {e}")
                    print(traceback.format_exc())
                    return jsonify({"error": str(e), "type": type(e).__name__}), 500

        return jsonify({"event": "issue_comment", "action": action, "status": "processed"})

    # Acknowledge other events
    return jsonify({"event": event_type, "status": "received"})


# Manual trigger endpoint for posting comments
@app.route("/api/post-comment", methods=["POST"])
def manual_post_comment():
    """Manually post a comment (requires auth)."""
    # Simple API key auth
    api_key = request.headers.get("X-API-Key", "")
    expected_key = os.environ.get("BOT_API_KEY", "")

    if not expected_key or api_key != expected_key:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json
    repo = data.get("repo")  # e.g., "swarm-ai-safety/swarm"
    issue_number = data.get("issue_number")
    body = data.get("body")
    installation_id = data.get("installation_id")

    if not all([repo, issue_number, body, installation_id]):
        return jsonify({"error": "Missing required fields: repo, issue_number, body, installation_id"}), 400

    try:
        result = post_comment(repo, issue_number, body, installation_id)
        return jsonify({"status": "posted", "comment_id": result.get("id")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
