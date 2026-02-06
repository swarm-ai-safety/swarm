"""GitHub App webhook handler for swarm-ai-bot."""

import hashlib
import hmac
import json
import os

from flask import Flask, request, jsonify

app = Flask(__name__)


def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify GitHub webhook signature."""
    if not signature or not secret:
        return False
    expected = "sha256=" + hmac.new(
        secret.encode(), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


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
    webhook_secret = os.environ.get("GITHUB_WEBHOOK_SECRET", "")

    if webhook_secret and not verify_signature(payload, signature, webhook_secret):
        return jsonify({"error": "Invalid signature"}), 401

    # Get event type
    event_type = request.headers.get("X-GitHub-Event", "")

    # Handle ping event
    if event_type == "ping":
        return jsonify({"message": "pong"})

    # Acknowledge other events
    return jsonify({"event": event_type, "status": "received"})
