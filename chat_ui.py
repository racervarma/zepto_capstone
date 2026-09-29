"""
Zepto Capstone - Support Assistant Web UI

A Gradio chat interface for the FastAPI Support Assistant.

This removes the need to use cURL in the terminal. The UI runs on port 7861
so it never conflicts with the FastAPI server on port 8000.

Start the backend first, in a separate terminal:

    uvicorn support_assistant.main:app --port 8000

Then start this UI:

    python chat_ui.py
"""

from pathlib import Path
import os
from urllib.parse import urlsplit

import gradio as gr
import requests


# ============================================================
# Configuration
# ============================================================

API_URL = os.getenv("SUPPORT_API_URL", "http://127.0.0.1:8000")
ASK_ENDPOINT = f"{API_URL}/ask"
HEALTH_ENDPOINT = f"{API_URL}/health"

UI_PORT = int(os.getenv("GRADIO_PORT", "7861"))

DEFAULT_API_PORT = 8000
DEFAULT_API_HOST = "127.0.0.1"

REQUEST_TIMEOUT = 30
SOURCES_PREFIX = "Sources: "


# ============================================================
# Backend connectivity
# ============================================================

def check_backend() -> bool:
    """Report whether the FastAPI server is reachable, without raising."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def backend_port() -> int:
    """
    Read the port out of the configured API URL.

    Falls back to DEFAULT_API_PORT when the URL omits a port or carries a
    non-numeric one, so the hint never renders a nonsense command.
    """
    try:
        return int(urlsplit(API_URL).port or DEFAULT_API_PORT)
    except (TypeError, ValueError):
        return DEFAULT_API_PORT


def start_instructions() -> str:
    """
    Build the message shown when the backend cannot be reached.

    The suggested command targets the port this UI is actually configured to
    call, so the hint stays correct when SUPPORT_API_URL is overridden. When
    that URL points somewhere other than the default local backend, the Docker
    command is offered as well, since a non-default port is usually a
    container mapping.
    """
    port = backend_port()

    message = (
        "Cannot reach the Support Assistant backend at "
        f"{API_URL}.\n\n"
        "Start it in a separate terminal, then try again:\n\n"
        f"    uvicorn support_assistant.main:app --port {port}"
    )

    if port != DEFAULT_API_PORT:
        message += (
            "\n\nIf the backend runs in Docker instead, start it with:\n\n"
            f"    docker run -p {port}:7860 zepto-support"
        )

    return message


# ============================================================
# Response formatting
# ============================================================

def format_sources(sources) -> str:
    """
    Render the source list as a single comma-separated line.

    Non-list values and non-string entries are ignored rather than iterated,
    so a malformed payload cannot produce one character per source.
    """
    if not isinstance(sources, list):
        return ""

    cleaned = [
        str(source).strip()
        for source in sources
        if isinstance(source, str) and source.strip()
    ]

    if not cleaned:
        return ""

    return SOURCES_PREFIX + ", ".join(cleaned)


def format_response(payload: dict) -> str:
    """
    Build the chatbot reply from the /ask response body.

    The answer is shown first, followed by a new line naming the sources
    used. When the backend returns no sources, the answer is shown alone.
    """
    answer = str(payload.get("answer", "")).strip()
    formatted_sources = format_sources(payload.get("sources"))

    if not formatted_sources:
        return answer

    if not answer:
        return formatted_sources

    return f"{answer}\n{formatted_sources}"


def parse_error(response) -> str:
    """Extract a human-readable message from a non-2xx response."""
    try:
        detail = response.json().get("detail")
    except ValueError:
        return response.text.strip() or f"HTTP {response.status_code}"

    if isinstance(detail, list):
        messages = [
            str(item.get("msg", item))
            for item in detail
            if isinstance(item, dict)
        ]
        detail = "; ".join(messages) or None

    if detail:
        return str(detail)

    return f"The assistant returned HTTP {response.status_code}."


# ============================================================
# Chat backend
# ============================================================

def chat(message: str, history: list) -> str:
    """
    Send a user message to the FastAPI /ask endpoint and return the reply.

    Every failure mode is handled so the user always receives an actionable
    message instead of a stack trace.
    """
    message = (message or "").strip()

    if not message:
        return "Please type a question."

    try:
        response = requests.post(
            ASK_ENDPOINT,
            json={"query": message},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.ConnectionError:
        return start_instructions()
    except requests.exceptions.Timeout:
        return (
            f"The Support Assistant did not respond within "
            f"{REQUEST_TIMEOUT} seconds. It may still be loading the "
            "embedding model on first run, or the backend may be "
            f"unresponsive. Backend: {API_URL}"
        )
    except requests.exceptions.RequestException as error:
        return f"Could not reach the Support Assistant: {error}"

    if response.status_code >= 400:
        return f"The assistant returned an error: {parse_error(response)}"

    try:
        payload = response.json()
    except ValueError:
        return (
            "The assistant returned a response that was not valid JSON. "
            f"Backend: {API_URL}"
        )

    if not isinstance(payload, dict):
        return (
            "The assistant returned an unexpected response structure. "
            f"Backend: {API_URL}"
        )

    return format_response(payload)


# ============================================================
# Interface
# ============================================================

def build_interface() -> gr.ChatInterface:
    """Assemble the Gradio chat interface."""
    return gr.ChatInterface(
        fn=chat,
        title="Zepto Support Assistant",
        description=(
            "Ask a question about Zepto policies: delivery, returns and "
            "refunds, membership, order tracking, cancellation, damaged "
            "items, gift cards, or support hours."
        ),
        examples=[
            "What is the delivery fee for orders under INR 149?",
            "How do I return a damaged item?",
            "What are the Zepto membership tiers?",
            "Can I cancel an order after it is packed?",
            "How long is a gift card valid?",
            "What are your support hours?",
        ],
    )


def main() -> None:
    """Preflight the backend connection and launch the UI."""
    print("=" * 72)
    print("Zepto Support Assistant - Web UI")
    print("=" * 72)
    print(f"Backend : {API_URL}")
    print(f"UI      : http://127.0.0.1:{UI_PORT}")
    print()

    if check_backend():
        print(f"Backend is up at {API_URL}.")
    else:
        print(f"WARNING: no response from {API_URL}.")
        print("The UI will still start so you can launch the backend")
        print("and retry without reloading this page.")
        print()
        print(f"    uvicorn support_assistant.main:app --port {backend_port()}")
        print()

    build_interface().launch(
        server_name="127.0.0.1",
        server_port=UI_PORT,
    )


if __name__ == "__main__":
    main()
