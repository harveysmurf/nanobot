"""Webhook channel — receives HTTP POST requests and returns agent responses.

Designed for Home Assistant's Webhook Conversation integration but works
with any client that POSTs JSON with a ``query`` field and expects a JSON
response with an ``output`` field.

Request-response flow:
  1. Client POSTs ``{"query": "...", ...}`` to the webhook
  2. Channel generates a unique ``chat_id`` and creates an asyncio Future
  3. Message is forwarded to the agent via ``_handle_message()``
  4. ``send()`` is called by the channel manager when the agent replies —
     it resolves the Future with the response text
  5. The HTTP handler awaits the Future and returns ``{"output": "..."}``
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from aiohttp import web
from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel


_VOICE_TAG = (
    "[Voice command via Home Assistant]\n"
    "RULES: This is spoken via TTS. Reply in 1-2 short sentences only. "
    "No markdown, no asterisks, no bullet points, no emojis, no formatting. "
    "Use plain text only. If the user writes in Cyrillic, respond in BULGARIAN (not Russian). Respond in the SAME LANGUAGE the user spoke."
)

_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")


def _detect_lang(text: str) -> str:
    """Detect language from text: Cyrillic-heavy -> 'bg', otherwise 'en'."""
    cyrillic = len(_CYRILLIC_RE.findall(text))
    latin = len(re.findall(r"[a-zA-Z]", text))
    return "bg" if cyrillic > latin else "en"


class WebhookChannel(BaseChannel):
    """HTTP webhook channel with synchronous request-response semantics."""

    name = "webhook"
    display_name = "Webhook"

    def __init__(self, config: Any, bus: MessageBus) -> None:
        super().__init__(config, bus)
        # Pending request futures keyed by chat_id
        self._pending: dict[str, asyncio.Future[str]] = {}
        # Detected language per chat_id, used to tag the response for TTS
        self._lang: dict[str, str] = {}
        self._runner: web.AppRunner | None = None

    def is_allowed(self, sender_id: str) -> bool:
        """Check allowFrom — supports both dict and object config."""
        if isinstance(self.config, dict):
            allow_list = self.config.get("allowFrom", self.config.get("allow_from", []))
        else:
            allow_list = getattr(self.config, "allow_from", [])
        if not allow_list:
            return False
        if "*" in allow_list:
            return True
        return str(sender_id) in allow_list

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {
            "enabled": False,
            "port": 9100,
            "host": "0.0.0.0",
            "path": "/",
            "timeout": 120,
            "allowFrom": ["*"],
            "outputField": "output",
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the HTTP server and block until stopped."""
        self._running = True

        port = self.config.get("port", 9100) if isinstance(self.config, dict) else getattr(self.config, "port", 9100)
        host = self.config.get("host", "0.0.0.0") if isinstance(self.config, dict) else getattr(self.config, "host", "0.0.0.0")
        path = self.config.get("path", "/") if isinstance(self.config, dict) else getattr(self.config, "path", "/")

        app = web.Application()
        app.router.add_post(path, self._on_request)
        # Also listen on /webhook for HA Webhook Conversation compatibility
        if path != "/webhook":
            app.router.add_post("/webhook", self._on_request)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, host, port)
        await site.start()

        logger.info("Webhook listening on http://{}:{}{}", host, port, path)

        # Block until stopped — required by BaseChannel contract
        while self._running:
            await asyncio.sleep(1)

        await self._runner.cleanup()

    async def stop(self) -> None:
        self._running = False
        # Cancel any pending request futures
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()

    # ------------------------------------------------------------------
    # Inbound (HTTP POST → agent)
    # ------------------------------------------------------------------

    async def _on_request(self, request: web.Request) -> web.Response:
        """Handle an incoming webhook POST request."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        # Support both "query" (HA Webhook Conversation) and "text" (generic)
        query = body.get("query") or body.get("text", "")
        if not query:
            return web.json_response({"error": "Missing 'query' field"}, status=400)

        # Always use fixed chat_id for HA sessions - single persistent conversation
        chat_id = "ha-voice"
        sender_ip = request.remote or "unknown"
        output_field = (
            self.config.get("outputField", "output")
            if isinstance(self.config, dict)
            else getattr(self.config, "output_field", "output")
        )
        timeout = (
            self.config.get("timeout", 120)
            if isinstance(self.config, dict)
            else getattr(self.config, "timeout", 120)
        )

        logger.debug("Webhook request from {}: {}", sender_ip, query[:120])

        # Build metadata from the HA payload
        metadata: dict[str, Any] = {}
        for key in ("language", "conversation_id", "messages", "system_prompt",
                     "agent_id", "device_id", "device_info", "exposed_entities",
                     "user_id", "stream"):
            if key in body:
                metadata[key] = body[key]

        # Detect language from the user's query for TTS voice selection
        detected_lang = _detect_lang(query)
        self._lang[chat_id] = detected_lang

        content = f"{_VOICE_TAG}\n{query}"

        # Create a future to wait for the agent's response
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending[chat_id] = future

        try:
            await self._handle_message(
                sender_id=sender_ip,
                chat_id=chat_id,
                content=content,
                metadata=metadata,
            )

            response_text = await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Webhook request timed out after {}s", timeout)
            return web.json_response(
                {output_field: "Sorry, the request timed out."},
                status=200,
            )
        except Exception as e:
            logger.error("Webhook request failed: {}", e)
            return web.json_response(
                {output_field: "Sorry, something went wrong."},
                status=200,
            )
        finally:
            self._pending.pop(chat_id, None)
            lang = self._lang.pop(chat_id, None)

        # Prepend [lang:xx] tag so Smart TTS can pick the right voice
        if lang:
            response_text = f"[lang:{lang}]{response_text}"

        return web.json_response({output_field: response_text})

    # ------------------------------------------------------------------
    # Outbound (agent → HTTP response)
    # ------------------------------------------------------------------

    async def send(self, msg: OutboundMessage) -> None:
        """Resolve the pending future for the given chat_id.

        The channel manager calls this for every outbound message targeted
        at the webhook channel.  Progress and stream-delta messages are
        skipped — only the final response resolves the HTTP request.
        """
        meta = msg.metadata or {}
        if meta.get("_progress") or meta.get("_stream_delta"):
            return

        future = self._pending.get(msg.chat_id)
        if future and not future.done():
            future.set_result(msg.content)
        else:
            logger.debug(
                "Webhook send: no pending request for chat_id={}",
                msg.chat_id[:12],
            )

    async def send_delta(
        self, chat_id: str, delta: str, metadata: dict[str, Any] | None = None,
    ) -> None:
        """Handle streaming deltas — resolve the future on stream end."""
        meta = metadata or {}
        if meta.get("_stream_end"):
            future = self._pending.get(chat_id)
            if future and not future.done():
                future.set_result(delta)
