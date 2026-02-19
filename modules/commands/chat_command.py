#!/usr/bin/env python3
"""
Chat command for the MeshCore Bot
Provides AI chatbot functionality via any OpenAI-compatible API endpoint
as a fallback for unrecognized messages.  When enabled, any message that
doesn't match a known command is forwarded to the configured LLM and the
response is sent back to the user.
"""

import re
import asyncio
import time
from collections import deque
from typing import Any, Dict, Optional
from .base_command import BaseCommand
from ..models import MeshMessage


class ChatCommand(BaseCommand):
    """Handles AI chat via an OpenAI-compatible API as a fallback for
    unrecognized messages.

    This command is special: it does NOT use keyword matching.  Instead, the
    message handler invokes it explicitly when no other command matched.
    The ``keywords`` list is intentionally empty so the normal plugin
    matching logic never triggers it.
    """

    # Plugin metadata
    name = "chat"
    keywords = []  # Empty — this is a fallback, not a keyword-triggered command
    description = "AI chatbot via OpenAI-compatible API (fallback for unrecognized messages)"
    category = "ai"
    cooldown_seconds = 0  # Per-user cooldown loaded from config
    requires_dm = False
    requires_internet = True  # Needs network access to reach the LLM API

    def __init__(self, bot: Any):
        super().__init__(bot)

        # ── Load LLM / OpenAI-compatible API configuration ─────
        self.chat_enabled = self.get_config_value(
            'Chat_Command', 'enabled', fallback=True, value_type='bool')

        self.api_url = self.get_config_value(
            'Chat_Command', 'api_url', fallback='')
        self.model = self.get_config_value(
            'Chat_Command', 'model', fallback='gpt-4o-mini')
        self.api_key = self.get_config_value(
            'Chat_Command', 'api_key', fallback='')
        self.system_prompt = self.get_config_value(
            'Chat_Command', 'system_prompt',
            fallback=(
                "You are a helpful assistant responding over a low-bandwidth "
                "radio mesh network. Keep every reply under 190 characters. "
                "Be concise and direct. No markdown formatting. "
                "Try to be casual, and avoid emojis as they will not render well."
            ))
        self.max_history = self.get_config_value(
            'Chat_Command', 'max_history', fallback=20, value_type='int')
        self.api_timeout = self.get_config_value(
            'Chat_Command', 'api_timeout', fallback=120, value_type='int')
        self.cooldown_seconds = self.get_config_value(
            'Chat_Command', 'cooldown_seconds', fallback=5, value_type='int')
        self.dm_only = self.get_config_value(
            'Chat_Command', 'dm_only', fallback=False, value_type='bool')

        # ── Per-user conversation history ──────────────────────
        # Maps sender_id -> deque of {"role": ..., "content": ...}
        self._histories: Dict[str, deque] = {}

        # Auto-disable chat if no API URL is configured
        if not self.api_url:
            self.chat_enabled = False
            self.logger.warning(
                "ChatCommand disabled: no api_url configured in [Chat_Command]")
        else:
            self.logger.info(
                f"ChatCommand initialised: model={self.model}, "
                f"url={self.api_url}, enabled={self.chat_enabled}")

    # ── Plugin interface overrides ─────────────────────────────

    def should_execute(self, message: MeshMessage) -> bool:
        """Never match via normal keyword scanning."""
        return False

    def get_response_format(self) -> Optional[str]:
        """No static response format — we generate responses dynamically."""
        return None

    # ── Public API (called from message_handler) ───────────────

    async def handle_fallback(self, message: MeshMessage) -> bool:
        """Process an unrecognized message through the LLM API.

        Called explicitly by the message handler when no command matched.

        Args:
            message: The incoming message.

        Returns:
            bool: True if a response was sent, False otherwise.
        """
        if not self.chat_enabled:
            return False

        # Respect DM-only setting
        if self.dm_only and not message.is_dm:
            return False

        # Check channel access
        if not self.is_channel_allowed(message):
            return False

        # Check per-user cooldown
        if self.cooldown_seconds > 0:
            can_exec, remaining = self.check_cooldown(
                message.sender_id if message.sender_id else None)
            if not can_exec:
                self.logger.debug(
                    f"Chat cooldown for {message.sender_id}: {remaining:.1f}s remaining")
                return False

        content = message.content.strip()
        if not content:
            return False

        sender = message.sender_id or "unknown"

        # Build / retrieve per-user history
        if sender not in self._histories:
            self._histories[sender] = deque(maxlen=self.max_history)

        history = self._histories[sender]
        history.append({"role": "user", "content": content})

        try:
            reply = await self._query_llm(list(history))
        except Exception as e:
            self.logger.error(f"LLM query failed: {e}")
            reply = self.translate('commands.chat.error', error=str(e))
            if reply == 'commands.chat.error':
                reply = f"Chat error: {e}"

        # Truncate to fit MeshCore message limits
        max_len = self.get_max_message_length(message)
        reply = self._truncate_reply(reply, max_len)

        # Store assistant reply in history
        history.append({"role": "assistant", "content": reply})

        # Record cooldown
        self.record_execution(message.sender_id if message.sender_id else None)

        # Store for stats / web viewer tracking
        self.last_response = reply

        # Send the response
        success = await self.send_response(message, reply)

        # Record in stats
        if 'stats' in self.bot.command_manager.commands:
            stats_cmd = self.bot.command_manager.commands['stats']
            if stats_cmd:
                stats_cmd.record_command(message, 'chat', success)

        return success

    async def execute(self, message: MeshMessage) -> bool:
        """Standard execute entry point (delegates to handle_fallback)."""
        return await self.handle_fallback(message)

    # ── LLM helpers ────────────────────────────────────────────

    async def _query_llm(self, history: list) -> str:
        """Call an OpenAI-compatible chat completions API in a background thread.

        Works with OpenAI, LM Studio, vLLM, LocalAI, and any other provider
        that implements the OpenAI chat completions contract.

        Args:
            history: List of message dicts for conversation context.

        Returns:
            str: The assistant's reply text.
        """
        import requests  # local import to avoid hard dependency when disabled

        def _blocking_call() -> str:
            messages = [
                {"role": "system", "content": self.system_prompt}
            ] + history
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
            }
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            resp = requests.post(
                self.api_url, json=payload, headers=headers,
                timeout=self.api_timeout)
            resp.raise_for_status()
            data = resp.json()
            # OpenAI-compatible response format
            answer = data["choices"][0]["message"]["content"]
            # Strip <think>…</think> blocks from thinking models
            answer = re.sub(
                r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
            return answer

        return await asyncio.to_thread(_blocking_call)

    @staticmethod
    def _truncate_reply(reply: str, max_bytes: int) -> str:
        """Truncate a reply so its UTF-8 encoding fits within *max_bytes*.

        Args:
            reply: The reply string to truncate.
            max_bytes: Maximum byte length.

        Returns:
            str: The (possibly truncated) reply.
        """
        if len(reply.encode("utf-8")) <= max_bytes:
            return reply
        while len(reply.encode("utf-8")) > max_bytes - 1:
            reply = reply[:-1]
        return reply.rstrip() + "…"
