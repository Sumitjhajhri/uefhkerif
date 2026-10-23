"""Model backends.

The agent talks to a `ModelBackend`, never to a vendor SDK directly. There are
two implementations:

  AnthropicBackend  - a real API call. Used for real runs.
  ScriptedBackend   - returns a fixed list of responses in order. Used by the
                      test suite.

The reason for the split is that an evaluation harness whose own tests depend
on a live model is not a harness you can trust: it would be non-deterministic,
cost money per run, and fail when the network does. With ScriptedBackend the
test suite proves that the *graders* behave correctly against known-good and
known-bad answers, which is the part that has to be right.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Protocol


class ModelError(RuntimeError):
    """Raised when a backend cannot produce a response.

    Deliberately distinct from a bad *answer*: a bad answer is data the eval
    harness grades, a ModelError is an execution failure the run must record
    and stop on.
    """


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)


class ModelBackend(Protocol):
    name: str

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse: ...


class ScriptedBackend:
    """Returns pre-written responses in order.

    Every call is recorded, so a test can assert not only on the final answer
    but on what the agent actually sent - including whether the retrieved
    context made it into the prompt at all.

    The recording is a deep copy. The agent appends to its message list in
    place, so storing the list by reference would make every recorded call
    show the *final* state of the conversation rather than the state at the
    time of that call. A log that changes after the fact is worse than no log:
    it is confidently wrong about what happened, which is the same defect this
    project exists to measure.
    """

    name = "scripted"

    def __init__(self, responses: list[ModelResponse]):
        self._responses = list(responses)
        self._i = 0
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        self.calls.append(
            {
                "system": system,
                "messages": copy.deepcopy(messages),
                "tools": tools,
            }
        )
        if self._i >= len(self._responses):
            # A script running out means the agent looped more times than the
            # test expected. Failing loudly here beats returning a default and
            # letting the test pass for the wrong reason.
            raise ModelError(
                f"ScriptedBackend exhausted after {self._i} calls; "
                "the agent made more model calls than the script provides"
            )
        response = self._responses[self._i]
        self._i += 1
        return response


class AnthropicBackend:
    """Live Anthropic API.

    Requires ANTHROPIC_API_KEY. Imported lazily so the package, its tests and
    the offline parts of the eval all work on a machine with no SDK installed
    and no key configured.
    """

    name = "anthropic"

    def __init__(self, model: str = "claude-haiku-5-5", max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ModelError(
                    "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and "
                    "fill it in, or run with --backend scripted."
                )
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - env dependent
                raise ModelError(
                    "The 'anthropic' package is not installed. "
                    "Run: pip install -r requirements.txt"
                ) from exc
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        try:
            raw = client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - vendor errors are not a stable type
            raise ModelError(f"Anthropic API call failed: {exc}") from exc

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in raw.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        return ModelResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=raw.stop_reason or "end_turn",
            input_tokens=raw.usage.input_tokens,
            output_tokens=raw.usage.output_tokens,
        )


def get_backend(name: str, **kwargs: Any) -> ModelBackend:
    if name == "anthropic":
        return AnthropicBackend(**kwargs)
    raise ValueError(
        f"Unknown backend {name!r}. Use 'anthropic', or construct a "
        "ScriptedBackend directly in tests."
    )
