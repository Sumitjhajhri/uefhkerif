"""The agent harness.

Two versions of the same agent, differing only in configuration:

  v1 - "helpful". Retrieves, then answers. The system prompt asks it to be
       useful and does not constrain it to the retrieved passages.
  v2 - "grounded". Must answer only from retrieved passages, must cite chunk
       ids, must declare a question unanswerable rather than guess. Its
       citations are then verified against the passages it actually received.

v1 exists to be measured, not shipped. The project's result is the difference
between the two on the same test set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .manifest import RunManifest, StageRecord
from .models import ModelBackend, ModelError
from .retriever import Retriever
from .tools import TOOLS, ToolError, run_search

V1_SYSTEM = """You are a helpful assistant answering questions about a set of documents.

Use the search_corpus tool to find relevant passages, then give the user a \
clear, complete answer using the `answer` tool. Be helpful and informative."""

V2_SYSTEM = """You are a careful research assistant answering questions strictly \
from a fixed document corpus.

Rules, in order of priority:

1. Answer ONLY from passages returned by search_corpus. Your own background \
knowledge is not evidence and must not appear in the answer, even when you are \
confident it is correct.
2. Every factual claim must be supported by a retrieved passage, and you must \
list the chunk ids of the supporting passages in `citations`.
3. If the retrieved passages do not contain the answer, call `answer` with \
answerable=false and say plainly that the documents do not cover it. This is a \
correct and expected outcome, not a failure. Do not substitute a general answer.
4. If a question is partly covered, answer only the covered part and say which \
part is not covered.

You may search more than once with different terms before concluding that \
something is absent."""


@dataclass
class AgentConfig:
    version: str = "v2"
    system_prompt: str = V2_SYSTEM
    verify_citations: bool = True
    max_iterations: int = 6
    top_k: int = 4
    context_char_budget: int = 6000

    @classmethod
    def v1(cls) -> "AgentConfig":
        return cls(
            version="v1",
            system_prompt=V1_SYSTEM,
            verify_citations=False,
        )

    @classmethod
    def v2(cls) -> "AgentConfig":
        return cls(
            version="v2",
            system_prompt=V2_SYSTEM,
            verify_citations=True,
        )


@dataclass
class AgentResult:
    question: str
    answer: str
    answerable: bool
    citations: list[str] = field(default_factory=list)
    unsupported_citations: list[str] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    manifest: RunManifest | None = None


class AgentRunError(RuntimeError):
    """The run could not complete. Distinct from a bad answer."""


class Agent:
    def __init__(
        self,
        backend: ModelBackend,
        retriever: Retriever,
        config: AgentConfig | None = None,
    ):
        self.backend = backend
        self.retriever = retriever
        self.config = config or AgentConfig.v2()

    def run(self, question: str) -> AgentResult:
        if not question or not question.strip():
            raise AgentRunError("question must be a non-empty string")

        cfg = self.config
        manifest = RunManifest(
            question=question,
            agent_version=cfg.version,
            backend=getattr(self.backend, "name", "unknown"),
        ).start()

        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        seen_chunk_ids: list[str] = []

        try:
            for iteration in range(cfg.max_iterations):
                with manifest.stage(f"model_call_{iteration}") as stage:
                    response = self.backend.complete(
                        system=cfg.system_prompt, messages=messages, tools=TOOLS
                    )
                    manifest.model_calls += 1
                    manifest.input_tokens += response.input_tokens
                    manifest.output_tokens += response.output_tokens
                    stage.detail = {
                        "stop_reason": response.stop_reason,
                        "tool_calls": [t.name for t in response.tool_calls],
                    }

                if not response.wants_tool:
                    # The model answered in prose instead of calling `answer`.
                    # Accepted, but recorded: an uncited answer cannot be
                    # citation-checked, and the grader treats it accordingly.
                    result = self._finalise(
                        manifest,
                        question,
                        answer=response.text.strip(),
                        answerable=True,
                        citations=[],
                        seen_chunk_ids=seen_chunk_ids,
                        note="model answered without calling the answer tool",
                    )
                    return result

                assistant_content: list[dict[str, Any]] = []
                if response.text:
                    assistant_content.append({"type": "text", "text": response.text})
                for call in response.tool_calls:
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }
                    )
                messages.append({"role": "assistant", "content": assistant_content})

                tool_results: list[dict[str, Any]] = []
                finished: AgentResult | None = None

                for call in response.tool_calls:
                    manifest.tool_calls.append(
                        {"name": call.name, "arguments": call.arguments}
                    )

                    if call.name == "search_corpus":
                        with manifest.stage("search") as stage:
                            try:
                                text, record = run_search(
                                    self.retriever,
                                    call.arguments,
                                    k=cfg.top_k,
                                    max_chars=cfg.context_char_budget,
                                )
                            except ToolError as exc:
                                # Tell the model what it did wrong and let it
                                # retry. A malformed tool call is a recoverable
                                # condition, not a crash.
                                text, record = f"Tool error: {exc}", []
                            stage.detail = {
                                "query": call.arguments.get("query"),
                                "hits": len(record),
                            }
                        manifest.retrieved.extend(record)
                        for entry in record:
                            if entry["chunk_id"] not in seen_chunk_ids:
                                seen_chunk_ids.append(entry["chunk_id"])
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": call.id,
                                "content": text,
                            }
                        )

                    elif call.name == "answer":
                        args = call.arguments
                        finished = self._finalise(
                            manifest,
                            question,
                            answer=str(args.get("answer", "")).strip(),
                            answerable=bool(args.get("answerable", True)),
                            citations=[
                                str(c) for c in (args.get("citations") or [])
                            ],
                            seen_chunk_ids=seen_chunk_ids,
                        )

                    else:
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": call.id,
                                "content": f"Unknown tool: {call.name}",
                                "is_error": True,
                            }
                        )

                if finished is not None:
                    return finished

                messages.append({"role": "user", "content": tool_results})

            raise AgentRunError(
                f"agent did not finish within {cfg.max_iterations} iterations"
            )

        except (ModelError, AgentRunError) as exc:
            manifest.finish("failed", error=f"{type(exc).__name__}: {exc}")
            raise AgentRunError(str(exc)) from exc

    def _finalise(
        self,
        manifest: RunManifest,
        question: str,
        answer: str,
        answerable: bool,
        citations: list[str],
        seen_chunk_ids: list[str],
        note: str | None = None,
    ) -> AgentResult:
        unsupported: list[str] = []
        if self.config.verify_citations:
            with manifest.stage("verify_citations") as stage:
                unsupported = [c for c in citations if c not in seen_chunk_ids]
                stage.detail = {
                    "cited": len(citations),
                    "unsupported": len(unsupported),
                }

        manifest.answer = answer
        manifest.citations = citations
        if note:
            manifest.stages.append(
                StageRecord(name="note", status="ok", detail={"note": note})
            )
        manifest.finish("ok")

        return AgentResult(
            question=question,
            answer=answer,
            answerable=answerable,
            citations=citations,
            unsupported_citations=unsupported,
            retrieved_chunk_ids=list(seen_chunk_ids),
            manifest=manifest,
        )
