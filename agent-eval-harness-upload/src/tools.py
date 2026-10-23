"""Tool definitions and dispatch.

Two tools. `search_corpus` is how the agent gets evidence; `answer` is how it
finishes. Making the final answer a *tool call* rather than free text is a
design choice worth defending: it forces the citation list to be a structured
field the grader can check mechanically, instead of prose a regex has to guess
at. If the model wants to answer, it has to say which chunks it used.
"""

from __future__ import annotations

from typing import Any

from .retriever import Retriever

SEARCH_TOOL: dict[str, Any] = {
    "name": "search_corpus",
    "description": (
        "Search the document corpus for passages relevant to a query. "
        "Returns passages with their chunk ids. May return no results, which "
        "means the corpus does not cover the query."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search terms. Prefer specific nouns from the question.",
            }
        },
        "required": ["query"],
    },
}

ANSWER_TOOL: dict[str, Any] = {
    "name": "answer",
    "description": (
        "Give the final answer. Use `citations` to list the chunk ids that "
        "support it. If the corpus does not contain the answer, set "
        "`answerable` to false and leave citations empty."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "answerable": {
                "type": "boolean",
                "description": (
                    "False if the retrieved passages do not contain the answer."
                ),
            },
            "answer": {
                "type": "string",
                "description": (
                    "The answer, or a short statement that the documents do "
                    "not cover the question."
                ),
            },
            "citations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Chunk ids supporting the answer, e.g. ['handbook#2'].",
            },
        },
        "required": ["answerable", "answer"],
    },
}

TOOLS: list[dict[str, Any]] = [SEARCH_TOOL, ANSWER_TOOL]


class ToolError(RuntimeError):
    """A tool was called in a way the agent must be told about, not crash on."""


def run_search(
    retriever: Retriever, arguments: dict[str, Any], k: int, max_chars: int
) -> tuple[str, list[dict[str, Any]]]:
    """Execute a search and render it for the context window.

    Returns (rendered_text, retrieval_record). The record goes to the manifest
    so that a later reader can see exactly which passages were available to
    the model, and what they scored.
    """
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ToolError("search_corpus requires a non-empty 'query' string")

    results = retriever.search(query, k=k)
    record = [
        {"chunk_id": r.chunk.chunk_id, "score": round(r.score, 4)} for r in results
    ]

    if not results:
        return (
            "No passages in the corpus matched that query.",
            record,
        )

    # Context budget. Truncation is reported in-band rather than silently
    # dropping passages, because an answer that missed evidence it never saw
    # is a different defect from an answer that ignored evidence it had.
    rendered: list[str] = []
    used = 0
    dropped = 0
    for result in results:
        block = result.chunk.render()
        if used + len(block) > max_chars and rendered:
            dropped += 1
            continue
        rendered.append(block)
        used += len(block)

    text = "\n\n---\n\n".join(rendered)
    if dropped:
        text += f"\n\n[{dropped} further passage(s) omitted: context budget reached]"
    return text, record
