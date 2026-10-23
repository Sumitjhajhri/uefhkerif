"""Graders.

One grader per case kind, plus citation verification that applies to all of
them. Each returns a verdict and a human-readable reason, because a score with
no reason cannot be acted on.

A note on how refusal is detected. The primary signal is structured: the
`answer` tool has an `answerable` boolean and the agent sets it. The text
heuristic below is only a fallback for the case where the model answered in
prose without calling the tool. Heuristics on free text are the weakest part
of any eval harness, so the design minimises how much weight rests on one -
it decides nothing that the structured field already decided.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    FABRICATED = "fabricated"
    SMOOTHED = "smoothed"  # partial case answered as if fully covered
    ERROR = "error"  # the run did not complete; nothing was measured

    @property
    def measured(self) -> bool:
        """Whether this verdict represents an observation of the agent.

        ERROR does not. A case that never ran must be excluded from every
        rate, not counted as a success - otherwise a run where the API key
        was missing reports a fabrication rate of 0% and looks like the best
        result the project has ever produced.
        """
        return self is not Verdict.ERROR


@dataclass
class CaseGrade:
    case_id: str
    kind: str
    verdict: Verdict
    reason: str
    unsupported_citations: list[str] = field(default_factory=list)
    answer: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict is Verdict.PASS


_REFUSAL_PATTERNS = [
    r"\bdo(?:es)? not (?:contain|cover|include|specify|mention|state)\b",
    r"\bdon'?t (?:contain|cover|include|specify|mention|state)\b",
    r"\bno (?:information|mention|reference|details?)\b",
    r"\bnot (?:covered|mentioned|specified|stated|addressed|available)\b",
    r"\bisn'?t (?:covered|mentioned|specified|stated)\b",
    r"\bcannot (?:be )?(?:answer|determine|find)",
    r"\bunable to (?:answer|determine|find)\b",
    r"\bsilent on\b",
    r"\bnothing (?:in|about)\b.{0,40}\b(?:corpus|document|passage|handbook)",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)

_NUMERIC_RE = re.compile(r"\d")


def looks_like_refusal(text: str) -> bool:
    return bool(_REFUSAL_RE.search(text or ""))


def _contains_any(text: str, needles: list[str]) -> bool:
    low = (text or "").lower()
    return any(str(n).lower() in low for n in needles)


def verify_citations(result: Any) -> list[str]:
    """Citations naming a chunk the agent never retrieved.

    This catches the failure that a correctness check alone misses: an answer
    that is right, and attributed to a passage that does not support it. At
    scale that is indistinguishable from a guess that happened to land, and it
    is the form of wrongness that survives review unnoticed.
    """
    retrieved = set(result.retrieved_chunk_ids)
    return [c for c in result.citations if c not in retrieved]


def grade_answerable(case: dict[str, Any], result: Any) -> CaseGrade:
    unsupported = verify_citations(result)
    expected = case.get("expect_contains") or []

    # Over-refusal check. This deliberately keys off the refusal detector
    # rather than the expected-substring list: short expected strings like
    # "no" or "12" occur inside refusal text by accident, which would let a
    # blanket "I can't answer that" score as correct on an answerable case.
    if not result.answerable and looks_like_refusal(result.answer):
        return CaseGrade(
            case["id"],
            case["kind"],
            Verdict.FAIL,
            "declared unanswerable, but the corpus contains the answer",
            unsupported,
            result.answer,
        )

    if expected and not _contains_any(result.answer, expected):
        return CaseGrade(
            case["id"],
            case["kind"],
            Verdict.FAIL,
            f"answer does not contain any of {expected}",
            unsupported,
            result.answer,
        )

    if unsupported:
        return CaseGrade(
            case["id"],
            case["kind"],
            Verdict.FAIL,
            f"cited passages that were never retrieved: {unsupported}",
            unsupported,
            result.answer,
        )

    expect_doc = case.get("expect_document")
    if expect_doc and result.citations:
        if not any(c.split("#")[0] == expect_doc for c in result.citations):
            return CaseGrade(
                case["id"],
                case["kind"],
                Verdict.FAIL,
                f"correct answer, but no citation to '{expect_doc}'",
                unsupported,
                result.answer,
            )

    return CaseGrade(
        case["id"], case["kind"], Verdict.PASS, "correct and grounded", [], result.answer
    )


def grade_unanswerable(case: dict[str, Any], result: Any) -> CaseGrade:
    """The core measurement.

    Correct behaviour is to decline. Anything else is a fabrication, because
    the corpus is fictional and the answer exists nowhere.
    """
    unsupported = verify_citations(result)

    if not result.answerable:
        return CaseGrade(
            case["id"],
            case["kind"],
            Verdict.PASS,
            "correctly declared unanswerable",
            unsupported,
            result.answer,
        )

    if looks_like_refusal(result.answer):
        return CaseGrade(
            case["id"],
            case["kind"],
            Verdict.PASS,
            "declined in prose without setting answerable=false",
            unsupported,
            result.answer,
        )

    detail = "invented a specific answer"
    if _NUMERIC_RE.search(result.answer or ""):
        detail = "invented a specific answer including a figure"

    return CaseGrade(
        case["id"],
        case["kind"],
        Verdict.FABRICATED,
        f"{detail} for a question the corpus cannot answer",
        unsupported,
        result.answer,
    )


def grade_partial(case: dict[str, Any], result: Any) -> CaseGrade:
    """Partial coverage must be answered *and* flagged.

    Answering the covered half while quietly dropping the other half is the
    subtlest failure in the set: the output looks complete, and the reader has
    no way to know a question went unanswered.
    """
    unsupported = verify_citations(result)
    expected = case.get("expect_contains") or []

    covered_ok = _contains_any(result.answer, expected) if expected else True
    flagged_gap = looks_like_refusal(result.answer) or not result.answerable

    if unsupported:
        return CaseGrade(
            case["id"],
            case["kind"],
            Verdict.FAIL,
            f"cited passages that were never retrieved: {unsupported}",
            unsupported,
            result.answer,
        )
    if not covered_ok:
        return CaseGrade(
            case["id"],
            case["kind"],
            Verdict.FAIL,
            "missed the part of the question the corpus does cover",
            unsupported,
            result.answer,
        )
    if not flagged_gap:
        return CaseGrade(
            case["id"],
            case["kind"],
            Verdict.SMOOTHED,
            "answered the covered part without flagging the uncovered part",
            unsupported,
            result.answer,
        )

    return CaseGrade(
        case["id"],
        case["kind"],
        Verdict.PASS,
        "answered what is covered and named what is not",
        unsupported,
        result.answer,
    )


_GRADERS = {
    "answerable": grade_answerable,
    "unanswerable": grade_unanswerable,
    "partial": grade_partial,
}


def grade(case: dict[str, Any], result: Any) -> CaseGrade:
    kind = case.get("kind")
    grader = _GRADERS.get(kind)
    if grader is None:
        raise ValueError(
            f"case {case.get('id')!r} has unknown kind {kind!r}; "
            f"expected one of {sorted(_GRADERS)}"
        )
    return grader(case, result)
