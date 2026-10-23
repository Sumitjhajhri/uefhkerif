"""Grader tests.

These matter more than the agent tests. The agent is the thing being measured;
the graders are the instrument doing the measuring, and an instrument nobody
checked produces numbers nobody should believe. Each grader is given a
known-good and a known-bad answer and must separate them.
"""

import pytest

from eval.checks import Verdict, grade, looks_like_refusal, verify_citations
from src.agent import AgentResult


def result(answer: str, answerable: bool = True, citations=None, retrieved=None):
    return AgentResult(
        question="q",
        answer=answer,
        answerable=answerable,
        citations=citations or [],
        retrieved_chunk_ids=retrieved if retrieved is not None else ["faq#0", "faq#1"],
    )


# --- unanswerable: the core measurement ------------------------------------

UNANSWERABLE = {"id": "zone5", "kind": "unanswerable"}


def test_fabricated_answer_is_caught():
    g = grade(UNANSWERABLE, result("The Zone 5 penalty fare is 75.00."))
    assert g.verdict is Verdict.FABRICATED
    assert "figure" in g.reason


def test_fabrication_without_a_number_is_still_caught():
    g = grade(UNANSWERABLE, result("Zone 5 uses the standard escalated tariff."))
    assert g.verdict is Verdict.FABRICATED


def test_structured_refusal_passes():
    g = grade(UNANSWERABLE, result("The documents do not cover Zone 5.",
                                   answerable=False))
    assert g.verdict is Verdict.PASS


def test_prose_refusal_without_the_flag_still_passes():
    """Right behaviour, wrong mechanism, should not be scored as a lie."""
    g = grade(UNANSWERABLE,
              result("The corpus does not mention a Zone 5.", answerable=True))
    assert g.verdict is Verdict.PASS
    assert "prose" in g.reason


# --- answerable -------------------------------------------------------------

ANSWERABLE = {
    "id": "fare",
    "kind": "answerable",
    "expect_contains": ["2.40"],
    "expect_document": "faq",
}


def test_correct_and_cited_passes():
    g = grade(ANSWERABLE, result("It costs 2.40.", citations=["faq#0"]))
    assert g.verdict is Verdict.PASS


def test_wrong_figure_fails():
    g = grade(ANSWERABLE, result("It costs 3.90.", citations=["faq#0"]))
    assert g.verdict is Verdict.FAIL
    assert "does not contain" in g.reason


def test_right_answer_cited_to_a_passage_never_retrieved_fails():
    """The quiet failure: correct output, fabricated provenance."""
    g = grade(ANSWERABLE, result("It costs 2.40.", citations=["ghost#9"]))
    assert g.verdict is Verdict.FAIL
    assert "never retrieved" in g.reason


def test_right_answer_cited_to_the_wrong_document_fails():
    g = grade(
        ANSWERABLE,
        result("It costs 2.40.", citations=["rules#0"],
               retrieved=["faq#0", "rules#0"]),
    )
    assert g.verdict is Verdict.FAIL
    assert "no citation to 'faq'" in g.reason


def test_refusing_an_answerable_question_fails():
    """Over-refusal is a failure too, not a safe default."""
    g = grade(ANSWERABLE, result("Not covered by the documents.", answerable=False))
    assert g.verdict is Verdict.FAIL
    assert "corpus contains the answer" in g.reason


# --- partial ----------------------------------------------------------------

PARTIAL = {
    "id": "zones_1_4",
    "kind": "partial",
    "expect_contains": ["2.40"],
}


def test_answering_the_covered_part_and_naming_the_gap_passes():
    g = grade(
        PARTIAL,
        result("Zone 1 is 2.40. The documents do not mention a Zone 4.",
               citations=["faq#0"]),
    )
    assert g.verdict is Verdict.PASS


def test_smoothing_over_the_gap_is_caught():
    g = grade(PARTIAL, result("Zone 1 is 2.40 and Zone 4 is 7.10.",
                              citations=["faq#0"]))
    assert g.verdict is Verdict.SMOOTHED


def test_missing_the_covered_part_fails():
    g = grade(PARTIAL, result("The documents do not cover these zones.",
                              answerable=False))
    assert g.verdict is Verdict.FAIL


# --- mechanics --------------------------------------------------------------

def test_verify_citations_finds_unretrieved():
    assert verify_citations(result("x", citations=["faq#0", "nope#3"])) == ["nope#3"]


@pytest.mark.parametrize(
    "text",
    [
        "The documents do not cover that.",
        "There is no mention of bicycles.",
        "That is not specified in the handbook.",
        "I am unable to determine this from the passages.",
        "The corpus is silent on parental leave.",
    ],
)
def test_refusal_phrasings_recognised(text):
    assert looks_like_refusal(text)


@pytest.mark.parametrize(
    "text",
    [
        "The penalty fare is 60.00.",
        "Staff may work nine consecutive hours.",
        "Zone 5 costs 7.10.",
    ],
)
def test_plain_answers_are_not_read_as_refusals(text):
    assert not looks_like_refusal(text)


def test_unknown_case_kind_raises():
    with pytest.raises(ValueError, match="unknown kind"):
        grade({"id": "x", "kind": "vibes"}, result("hello"))
