"""End-to-end test of the evaluation pipeline.

Runs the real cases file, the real corpus and the real graders against two
scripted agents - one that fabricates and one that declines - and checks the
reported fabrication rate comes out at 100% and 0% respectively.

This is the test that proves the headline number means what the README says it
means. Without it, a bug anywhere between the agent and the report could move
that number and nothing would notice.
"""

from pathlib import Path

import pytest

from eval.checks import Verdict, grade
from eval.run_eval import load_cases, render_report, summarise
from src.agent import Agent, AgentConfig
from src.corpus import build_chunks
from src.models import ModelResponse, ScriptedBackend, ToolCall
from src.retriever import TfidfRetriever

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def cases():
    return load_cases(ROOT / "eval" / "cases.yaml")


@pytest.fixture(scope="module")
def real_retriever():
    return TfidfRetriever(build_chunks(ROOT / "corpus"))


def one_shot(answer: str, answerable: bool, citations=None) -> ScriptedBackend:
    """An agent that answers immediately, without searching."""
    return ScriptedBackend([
        ModelResponse(
            tool_calls=[
                ToolCall(
                    id="a1",
                    name="answer",
                    arguments={
                        "answerable": answerable,
                        "answer": answer,
                        "citations": citations or [],
                    },
                )
            ],
            stop_reason="tool_use",
        )
    ])


def test_cases_file_is_well_formed(cases):
    kinds = {c["kind"] for c in cases}
    assert kinds <= {"answerable", "unanswerable", "partial"}
    assert sum(1 for c in cases if c["kind"] == "unanswerable") >= 5
    assert sum(1 for c in cases if c["kind"] == "answerable") >= 5
    for case in cases:
        assert case["question"].strip()
        if case["kind"] in {"answerable", "partial"}:
            assert case.get("expect_contains"), f"{case['id']} needs expect_contains"
        else:
            assert case.get("trap"), f"{case['id']} needs a trap explanation"


def test_always_fabricating_agent_scores_one_hundred_percent(cases, real_retriever):
    unanswerable = [c for c in cases if c["kind"] == "unanswerable"]
    grades = []
    for case in unanswerable:
        agent = Agent(one_shot("The applicable figure is 42.00.", answerable=True),
                      real_retriever, AgentConfig.v1())
        grades.append(grade(case, agent.run(case["question"])))

    assert all(g.verdict is Verdict.FABRICATED for g in grades)
    assert summarise(grades)["fabrication_rate"] == 1.0


def test_always_declining_agent_scores_zero_percent(cases, real_retriever):
    unanswerable = [c for c in cases if c["kind"] == "unanswerable"]
    grades = []
    for case in unanswerable:
        agent = Agent(
            one_shot("The documents do not cover that.", answerable=False),
            real_retriever,
            AgentConfig.v2(),
        )
        grades.append(grade(case, agent.run(case["question"])))

    assert all(g.verdict is Verdict.PASS for g in grades)
    assert summarise(grades)["fabrication_rate"] == 0.0


def test_declining_everything_destroys_the_answerable_score(cases, real_retriever):
    """Guards against the degenerate fix.

    An agent that refuses every question has a fabrication rate of zero and is
    useless. The suite has to make that visible, otherwise 'improve v1 to v2'
    could be satisfied by teaching it to say no to everything.
    """
    answerable = [c for c in cases if c["kind"] == "answerable"]
    grades = []
    for case in answerable:
        agent = Agent(
            one_shot("The documents do not cover that.", answerable=False),
            real_retriever,
            AgentConfig.v2(),
        )
        grades.append(grade(case, agent.run(case["question"])))

    assert summarise(grades)["answerable_accuracy"] == 0.0


def test_report_renders_with_the_headline_number(cases, real_retriever):
    case = next(c for c in cases if c["kind"] == "unanswerable")
    agent = Agent(one_shot("It is 42.00.", answerable=True), real_retriever,
                  AgentConfig.v1())
    grades = [grade(case, agent.run(case["question"]))]
    report = render_report("v1", "scripted", summarise(grades), grades)

    assert "Fabrication rate" in report
    assert "100%" in report
    assert case["id"] in report
    assert "Failures in full" in report


def test_duplicate_case_ids_rejected(tmp_path):
    path = tmp_path / "cases.yaml"
    path.write_text(
        "- id: a\n  kind: unanswerable\n  question: q\n  trap: t\n"
        "- id: a\n  kind: unanswerable\n  question: q\n  trap: t\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate case id"):
        load_cases(path)


def test_errored_cases_are_excluded_from_every_rate():
    """Regression test for a bug this harness had.

    When every case failed because no API key was set, the runner reported a
    fabrication rate of 0% - the best possible score, from a run in which
    nothing was measured at all. A harness that reports a clean number for a
    run that did not happen is the precise failure this project exists to
    detect, so it gets a test.
    """
    from eval.checks import CaseGrade

    grades = [
        CaseGrade("a", "unanswerable", Verdict.ERROR, "run failed: no key"),
        CaseGrade("b", "unanswerable", Verdict.ERROR, "run failed: no key"),
        CaseGrade("c", "unanswerable", Verdict.FABRICATED, "invented"),
        CaseGrade("d", "answerable", Verdict.ERROR, "run failed: no key"),
        CaseGrade("e", "answerable", Verdict.PASS, "fine"),
    ]
    summary = summarise(grades)

    assert summary["errored"] == 3
    assert summary["measured"] == 2
    # One unanswerable case ran and it fabricated: 100%, not 33%.
    assert summary["fabrication_rate"] == 1.0
    assert summary["unanswerable_count"] == 1
    assert summary["answerable_accuracy"] == 1.0


def test_report_warns_loudly_on_an_incomplete_run():
    from eval.checks import CaseGrade

    grades = [
        CaseGrade("a", "unanswerable", Verdict.ERROR, "run failed: no key"),
        CaseGrade("b", "unanswerable", Verdict.PASS, "declined"),
    ]
    report = render_report("v2", "anthropic", summarise(grades), grades)

    assert "did not run" in report
    assert "not a result" in report


def test_a_run_where_nothing_ran_reports_no_rate_at_all():
    from eval.checks import CaseGrade

    grades = [CaseGrade(c, "unanswerable", Verdict.ERROR, "boom")
              for c in ("a", "b", "c")]
    summary = summarise(grades)

    assert summary["fabrication_rate"] is None
    assert summary["measured"] == 0
