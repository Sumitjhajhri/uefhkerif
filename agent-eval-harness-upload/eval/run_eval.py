"""Run the evaluation suite and write a report.

    python -m eval.run_eval --version v2
    python -m eval.run_eval --version v1 --out results/

Produces a markdown report and the raw JSON beside it, plus one run manifest
per case under results/manifests/.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.checks import CaseGrade, Verdict, grade  # noqa: E402
from src.agent import Agent, AgentConfig, AgentRunError  # noqa: E402
from src.corpus import build_chunks  # noqa: E402
from src.models import ModelError, get_backend  # noqa: E402
from src.retriever import TfidfRetriever  # noqa: E402


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path} must contain a non-empty list of cases")

    seen: set[str] = set()
    for case in cases:
        case_id = case.get("id")
        if not case_id:
            raise ValueError(f"case without an id: {case}")
        if case_id in seen:
            raise ValueError(f"duplicate case id: {case_id}")
        seen.add(case_id)
    return cases


def summarise(grades: list[CaseGrade]) -> dict[str, Any]:
    by_kind: dict[str, Counter] = {}
    for g in grades:
        by_kind.setdefault(g.kind, Counter())[g.verdict.value] += 1

    errors = [g for g in grades if not g.verdict.measured]
    # Every rate below is computed over cases that actually ran. A case that
    # errored was not an observation of the agent's behaviour, and folding it
    # in either direction would put a number on something nobody measured.
    measured = [g for g in grades if g.verdict.measured]

    unanswerable = [g for g in measured if g.kind == "unanswerable"]
    answerable = [g for g in measured if g.kind == "answerable"]
    partial = [g for g in measured if g.kind == "partial"]

    fabricated = sum(1 for g in unanswerable if g.verdict is Verdict.FABRICATED)
    smoothed = sum(1 for g in partial if g.verdict is Verdict.SMOOTHED)

    return {
        "total_cases": len(grades),
        "errored": len(errors),
        "measured": len(measured),
        "passed": sum(1 for g in measured if g.ok),
        "fabrication_rate": round(fabricated / len(unanswerable), 4)
        if unanswerable
        else None,
        "fabricated_count": fabricated,
        "unanswerable_count": len(unanswerable),
        "answerable_accuracy": round(
            sum(1 for g in answerable if g.ok) / len(answerable), 4
        )
        if answerable
        else None,
        "smoothed_count": smoothed,
        "partial_count": len(partial),
        "unsupported_citation_cases": sum(
            1 for g in measured if g.unsupported_citations
        ),
        "by_kind": {k: dict(v) for k, v in by_kind.items()},
    }


def render_report(version: str, backend: str, summary: dict[str, Any],
                  grades: list[CaseGrade]) -> str:
    fab = summary["fabrication_rate"]
    fab_pct = "n/a" if fab is None else f"{fab * 100:.0f}%"
    acc = summary["answerable_accuracy"]
    acc_pct = "n/a" if acc is None else f"{acc * 100:.0f}%"

    lines = [
        f"# Evaluation report — agent {version}",
        "",
        f"Backend: `{backend}`  ·  Cases: {summary['total_cases']}  ·  "
        f"Measured: {summary['measured']}  ·  "
        f"Passed: {summary['passed']}/{summary['measured']}",
        "",
    ]

    if summary["errored"]:
        lines += [
            f"> **{summary['errored']} of {summary['total_cases']} cases did not "
            f"run.** The figures below cover only the "
            f"{summary['measured']} that did, and are not a result for this "
            "agent. Fix the errors listed at the end and re-run before "
            "quoting anything here.",
            "",
        ]

    lines += [
        "## Headline",
        "",
        f"| Metric | Result |",
        f"| --- | --- |",
        f"| **Fabrication rate** (unanswerable questions given an invented answer) "
        f"| **{fab_pct}** ({summary['fabricated_count']}/"
        f"{summary['unanswerable_count']}) |",
        f"| Accuracy on answerable questions | {acc_pct} |",
        f"| Partial questions smoothed over | {summary['smoothed_count']}/"
        f"{summary['partial_count']} |",
        f"| Cases citing a passage never retrieved | "
        f"{summary['unsupported_citation_cases']} |",
        "",
        "## Case detail",
        "",
        "| Case | Kind | Verdict | Reason |",
        "| --- | --- | --- | --- |",
    ]
    for g in grades:
        reason = g.reason.replace("|", "\\|")
        lines.append(f"| `{g.case_id}` | {g.kind} | **{g.verdict.value}** | {reason} |")

    failures = [g for g in grades if not g.ok]
    if failures:
        lines += ["", "## Failures in full", ""]
        for g in failures:
            lines += [
                f"### `{g.case_id}` — {g.verdict.value}",
                "",
                f"{g.reason}",
                "",
                "> " + (g.answer or "(no answer)").replace("\n", "\n> "),
                "",
            ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the agent evaluation suite.")
    parser.add_argument("--version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--backend", default="anthropic")
    parser.add_argument(
        "--model",
        default="claude-haiku-5-5",
        help=(
            "Any Claude model id. The harness is model-agnostic; the "
            "default is the cheapest current model because a full run "
            "costs a few cents on it. Pass a larger model to compare "
            "fabrication rates across models."
        ),
    )
    parser.add_argument("--corpus", default=str(ROOT / "corpus"))
    parser.add_argument("--cases", default=str(ROOT / "eval" / "cases.yaml"))
    parser.add_argument("--out", default=str(ROOT / "results"))
    args = parser.parse_args(argv)

    cases = load_cases(Path(args.cases))
    chunks = build_chunks(args.corpus)
    retriever = TfidfRetriever(chunks)

    try:
        backend = get_backend(args.backend, model=args.model)
    except (ValueError, ModelError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    config = AgentConfig.v1() if args.version == "v1" else AgentConfig.v2()
    agent = Agent(backend=backend, retriever=retriever, config=config)

    out_dir = Path(args.out)
    manifest_dir = out_dir / "manifests" / args.version
    grades: list[CaseGrade] = []

    print(f"Running {len(cases)} cases against agent {args.version} "
          f"({len(chunks)} chunks)\n")

    consecutive_errors = 0
    for case in cases:
        try:
            result = agent.run(case["question"])
            consecutive_errors = 0
        except (AgentRunError, ModelError) as exc:
            grades.append(
                CaseGrade(
                    case["id"],
                    case["kind"],
                    Verdict.ERROR,
                    f"run failed: {exc}",
                    [],
                    "",
                )
            )
            print(f"  {case['id']:<32} ERROR  {exc}")
            consecutive_errors += 1
            # Three in a row is a broken configuration, not bad luck. Stopping
            # avoids burning API spend on a run whose output would be discarded
            # anyway.
            if consecutive_errors >= 3:
                print(
                    "\nAborting: 3 consecutive failures. "
                    "This is a configuration problem, not a result.",
                    file=sys.stderr,
                )
                break
            continue

        if result.manifest is not None:
            result.manifest.write(manifest_dir)

        case_grade = grade(case, result)
        grades.append(case_grade)
        mark = "ok  " if case_grade.ok else case_grade.verdict.value.upper()
        print(f"  {case['id']:<32} {mark}")

    summary = summarise(grades)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = render_report(args.version, getattr(backend, "name", args.backend),
                           summary, grades)
    (out_dir / f"{args.version}_report.md").write_text(report, encoding="utf-8")
    (out_dir / f"{args.version}_report.json").write_text(
        json.dumps(
            {"summary": summary, "grades": [asdict(g) for g in grades]},
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    if summary["errored"]:
        print(
            f"\n{summary['errored']} of {summary['total_cases']} cases did not run. "
            "The figures below are not a result for this agent.",
            file=sys.stderr,
        )

    fab = summary["fabrication_rate"]
    print(
        f"\nFabrication rate: "
        f"{'n/a' if fab is None else f'{fab * 100:.0f}%'} "
        f"({summary['fabricated_count']}/{summary['unanswerable_count']})"
        f"{'  [INCOMPLETE RUN]' if summary['errored'] else ''}"
    )
    print(f"Report written to {out_dir / f'{args.version}_report.md'}")
    return 1 if summary["errored"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
