"""Per-run audit record.

Every agent run writes a manifest: what was asked, which chunks came back and
with what scores, every tool call and its result, token counts, timings, the
final answer, and any exception text.

The reason is specific. When an agent gives a wrong answer you need to know
*which* stage was wrong - did retrieval miss the relevant chunk, or did
retrieval succeed and the model ignore it? Those have opposite fixes, and
without a record of what was in the context window you cannot tell them apart
after the fact. A failing eval score with no manifest tells you that something
is wrong and nothing about what.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StageRecord:
    name: str
    status: str = "pending"  # pending | ok | failed | skipped
    duration_ms: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class RunManifest:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    question: str = ""
    agent_version: str = ""
    backend: str = ""
    started_at: str = ""
    duration_ms: float = 0.0
    status: str = "pending"  # pending | ok | failed
    stages: list[StageRecord] = field(default_factory=list)
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    error: str | None = None

    _t0: float = field(default=0.0, repr=False)

    def start(self) -> "RunManifest":
        self._t0 = time.perf_counter()
        self.started_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        return self

    def stage(self, name: str) -> "_StageTimer":
        return _StageTimer(self, name)

    def finish(self, status: str, error: str | None = None) -> "RunManifest":
        self.duration_ms = round((time.perf_counter() - self._t0) * 1000, 2)
        self.status = status
        self.error = error
        return self

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("_t0", None)
        return data

    def write(self, directory: str | Path) -> Path:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        target = path / f"run_{self.run_id}.json"
        target.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return target


class _StageTimer:
    """Context manager that records a stage whether or not it succeeds.

    A stage that raises is recorded as failed, with the exception text, and
    the exception is re-raised. The failure path is the one that has to leave
    a trace - a stage that silently vanished from the manifest is worse than
    no manifest, because it looks complete.
    """

    def __init__(self, manifest: RunManifest, name: str):
        self.record = StageRecord(name=name)
        manifest.stages.append(self.record)
        self._t0 = 0.0

    def __enter__(self) -> StageRecord:
        self._t0 = time.perf_counter()
        return self.record

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.record.duration_ms = round((time.perf_counter() - self._t0) * 1000, 2)
        if exc_type is None:
            self.record.status = "ok"
        else:
            self.record.status = "failed"
            self.record.error = f"{exc_type.__name__}: {exc}"
        return False  # never swallow
