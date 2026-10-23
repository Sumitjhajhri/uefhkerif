import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from src.corpus import Chunk  # noqa: E402
from src.models import ModelResponse, ToolCall  # noqa: E402
from src.retriever import TfidfRetriever  # noqa: E402


@pytest.fixture
def chunks():
    return [
        Chunk("faq#0", "faq", "A standard single within Zone 1 costs 2.40.", 0),
        Chunk("faq#1", "faq", "The penalty fare is 60.00, reduced to 30.00 if "
                              "paid within twenty-one days.", 1),
        Chunk("rules#0", "rules", "Staff may work no more than nine consecutive "
                                  "hours on duty.", 0),
    ]


@pytest.fixture
def retriever(chunks):
    return TfidfRetriever(chunks, min_score=0.01)


def search_call(query: str, call_id: str = "t1") -> ModelResponse:
    return ModelResponse(
        tool_calls=[ToolCall(id=call_id, name="search_corpus",
                             arguments={"query": query})],
        stop_reason="tool_use",
        input_tokens=100,
        output_tokens=20,
    )


def answer_call(answer: str, answerable: bool = True,
                citations: list[str] | None = None,
                call_id: str = "t2") -> ModelResponse:
    return ModelResponse(
        tool_calls=[
            ToolCall(
                id=call_id,
                name="answer",
                arguments={
                    "answerable": answerable,
                    "answer": answer,
                    "citations": citations or [],
                },
            )
        ],
        stop_reason="tool_use",
        input_tokens=120,
        output_tokens=40,
    )
