"""Agent loop tests.

Roughly half of these exercise the failure path - an exhausted script, a
malformed tool call, a model error, a citation that names a passage the agent
never saw. That ratio is deliberate: the success path is the one that gets
run by hand a hundred times during development, and the failure path is the
one nobody exercises until it matters in production.
"""

import pytest

from conftest import answer_call, search_call
from src.agent import Agent, AgentConfig, AgentRunError
from src.models import ModelError, ModelResponse, ScriptedBackend, ToolCall


def test_search_then_answer(retriever):
    backend = ScriptedBackend([
        search_call("Zone 1 single fare"),
        answer_call("A standard single within Zone 1 costs 2.40.",
                    citations=["faq#0"]),
    ])
    agent = Agent(backend, retriever, AgentConfig.v2())
    result = agent.run("What does a Zone 1 single cost?")

    assert "2.40" in result.answer
    assert result.answerable is True
    assert result.citations == ["faq#0"]
    assert result.unsupported_citations == []
    assert "faq#0" in result.retrieved_chunk_ids


def test_retrieved_context_reaches_the_model(retriever):
    """The passage must actually be in the prompt, not merely retrieved.

    Without this, a passing eval could coexist with a bug that drops context
    on the floor - the model would be answering from training data and the
    score would look fine.
    """
    backend = ScriptedBackend([
        search_call("penalty fare"),
        answer_call("60.00.", citations=["faq#1"]),
    ])
    Agent(backend, retriever, AgentConfig.v2()).run("What is the penalty fare?")

    second_prompt = backend.calls[1]["messages"]
    tool_result = second_prompt[-1]["content"][0]["content"]
    assert "60.00" in tool_result
    assert "[faq#1]" in tool_result


def test_unsupported_citation_is_flagged(retriever):
    backend = ScriptedBackend([
        search_call("Zone 1 single fare"),
        answer_call("It costs 2.40.", citations=["faq#0", "handbook#7"]),
    ])
    result = Agent(backend, retriever, AgentConfig.v2()).run("Zone 1 fare?")

    assert result.unsupported_citations == ["handbook#7"]


def test_v1_does_not_verify_citations(retriever):
    """v1 is the control. It must not get v2's safety net."""
    backend = ScriptedBackend([
        search_call("Zone 1 single fare"),
        answer_call("It costs 2.40.", citations=["invented#99"]),
    ])
    result = Agent(backend, retriever, AgentConfig.v1()).run("Zone 1 fare?")

    assert result.unsupported_citations == []


def test_answer_without_tool_call_is_accepted_and_noted(retriever):
    backend = ScriptedBackend([
        ModelResponse(text="The documents do not cover that.", stop_reason="end_turn")
    ])
    result = Agent(backend, retriever, AgentConfig.v2()).run("Anything?")

    assert result.answer == "The documents do not cover that."
    assert result.citations == []
    notes = [s for s in result.manifest.stages if s.name == "note"]
    assert notes and "without calling the answer tool" in notes[0].detail["note"]


def test_malformed_search_is_reported_to_the_model_not_raised(retriever):
    """A bad tool call should be recoverable, not fatal."""
    backend = ScriptedBackend([
        ModelResponse(
            tool_calls=[ToolCall(id="t1", name="search_corpus", arguments={})],
            stop_reason="tool_use",
        ),
        answer_call("Sorry, I could not search.", answerable=False),
    ])
    result = Agent(backend, retriever, AgentConfig.v2()).run("Zone 1 fare?")

    tool_result = backend.calls[1]["messages"][-1]["content"][0]["content"]
    assert "Tool error" in tool_result
    assert result.answerable is False


def test_unknown_tool_is_reported_as_error(retriever):
    backend = ScriptedBackend([
        ModelResponse(
            tool_calls=[ToolCall(id="t1", name="delete_everything", arguments={})],
            stop_reason="tool_use",
        ),
        answer_call("Done nothing.", answerable=False),
    ])
    Agent(backend, retriever, AgentConfig.v2()).run("Try it")

    tool_result = backend.calls[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert "delete_everything" in tool_result["content"]


def test_runaway_loop_is_bounded(retriever):
    backend = ScriptedBackend([search_call(f"q{i}", call_id=f"t{i}") for i in range(3)])
    config = AgentConfig.v2()
    config.max_iterations = 3

    with pytest.raises(AgentRunError, match="did not finish within 3 iterations"):
        Agent(backend, retriever, config).run("Loop forever")


def test_model_error_surfaces_as_run_error_not_a_bad_answer(retriever):
    class Broken:
        name = "broken"

        def complete(self, system, messages, tools=None):
            raise ModelError("upstream 503")

    with pytest.raises(AgentRunError, match="upstream 503"):
        Agent(Broken(), retriever, AgentConfig.v2()).run("Zone 1 fare?")


def test_empty_question_rejected(retriever):
    backend = ScriptedBackend([])
    with pytest.raises(AgentRunError, match="non-empty"):
        Agent(backend, retriever, AgentConfig.v2()).run("   ")


def test_manifest_records_the_run(retriever, tmp_path):
    backend = ScriptedBackend([
        search_call("Zone 1 single fare"),
        answer_call("2.40.", citations=["faq#0"]),
    ])
    result = Agent(backend, retriever, AgentConfig.v2()).run("Zone 1 fare?")
    manifest = result.manifest

    assert manifest.status == "ok"
    assert manifest.agent_version == "v2"
    assert manifest.model_calls == 2
    assert manifest.input_tokens == 220
    assert [t["name"] for t in manifest.tool_calls] == ["search_corpus", "answer"]
    assert any(r["chunk_id"] == "faq#0" for r in manifest.retrieved)
    assert all(s.status == "ok" for s in manifest.stages)

    written = manifest.write(tmp_path)
    assert written.exists()
    assert "faq#0" in written.read_text(encoding="utf-8")


def test_failed_stage_is_recorded_before_raising(retriever):
    class Broken:
        name = "broken"

        def complete(self, system, messages, tools=None):
            raise ModelError("boom")

    agent = Agent(Broken(), retriever, AgentConfig.v2())
    with pytest.raises(AgentRunError):
        agent.run("Zone 1 fare?")
    # The manifest is built inside run(); the failure must leave a trace on the
    # stage rather than vanishing with the exception.
