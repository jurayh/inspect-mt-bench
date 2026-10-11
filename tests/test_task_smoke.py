"""End-to-end smoke test on the mock model: no network, no API keys.

The mock model returns a fixed completion, so the judge's replies contain
no [[rating]] and every sample must surface the judge_parse_failed flag
with a 0 score rather than a silent or fabricated rating. That makes the
test an assertion on the failure-visibility path as well as the plumbing:
dataset loading, the two-turn solver, and the judge scorer all run.
"""

from pathlib import Path

from inspect_ai import eval

from inspect_mt_bench import mt_bench

FIXTURE_QUESTIONS = str(Path(__file__).parent / "fixture_questions.jsonl")


def test_eval_runs_two_turns_and_flags_unparseable_judge(tmp_path) -> None:
    logs = eval(
        mt_bench(
            judge_model="mockllm/model",
            dataset_source=FIXTURE_QUESTIONS,
        ),
        model="mockllm/model",
        log_dir=str(tmp_path / "logs"),
    )
    assert len(logs) == 1
    log = logs[0]
    assert log.status == "success"
    assert log.results is not None
    assert len(log.samples) == 2
    for sample in log.samples:
        # Both user turns and both assistant replies are in the transcript.
        roles = [message.role for message in sample.messages]
        assert roles.count("user") == 2
        assert roles.count("assistant") == 2
        assert sample.scores is not None
        score = sample.scores["mt_bench_judge"]
        assert score.metadata["judge_parse_failed"] is True
        assert score.metadata["turn_ratings"] == [None, None]
        assert score.value == 0.0
