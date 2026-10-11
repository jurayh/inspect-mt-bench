"""Offline tests: dataset shape, rating parsing, judge prompt construction."""

from pathlib import Path

from inspect_mt_bench.mt_bench import (
    EXPECTED_QUESTION_COUNT,
    load_mt_bench_dataset,
)
from inspect_mt_bench.scorer import build_judge_prompt, parse_rating

HERE = Path(__file__).parent
REAL_QUESTIONS = str(HERE / "data" / "question.jsonl")
FIXTURE_QUESTIONS = str(HERE / "fixture_questions.jsonl")


def test_real_dataset_shape() -> None:
    dataset = load_mt_bench_dataset(REAL_QUESTIONS)
    assert len(dataset) == EXPECTED_QUESTION_COUNT
    categories = {sample.metadata["category"] for sample in dataset}
    assert categories == {
        "writing",
        "roleplay",
        "reasoning",
        "math",
        "coding",
        "extraction",
        "stem",
        "humanities",
    }
    for sample in dataset:
        assert len(sample.metadata["turns"]) == 2
        assert sample.input == sample.metadata["turns"][0]


def test_fixture_dataset_metadata() -> None:
    dataset = load_mt_bench_dataset(FIXTURE_QUESTIONS)
    math_sample, writing_sample = list(dataset)
    assert math_sample.id == "991"
    assert math_sample.metadata["reference"] == ["4", "12"]
    assert writing_sample.metadata["reference"] == []


def test_parse_rating() -> None:
    assert parse_rating("Some reasoning.\n\n[[7]]") == 7.0
    assert parse_rating("[[10]]") == 10.0
    assert parse_rating("[[1]]") == 1.0
    assert parse_rating("[[3.5]]") == 3.5
    assert parse_rating("no rating here") is None
    assert parse_rating("") is None
    assert parse_rating("[[0]]") is None  # below the 1-10 rubric
    assert parse_rating("[[11]]") is None  # above the 1-10 rubric
    assert parse_rating("[[x]]") is None


def test_prompt_turn1_without_reference() -> None:
    system, user = build_judge_prompt(0, ["Write a haiku."], ["leaves fall down"], [])
    assert "helpful assistant" in system
    assert "Write a haiku." in user
    assert "leaves fall down" in user
    assert "[Reference answer]" not in user


def test_prompt_turn1_with_reference() -> None:
    _system, user = build_judge_prompt(0, ["What is 2+2?"], ["It is 4."], ["4"])
    assert "It is 4." in user
    assert "\n4" in user  # the reference answer itself is embedded


def test_prompt_turn2_multiturn() -> None:
    _, user = build_judge_prompt(
        1, ["First?", "Second?"], ["Answer one.", "Answer two."], []
    )
    assert "First?" in user and "Second?" in user
    assert "Answer one." in user and "Answer two." in user


def test_prompt_substitution_survives_braces_in_answers() -> None:
    _, user = build_judge_prompt(
        0, ["Explain dict syntax."], ["Use {'a': 1} and {question} literally."], []
    )
    assert "{'a': 1}" in user
    assert "and {question} literally." in user
