"""Single-answer MT-Bench judge scorer.

Each assistant turn is graded separately with the official FastChat judge
prompts (see judge_prompts.py), following the original grading scheme:
turn 1 with the single-turn templates, turn 2 with the multi-turn variants,
and the reference-guided ("math") templates whenever the question ships a
reference answer for that turn. The sample score is the mean of the parsed
turn ratings on the official 1-10 scale.

Two deliberate safeguards:

- The judge model is an explicit parameter with a fixed default. It never
  falls back to the model under evaluation. If a run configures the judge
  to be the same model as the evaluated one, the scorer logs a warning for
  every sample rather than failing the run.
- A judge reply whose rating cannot be parsed is never silently averaged
  away. The turn is recorded as None in the score metadata, and a sample
  with no parseable turn rating scores 0 with judge_parse_failed set, so
  judge failures stay visible in the eval log and in aggregate metrics.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState

from . import judge_prompts

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "openai/gpt-4o"

_RATING_PATTERN = re.compile(r"\[\[(\d+(?:\.\d+)?)\]\]")


def parse_rating(judge_output: str) -> float | None:
    """Extract the [[rating]] from a judge reply, or None if unusable."""
    match = _RATING_PATTERN.search(judge_output or "")
    if not match:
        return None
    value = float(match.group(1))
    if not 1.0 <= value <= 10.0:
        return None
    return value


_TOKEN_PATTERN = re.compile(r"\{([a-z_0-9]+)\}")


def _fill(template: str, values: dict[str, str]) -> str:
    """Substitute {token} placeholders in a single pass over the template.

    The official templates are plain text with a small fixed placeholder
    set, so token replacement is used instead of str.format. Substitution
    is single-pass: inserted answers and reference texts are never
    rescanned, so braces inside them survive untouched.
    """

    def replace(match: re.Match[str]) -> str:
        return values.get(match.group(1), match.group(0))

    return _TOKEN_PATTERN.sub(replace, template)


def build_judge_prompt(
    turn_index: int,
    questions: list[str],
    answers: list[str],
    references: list[str],
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for grading one turn."""
    has_refs = len(references) > turn_index and bool(references[turn_index])
    if turn_index == 0:
        spec = judge_prompts.SINGLE_MATH_V1 if has_refs else judge_prompts.SINGLE_V1
        values = {"question": questions[0], "answer": answers[0]}
        if has_refs:
            values["ref_answer_1"] = references[0]
    else:
        spec = (
            judge_prompts.SINGLE_MATH_V1_MULTI_TURN
            if has_refs
            else judge_prompts.SINGLE_V1_MULTI_TURN
        )
        values = {
            "question_1": questions[0],
            "answer_1": answers[0],
            "question_2": questions[1],
            "answer_2": answers[1],
        }
        if has_refs:
            values["ref_answer_1"] = references[0] if references else ""
            values["ref_answer_2"] = references[1]
    return spec["system_prompt"], _fill(spec["prompt_template"], values)


@scorer(metrics=[mean(), stderr()])
def mt_bench_judge(judge_model: str = DEFAULT_JUDGE_MODEL) -> Scorer:
    """Score a completed two-turn conversation with an MT-Bench judge."""

    async def score(state: TaskState, target: Target) -> Score:
        turns: list[str] = list(state.metadata.get("turns") or [])
        references: list[str] = list(state.metadata.get("reference") or [])
        answers = [
            message.text for message in state.messages if message.role == "assistant"
        ]

        evaluated_model = str(getattr(state, "model", "") or "")
        if evaluated_model and evaluated_model == judge_model:
            logger.warning(
                "MT-Bench judge model %s is the model under evaluation. "
                "Scores from a model judging itself are not comparable to "
                "standard MT-Bench results.",
                judge_model,
            )

        judge = get_model(judge_model)
        ratings: list[float | None] = []
        explanations: list[str] = []
        for turn_index in range(min(len(turns), len(answers))):
            system_prompt, user_prompt = build_judge_prompt(
                turn_index, turns, answers, references
            )
            output = await judge.generate(
                input=[
                    ChatMessageSystem(content=system_prompt),
                    ChatMessageUser(content=user_prompt),
                ]
            )
            completion = output.completion or ""
            ratings.append(parse_rating(completion))
            explanations.append(f"Turn {turn_index + 1} judge output:\n{completion}")

        parsed = [r for r in ratings if r is not None]
        metadata: dict[str, Any] = {
            "turn_ratings": ratings,
            "judge_model": judge_model,
        }
        if not parsed:
            metadata["judge_parse_failed"] = True
            value = 0.0
        else:
            value = sum(parsed) / len(parsed)
        return Score(
            value=value,
            answer=answers[-1] if answers else "",
            explanation="\n\n".join(explanations),
            metadata=metadata,
        )

    return score
