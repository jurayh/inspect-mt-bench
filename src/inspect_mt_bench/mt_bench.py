"""MT-Bench as an Inspect AI task.

MT-Bench (Zheng et al., 2023, arXiv 2306.05685) evaluates chat models on
80 two-turn questions across 8 categories (writing, roleplay, reasoning,
math, coding, extraction, STEM, humanities) graded by an LLM judge on a
1-10 scale. This implementation loads the official question set from the
FastChat repository pinned at a fixed commit, runs both turns, and grades
each turn with the official single-answer judge prompts.

Data pinning: every external asset is loaded from a URL that embeds the
FastChat commit SHA below. Bump FASTCHAT_COMMIT only deliberately, and
re-check the question count and judge prompt set when you do.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver

from .scorer import DEFAULT_JUDGE_MODEL, mt_bench_judge

FASTCHAT_COMMIT = "587d5cfa1609a43d192cedb8441cac3c17db105d"
QUESTION_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/"
    f"{FASTCHAT_COMMIT}/fastchat/llm_judge/data/mt_bench/question.jsonl"
)
EXPECTED_QUESTION_COUNT = 80


def _load_jsonl(source: str) -> list[dict]:
    """Load a JSONL file from a local path or an https URL (cached)."""
    if os.path.exists(source):
        text = Path(source).read_text(encoding="utf-8")
    else:
        cache_dir = Path.home() / ".cache" / "inspect_mt_bench"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.sha256(source.encode()).hexdigest()[:16]
        cache_path = cache_dir / f"{cache_key}.jsonl"
        if cache_path.exists():
            text = cache_path.read_text(encoding="utf-8")
        else:
            request = urllib.request.Request(
                source, headers={"User-Agent": "inspect-mt-bench"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                text = response.read().decode("utf-8")
            cache_path.write_text(text, encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_mt_bench_dataset(source: str = QUESTION_URL) -> MemoryDataset:
    """Build the MT-Bench dataset of two-turn samples."""
    records = _load_jsonl(source)
    samples = []
    for record in records:
        turns = list(record["turns"])
        samples.append(
            Sample(
                id=str(record["question_id"]),
                input=turns[0],
                target="",
                metadata={
                    "category": record["category"],
                    "turns": turns,
                    "reference": list(record.get("reference") or []),
                },
            )
        )
    return MemoryDataset(
        samples=samples, name="mt_bench", location=source, shuffled=False
    )


@solver
def two_turn_solver() -> Solver:
    """Generate a reply to turn 1, append turn 2, and generate again."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state = await generate(state)
        turns = list(state.metadata.get("turns") or [])
        if len(turns) > 1:
            state.messages.append(ChatMessageUser(content=turns[1]))
            state = await generate(state)
        return state

    return solve


@task
def mt_bench(
    judge_model: str = DEFAULT_JUDGE_MODEL,
    dataset_source: str = QUESTION_URL,
) -> Task:
    """MT-Bench: 80 two-turn questions graded by an LLM judge (1-10).

    Args:
        judge_model: Model that grades the answers. Defaults to a fixed
            GPT-4-class model. It never defaults to the model under
            evaluation; pointing it at the evaluated model triggers a
            warning in the scorer.
        dataset_source: Local path or URL for the MT-Bench question file.
            The default is the official FastChat file pinned by commit.
    """
    return Task(
        name="mt_bench",
        dataset=load_mt_bench_dataset(dataset_source),
        solver=two_turn_solver(),
        scorer=mt_bench_judge(judge_model=judge_model),
        version=1,
    )
