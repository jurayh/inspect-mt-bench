# MT-Bench for Inspect AI

An [Inspect AI](https://inspect.aisi.org.uk/) implementation of **MT-Bench**
(Zheng et al., 2023, [arXiv:2306.05685](https://arxiv.org/abs/2306.05685v3)):
80 two-turn questions across 8 categories (writing, roleplay, reasoning,
math, coding, extraction, STEM, humanities), graded by an LLM judge on the
official 1–10 scale with the original FastChat judge prompts.

MT-Bench is not part of the inspect_evals catalogue. This repository is the
upstream home of the eval; it is being submitted to the
[Inspect Evals Register](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/register)
so the listing appears in the official docs.

## Usage

Install and run against any Inspect model:

```bash
pip install git+https://github.com/jurayh/inspect-mt-bench
inspect eval inspect_mt_bench/mt_bench --model openai/gpt-4o-mini
```

Or from a checkout, against the task file directly:

```bash
uv sync  # or: pip install -e .
inspect eval src/inspect_mt_bench/mt_bench.py --model openai/gpt-4o-mini
```

The judge is configurable and never silently defaults to the model under
evaluation:

```bash
inspect eval inspect_mt_bench/mt_bench --model anthropic/claude-sonnet-4-5 \
  -T judge_model=openai/gpt-4o
```

## How it works

- **Dataset.** The official question file is loaded from the FastChat
  repository pinned at commit `587d5cfa1609a43d192cedb8441cac3c17db105d`
  (`fastchat/llm_judge/data/mt_bench/question.jsonl`, 80 questions, two
  turns each). Downloads are cached under `~/.cache/inspect_mt_bench/`.
- **Solver.** A small solver generates the turn-1 reply, appends the
  turn-2 user message, and generates again, so the judge sees the real
  two-turn conversation.
- **Scorer.** Each turn is graded separately, following the original
  single-answer grading scheme: turn 1 with the single-turn judge
  templates, turn 2 with the multi-turn variants, and the reference-guided
  ("math") templates whenever the question ships a per-turn reference
  answer (39 of 80 questions do, embedded in the question file). The
  sample score is the mean of the parsed turn ratings.
  Judge prompts are ported **verbatim** from FastChat's
  `judge_prompts.jsonl` at the same pinned commit — see
  `src/inspect_mt_bench/judge_prompts.py`.

### Porting decisions worth knowing

- **Judge default.** The paper graded with GPT-4 (`gpt-4-0613`). This
  implementation defaults to `openai/gpt-4o`, the current GPT-4-class
  judge, and takes the judge as an explicit task parameter
  (`judge_model`). Scores across different judges are not comparable;
  the judge used is recorded in every score's metadata.
- **Self-judging guard.** If the configured judge is the model under
  evaluation, the scorer logs a warning per sample. Silently grading a
  model with itself is how benchmark numbers stop meaning anything, so
  the configuration is surfaced rather than hidden.
- **Judge parse failures stay visible.** A judge reply with no parseable
  `[[rating]]` marks that turn as `None` in the score metadata; a sample
  with no parseable turn scores 0 with `judge_parse_failed` set, instead
  of being dropped or quietly averaged.
- **Out of scope for v1:** pairwise comparison against the bundled GPT-4
  reference answers (FastChat's `gpt-4.jsonl`). The reference answers are
  used here only through the per-turn references embedded in the question
  file, as in the original single-answer grading.

## Evidence

`evidence/` holds an [Evalwarden](https://github.com/jurayh/evalwarden)
audit of a smoke-run log of this eval (mock model, fixture subset):
integrity score **95, verdict PASS**. The single medium finding
(COST-003, all spend on non-passing attempts) is an artefact of the mock
judge, whose replies contain no ratings; it doubles as a demonstration
that judge parse failures surface in cost reporting instead of
disappearing.

## Development

```bash
pip install -e .[dev]
pytest
```

Tests are fully offline: dataset shape against the pinned question file,
rating parsing, judge prompt construction, and an end-to-end run on
Inspect's mock model.

## Licensing and attribution

Code in this repository is MIT-licensed (see `LICENSE`). The MT-Bench
questions and judge prompts come from
[lm-sys/FastChat](https://github.com/lm-sys/FastChat) (Apache-2.0), pinned
by commit as described above. MT-Bench itself is by Zheng et al.; please
cite the paper when reporting results.
