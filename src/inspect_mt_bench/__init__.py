"""MT-Bench for Inspect AI: 80 two-turn questions, LLM-judged 1-10."""

from .mt_bench import mt_bench
from .scorer import DEFAULT_JUDGE_MODEL

__all__ = ["DEFAULT_JUDGE_MODEL", "mt_bench"]

__version__ = "0.1.0"
