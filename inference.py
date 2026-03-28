"""
inference.py — Entry-point inference script for the Customer Support OpenEnv.

Required by the OpenEnv validator at the repo root.

This script runs the built-in rule-based baseline agent against all 3 tasks
and prints scores to stdout. No API key is needed.

For LLM-powered inference, set OPENAI_API_KEY and pass --llm flag.

Usage:
    python inference.py             # rule-based (no API key)
    python inference.py --llm       # LLM agent (requires OPENAI_API_KEY)
"""

import os
import sys

# Ensure project root is importable (works both locally and in Docker)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline.run_baseline import (
    run_rule_based_baseline,
    run_llm_baseline,
    print_results_table,
)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Customer Support OpenEnv — Inference / Baseline Runner"
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use LLM agent (gpt-4o-mini). Requires OPENAI_API_KEY env var.",
    )
    args = parser.parse_args()

    if args.llm:
        print("Running LLM baseline (gpt-4o-mini)...")
        results = run_llm_baseline()
        print_results_table(results, "llm (gpt-4o-mini)")
    else:
        print("Running rule-based baseline (deterministic, no API key needed)...")
        results = run_rule_based_baseline()
        print_results_table(results, "rule-based")


if __name__ == "__main__":
    main()
