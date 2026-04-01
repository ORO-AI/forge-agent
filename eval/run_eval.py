#!/usr/bin/env python3
"""
Evaluation runner for autoresearch.

Runs an agent against ShoppingBench problems and scores the results.
Outputs a JSON summary to stdout for the autoresearch loop to parse.

Usage:
    python autoresearch/run_eval.py \
        --agent-file autoresearch/agent.py \
        --problem-file data/eval_7_sample.jsonl \
        --max-workers 3 \
        --timeout 300
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Add src/agent to path so problem_scorer imports work
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "agent"))
sys.path.insert(0, str(REPO_ROOT))

from src.agent.sandbox_executor import execute_single_problem, load_problems

# Now problem_scorer can find rewards/ and util/
from problem_scorer import ProblemScorer

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def detect_task_type(query: str) -> str:
    """Detect task type from query content."""
    q = query.lower()
    if "voucher" in q or "budget" in q:
        return "voucher"
    if "shop" in q and any(kw in q for kw in ["both", "these", "offering", "sells", "offers"]):
        return "shop"
    if re.search(r"shops?\s+(?:offering|that\s+offer|selling)", q):
        return "shop"
    has_ord = bool(re.search(r"\b(?:first|second|third|lastly)\b", q))
    if has_ord and "budget" in q:
        return "voucher"
    if has_ord:
        return "shop"
    return "product"


def load_problems_with_rewards(problem_file: str) -> list[dict]:
    """Load problems from JSONL, keeping rewards and vouchers for scoring."""
    problems = []
    with open(problem_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                problem = json.loads(line)
                if "query" in problem:
                    problems.append(problem)
            except json.JSONDecodeError:
                continue
    return problems


def run_and_score(
    problems: list[dict],
    agent_file: str,
    max_workers: int = 3,
    timeout: float = 300.0,
) -> dict:
    """Run agent against problems and score results.

    Returns JSON-serializable dict with scores.
    """
    # Group problems by task type for scoring
    task_groups: dict[str, list[dict]] = defaultdict(list)
    for p in problems:
        task = detect_task_type(p["query"])
        task_groups[task].append(p)

    # Build scorer per task type
    scorers: dict[str, ProblemScorer] = {}
    for task, task_problems in task_groups.items():
        rewards = {}
        vouchers = {}
        for p in task_problems:
            query = p["query"]
            if "reward" in p:
                rewards[query] = p["reward"]
            if "voucher" in p:
                vouchers[query] = p["voucher"]
        scorers[task] = ProblemScorer(task=task, rewards=rewards, vouchers=vouchers)

    # Run agent against each problem (strip rewards first)
    all_scores = []
    per_category = defaultdict(list)
    failed_count = 0
    timeout_count = 0

    for p in problems:
        query = p["query"]
        task = detect_task_type(query)

        # Strip reward/voucher before giving to agent
        agent_input = {k: v for k, v in p.items() if k not in ("reward", "voucher")}

        # Execute
        result = execute_single_problem(
            problem=agent_input,
            timeout=timeout,
            agent_file=agent_file,
        )

        if not result.success:
            failed_count += 1
            if "timeout" in (result.error or "").lower():
                timeout_count += 1
            score_entry = {
                "query": query[:80],
                "task": task,
                "success": False,
                "error": result.error,
                "rule_score": 0.0,
                "format_score": 0.0,
                "gt_score": 0.0,
                "execution_time": result.execution_time,
            }
            all_scores.append(score_entry)
            per_category[task].append(score_entry)
            continue

        # Score
        scorer = scorers[task]
        score = scorer.score_problem(query=query, output=result.result)

        if score is None:
            score_entry = {
                "query": query[:80],
                "task": task,
                "success": True,
                "error": "no reward data for scoring",
                "rule_score": 0.0,
                "format_score": 0.0,
                "gt_score": 0.0,
                "execution_time": result.execution_time,
            }
        else:
            score_entry = {
                "query": query[:80],
                "task": task,
                "success": True,
                "rule_score": score.get("rule", 0.0),
                "format_score": score.get("format", 0.0),
                "gt_score": score.get("gt", 0.0),
                "length_score": score.get("length", 0.0),
                "product_score": score.get("product", 0.0),
                "shop_score": score.get("shop", 0.0),
                "budget_score": score.get("budget", 0.0),
                "execution_time": result.execution_time,
            }

        all_scores.append(score_entry)
        per_category[task].append(score_entry)

    # Aggregate using same logic as validator (success_rate)
    n = len(all_scores)
    avg = lambda key: sum(s.get(key, 0.0) for s in all_scores) / n if n > 0 else 0.0

    # Compute success_rate: same logic as validator progress_reporter._compute_aggregate
    success_count = 0
    for s in all_scores:
        task = s.get("task", "product")
        rule = s.get("rule_score", 0)
        rule_ok = rule >= 1.0
        if task == "product" and rule_ok:
            success_count += 1
        elif task == "shop" and rule_ok and s.get("shop_score", 0) >= 1:
            success_count += 1
        elif task == "voucher" and rule_ok and s.get("budget_score", 0) >= 1:
            success_count += 1

    success_rate = success_count / n if n > 0 else 0.0

    category_summaries = {}
    for task, scores in per_category.items():
        tn = len(scores)
        cat_success = 0
        for s in scores:
            rule_ok = s.get("rule_score", 0) >= 1.0
            if task == "product" and rule_ok:
                cat_success += 1
            elif task == "shop" and rule_ok and s.get("shop_score", 0) >= 1:
                cat_success += 1
            elif task == "voucher" and rule_ok and s.get("budget_score", 0) >= 1:
                cat_success += 1
        category_summaries[task] = {
            "count": tn,
            "success_rate": cat_success / tn if tn > 0 else 0,
            "avg_rule_score": sum(s.get("rule_score", 0) for s in scores) / tn if tn > 0 else 0,
            "avg_gt_score": sum(s.get("gt_score", 0) for s in scores) / tn if tn > 0 else 0,
        }

    return {
        "total_problems": n,
        "failed": failed_count,
        "timed_out": timeout_count,
        "success_rate": round(success_rate, 4),
        "success_count": success_count,
        "avg_rule_score": round(avg("rule_score"), 4),
        "avg_gt_score": round(avg("gt_score"), 4),
        "categories": category_summaries,
        "per_problem": all_scores,
    }


def main():
    parser = argparse.ArgumentParser(description="Run ShoppingBench evaluation")
    parser.add_argument("--agent-file", required=True, help="Path to agent.py")
    parser.add_argument("--problem-file", required=True, help="Path to eval JSONL")
    parser.add_argument("--max-workers", type=int, default=3, help="Parallel workers")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-problem timeout")
    args = parser.parse_args()

    # Ensure proxy URL is set
    if "SANDBOX_PROXY_URL" not in os.environ:
        os.environ["SANDBOX_PROXY_URL"] = "http://localhost:8080"

    problems = load_problems_with_rewards(args.problem_file)
    if not problems:
        print(json.dumps({"error": "No problems loaded", "avg_gt_score": 0.0}))
        sys.exit(1)

    results = run_and_score(
        problems=problems,
        agent_file=args.agent_file,
        max_workers=args.max_workers,
        timeout=args.timeout,
    )

    # Output JSON to stdout (autoresearch.sh reads this)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
