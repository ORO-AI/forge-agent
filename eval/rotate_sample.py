#!/usr/bin/env python3
"""
Rotate saturated problems out of the eval sample.

When a problem scores >= threshold (default 0.9 rule_score), swap it out
for a different problem from the same category that hasn't been saturated yet.

Usage:
    python3 autoresearch/rotate_sample.py \
        --eval-output '{"per_problem": [...]}' \
        --sample-file data/eval_3_sample.jsonl \
        --full-file data/eval_30.jsonl \
        --threshold 0.9
"""

import argparse
import json
import random
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-output", required=True, help="JSON string of eval results")
    parser.add_argument("--sample-file", required=True, help="Current sample JSONL file")
    parser.add_argument("--full-file", required=True, help="Full problem set JSONL file")
    parser.add_argument("--threshold", type=float, default=0.9, help="Score threshold for saturation")
    args = parser.parse_args()

    # Parse eval output
    try:
        eval_data = json.loads(args.eval_output)
    except json.JSONDecodeError:
        print("NO_CHANGE")
        return

    per_problem = eval_data.get("per_problem", [])
    if not per_problem:
        print("NO_CHANGE")
        return

    # Load current sample
    with open(args.sample_file) as f:
        sample_problems = [json.loads(line) for line in f if line.strip()]

    # Load full problem set
    with open(args.full_file) as f:
        all_problems = [json.loads(line) for line in f if line.strip()]

    # Find which sample queries are saturated
    saturated_queries = set()
    for result in per_problem:
        if result.get("rule_score", 0) >= args.threshold:
            saturated_queries.add(result["query"][:80])  # run_eval truncates to 80

    if not saturated_queries:
        print("NO_CHANGE")
        return

    # Get current sample queries for dedup
    current_queries = {p["query"] for p in sample_problems}

    # For each saturated problem, find a replacement from the same category
    swapped = 0
    new_sample = []

    for problem in sample_problems:
        query_prefix = problem["query"][:80]
        if query_prefix in saturated_queries:
            # Find the category of this problem
            category = problem.get("category", "").lower()
            if not category:
                # Detect from query
                q = problem["query"].lower()
                if "voucher" in q or "budget" in q:
                    category = "voucher"
                elif "shop" in q:
                    category = "shop"
                else:
                    category = "product"

            # Find replacement: same category, not already in sample
            candidates = [
                p for p in all_problems
                if p.get("category", "").lower() == category
                and p["query"] not in current_queries
            ]

            if candidates:
                replacement = random.choice(candidates)
                new_sample.append(replacement)
                current_queries.discard(problem["query"])
                current_queries.add(replacement["query"])
                swapped += 1
                # Print what we swapped for logging
                print(f"SWAPPED {category}: '{problem['query'][:50]}...' -> '{replacement['query'][:50]}...'",
                      file=sys.stderr)
            else:
                # No replacement available, keep it
                new_sample.append(problem)
        else:
            new_sample.append(problem)

    if swapped > 0:
        # Write updated sample
        with open(args.sample_file, "w") as f:
            for p in new_sample:
                f.write(json.dumps(p) + "\n")
        print(f"ROTATED_{swapped}")
    else:
        print("NO_CHANGE")


if __name__ == "__main__":
    main()
