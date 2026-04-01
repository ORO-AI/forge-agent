# Forge Agent

Forge is ORO's AI agent for [Subnet 15](https://docs.oroagents.com) on Bittensor. Every night, Forge runs an automated optimization loop using Claude to iteratively improve its ShoppingBench agent. The best agent is submitted to the subnet daily.

This repository contains Forge's agent code, updated on a 24-hour delay. Today's agent is already submitted. You'll see that code tomorrow.

## How It Works

1. **Midnight PT:** Forge kicks off an autoresearch loop (inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch))
2. **Overnight:** Claude iterates on `agent.py`, evaluating each change against ShoppingBench problems. Improvements are kept, regressions are reverted.
3. **11 AM PT:** Forge submits the best agent to the subnet (silent).
4. **12 PM PT:** Forge pushes yesterday's agent to this repo and narrates what it learned in the [SN15 Discord channel](https://discord.gg/bittensor).

## Repository Structure

```
agent.py                 # Yesterday's best agent (updated daily at 12 PM PT)
experiment_log.jsonl     # Yesterday's full iteration log
program.md               # Claude's iteration instructions
eval/
  run_eval.py            # Local evaluation script
  rotate_sample.py       # Problem rotation utility
history/
  YYYY-MM-DD/            # Daily snapshots
    agent.py
    experiment_log.jsonl
    summary.md           # Forge's daily narrative
docs/
  how-to-compete.md      # Guide for miners who want to fork this
```

## Compete With Forge

Want to beat Forge? See [docs/how-to-compete.md](docs/how-to-compete.md).

## License

MIT
