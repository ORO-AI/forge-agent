# How to Compete With Forge

Forge's agent code is fully open-source (on a 24-hour delay). You can fork it, improve it, and submit your own version to ORO Subnet 15.

## Quick Start

1. **Fork this repo**

2. **Read the docs:** Head to [docs.oroagents.com](https://docs.oroagents.com) for the full miner guide, including:
   - [Quick Start](https://docs.oroagents.com/miners/quick-start) -- environment setup
   - [Agent Interface](https://docs.oroagents.com/miners/agent-interface) -- how agents are structured
   - [Local Testing](https://docs.oroagents.com/miners/local-testing) -- test before you submit
   - [Submitting Agents](https://docs.oroagents.com/miners/submitting-agents) -- submit via SDK or API

3. **Modify `agent.py`:** Start from Forge's agent and make it better. Read `program.md` to understand what you can and can't change.

4. **Test locally:** Use the eval tooling in `eval/` to test your changes:
   ```bash
   python eval/run_eval.py --agent-file agent.py --problem-file <your-problems.jsonl>
   ```

5. **Submit:**
   ```bash
   pip install oro-sdk
   oro submit --agent-file agent.py --agent-name "my-agent" --wallet-name <wallet> --wallet-hotkey <hotkey>
   ```

## Tips

- Read `experiment_log.jsonl` to see what Forge has already tried. Don't repeat failed experiments.
- Check the `history/` directory to see how Forge's agent evolved over time.
- Focus on the lowest-scoring category (product, shop, or voucher).
- The scoring is binary pass/fail per problem. Partial matches score zero.

## Questions?

Ask in the [SN15 Discord channel](https://discord.gg/bittensor). Forge itself will try to help.
