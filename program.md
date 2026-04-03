# ShoppingBench Agent Optimization

You are optimizing a ShoppingBench shopping agent to maximize its evaluation score.
The agent runs against an e-commerce product search benchmark with 2.5M real products.

## Your Task

Read the agent file, make ONE targeted change to improve its score, and save the file.
The agent file path is provided at the end of this prompt.

## What You Can Modify

- System prompts (task detection, search strategy, formatting instructions)
- Tool-calling logic (how search queries are constructed, result filtering, retries)
- Search strategy (keyword extraction, pagination, when to broaden/narrow)
- Task routing (how product/shop/voucher tasks are detected and handled)
- Step budget allocation (when to stop searching and recommend)
- Result verification logic (how the agent checks products before recommending)
- Fallback behavior (what happens when primary search fails)
- The balance between code-driven logic vs LLM-driven decisions
- Any helper functions, data structures, or algorithms within the file

## What You MUST NOT Break

- The `agent_main(problem_data: Dict) -> List[Dict]` function signature
- The dialogue step format (must use `create_dialogue_step()` from agent_interface)
- Tool calls must use `execute_tool_call()` from agent_interface
- Tools are registered with `@Tool` decorator from agent_interface
- Must call `recommend_product` before `terminate`
- Must call `terminate` at the end
- Imports from `src.agent.agent_interface` and `src.agent.proxy_client` must stay

## Scoring (what you're optimizing)

The score is **success_rate**: the fraction of problems where the agent fully
satisfies ALL constraints. This is binary per problem (pass/fail):

- **Product task**: passes when rule_score >= 1.0 (title similarity >= 0.7 via
  semantic embeddings, price in range, correct services, matching SKU/attributes)
- **Shop task**: passes when rule_score >= 1.0 AND all products from the SAME shop
- **Voucher task**: passes when rule_score >= 1.0 AND total after discount <= budget

Partial constraint satisfaction scores ZERO. A product that matches 4 out of 5
constraints is still a failure. The agent must find products that match ALL
requirements: correct product type, brand, attributes, price range, and services.

## Task Types

The agent faces three task types, detected from the query:

1. **Product**: Find ONE product matching constraints (price, brand, attributes, services)
2. **Shop**: Find multiple products ALL from the SAME shop
3. **Voucher**: Find products within a budget after applying a coupon discount

## Available Tools (registered via @Tool decorator)

- `find_product(q, page, shop_id, price, sort, service)` — Search products (max 10/page, 5 pages)
- `view_product_information(product_ids)` — Get full product details (attributes, SKU, description)
- `recommend_product(product_ids)` — Final recommendation (call once, before terminate)
- `terminate(status)` — End the dialogue ("success" or "failure")
- `check_product_match(product_id, requirements)` — Verify product matches attributes
- `find_products_in_same_shop(product_queries)` — Multi-product same-shop search
- `calculate_voucher(product_prices, voucher_type, discount_value, threshold, budget, cap)` — Budget math

## Static Analysis Constraints (CRITICAL)

All submitted agents pass through automated static analysis. Your changes MUST NOT
introduce any of these violations or the agent will be REJECTED on submission:

1. **No hardcoded product IDs**: Never embed specific product IDs as string literals.
2. **No obfuscation imports**: Never import base64, binascii, or codecs.
3. **No suite-specific dictionary mappings**: Do not create dicts that map terms
   from the problem set to synonyms or alternatives. Specifically:
   - No dicts with 3+ entries where keys or values match reward product title words
   - No dicts where both key AND value match words from the same problem context
   (e.g. {"bike": "bicycle"} where both words appear in a problem query/product)
4. **No verbatim problem phrases**: Do not include strings containing 6+ word
   phrases (30+ chars) that appear verbatim in problem queries or product titles.
5. **No dangerous imports**: Do not import os (except getenv/environ), subprocess,
   socket, http, requests, pickle, shutil, or ctypes.
6. **No dangerous calls**: Do not use eval(), exec(), or __import__().
7. **No file writes**: Do not use open() in write mode or pathlib write methods.

**What IS allowed:**
- Generic NLP stopwords (common English words like "the", "and", "for", etc.)
- Service filter strings ("official", "freeShipping", "COD", "flashsale") as these
  are API parameter values, not problem-specific content
- Task type keywords ("voucher", "budget", "discount", "shop") as these are
  category labels used for routing logic
- Regex patterns for parsing prices, numbers, and general text patterns
- Any helper functions, algorithms, or data structures that don't reference
  specific problem content

**Rule of thumb:** If a string in your code would only make sense if you had seen
the specific problems being tested, it will be flagged. Generic shopping logic is fine.

## Strategy Guidelines

- Make ONE change per iteration. Small, testable changes.
- If a change improved scores, build on it next iteration.
- If a change hurt scores, try a DIFFERENT approach — don't retry the same idea.
- Focus on the LOWEST-scoring category first (check per-category scores in experiment log).
- The agent uses a HYBRID approach: code handles mechanics (task routing, search execution,
  price/service extraction, voucher math), LLM handles query understanding (keyword extraction).
- The `inference()` function is available for making LLM calls via Chutes. Use it sparingly
  (1-3 calls per problem max) for tasks where code/regex is brittle, like understanding
  what the user actually wants to find.
- DO NOT remove LLM calls or revert to a pure code-driven approach. The LLM keyword
  extraction is critical for handling diverse query phrasings.
- Product search uses Lucene full-text search. Query quality matters enormously.
  The LLM should produce short, product-focused search terms (2-5 words).

## Output

After making your change, briefly describe what you changed and why in a single sentence
printed to stdout. Then the eval will run automatically.
