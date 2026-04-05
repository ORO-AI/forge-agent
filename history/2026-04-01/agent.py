import json
import logging
import re
from os import getenv
from typing import Dict, List, Optional
from urllib.parse import quote_plus

from src.agent.agent_interface import (
    create_dialogue_step,
    execute_tool_call,
    Tool,
    generate_tool_call_id,
)
from src.agent.proxy_client import ProxyClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import hashlib as _hl
import struct as _su

_inference_client = ProxyClient(timeout=30, max_retries=3)
_search_client = ProxyClient(timeout=15, max_retries=1)


@Tool
def find_product(
    q: str, page: int = 1, shop_id: Optional[str] = None,
    price: Optional[str] = None, sort: Optional[str] = None,
    service: Optional[str] = None,
) -> List[Dict]:
    """Search for products matching query."""
    params = {"q": quote_plus(q), "page": page, "shop_id": shop_id,
              "price": price, "sort": sort, "service": service}
    if params.get("sort") == "default":
        params.pop("sort")
    if params.get("service") == "default":
        params.pop("service")
    elif params.get("service") and "default" in params["service"]:
        params["service"] = ",".join(
            x for x in params["service"].split(",") if x != "default")
    result = _search_client.get("/search/find_product", params)
    result = result if result is not None else []
    if shop_id and not result:
        retry = dict(params)
        retry.pop("service", None)
        result = _search_client.get("/search/find_product", retry) or []
    return result


@Tool
def find_products_in_same_shop(product_queries: str) -> Dict:
    """Find multiple products from the SAME shop."""
    try:
        specs = json.loads(product_queries) if isinstance(product_queries, str) else product_queries
    except json.JSONDecodeError:
        return {"found": False, "error": "Invalid JSON"}
    if not specs or not isinstance(specs, list):
        return {"found": False, "error": "Need non-empty list"}

    orig_query = ""
    if isinstance(specs[-1], dict) and specs[-1].get("_original_query"):
        orig_query = specs.pop()["_original_query"]

    broad_results = []
    for spec in specs:
        q = spec.get("q", "")
        params = {"q": quote_plus(q), "page": 1}
        if spec.get("price"):
            params["price"] = spec["price"]
        if spec.get("service"):
            params["service"] = spec["service"]
        results = _search_client.get("/search/find_product", params) or []
        broad_results.append(results)

    if not any(broad_results):
        return {"found": False, "error": "No results for any product", "shops_tried": 0}

    from collections import defaultdict as _dd
    shop_cov = _dd(lambda: _dd(list))
    for idx, results in enumerate(broad_results):
        for prod in results:
            sid = str(prod.get("shop_id", ""))
            if sid:
                shop_cov[sid][idx].append(prod)

    def _score_shop_coverage(sid):
        cov = shop_cov[sid]
        n_covered = len(cov)
        total_score = 0
        for idx, prods in cov.items():
            q = specs[idx].get("q", "") if idx < len(specs) else ""
            total_score += max((_score_product_relevance(p, orig_query or q) for p in prods), default=0)
        return (n_covered, total_score)

    candidates = sorted(shop_cov.keys(), key=_score_shop_coverage, reverse=True)

    max_shops = 3 if len(specs) >= 3 else 5
    for shop_id in candidates[:max_shops]:
        cov = shop_cov[shop_id]
        found = []
        ok = True
        for idx, spec in enumerate(specs):
            q = spec.get("q", "")
            score_q = orig_query or q
            if idx in cov and cov[idx]:
                best = _select_best_product(cov[idx], q or score_q, prefer_cheaper=True)
                if best:
                    found.append(best)
                    continue
            params = {"q": quote_plus(q), "page": 1, "shop_id": shop_id}
            results = _search_client.get("/search/find_product", params) or []
            if results:
                best = _select_best_product(results, q or score_q, prefer_cheaper=True)
                if best:
                    found.append(best)
                    continue
            ok = False
            break
        if ok and len(found) == len(specs):
            return {
                "found": True, "shop_id": shop_id,
                "products": [{"product_id": p.get("product_id"), "title": p.get("title", ""),
                              "price": p.get("price"), "shop_id": p.get("shop_id")}
                             for p in found],
                "shops_tried": candidates.index(shop_id) + 1,
            }

    return {"found": False, "error": f"No shop has all {len(specs)} products",
            "shops_tried": min(len(candidates), max_shops)}


@Tool
def recommend_product(product_ids: str) -> str:
    """Recommend products to the user."""
    return f"Having recommended the products to the user: {product_ids}."


@Tool
def terminate(status: str = "success") -> str:
    """End the dialogue."""
    return f"The interaction has been completed with status: {status}"


def _deduplicate_ids(ids: list) -> list:
    pass
    seen = set()
    out = []
    for pid in ids:
        pid = str(pid).strip()
        if pid and pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def _format_product_ids(ids: list, expected_order: list = None) -> str:
    pass
    ids = _deduplicate_ids(ids)
    if expected_order:
        known = {pid: i for i, pid in enumerate(expected_order)}
        ids = sorted(ids, key=lambda p: known.get(p, len(expected_order)))
    return ",".join(ids) if ids else ""


_STOPWORDS = frozenset(
    "the a an for with from that this i me my looking show find want need get "
    "buy also and in is it am im priced pesos php price between than above below "
    "more less over under of to or on at by its be can has have will would should "
    "products product items both these offering sells shop budget voucher discount "
    "first second third made using available support supports compatible please "
    "looking".split()
)


_product_detail_cache = {}


def _fetch_product_details(product_ids):
    pass
    if not product_ids:
        return {}
    uncached = [pid for pid in product_ids if pid not in _product_detail_cache]
    if uncached:
        ids_str = ",".join(uncached[:10])
        result = _search_client.get("/search/view_product_information", {"product_ids": ids_str})
        if result and isinstance(result, list):
            for p in result:
                _product_detail_cache[str(p.get("product_id", ""))] = p
    return {pid: _product_detail_cache[pid] for pid in product_ids if pid in _product_detail_cache}


def _score_product_relevance(product, query_text, detail=None):
    pass
    title = product.get("title", "").lower()
    t_words = set(re.findall(r"\b\w+\b", title))
    q_words = list(dict.fromkeys(w for w in re.findall(r"\b\w+\b", query_text.lower())
                                  if w not in _STOPWORDS and len(w) > 1))

    score = 0
    for qw in q_words:
        if qw in t_words:
            score += 2
        elif qw.endswith("s") and qw[:-1] in t_words:
            score += 2
        elif not qw.endswith("s") and (qw + "s") in t_words:
            score += 2
        elif len(qw) >= 3 and any(tw.startswith(qw) for tw in t_words if len(tw) > len(qw)):
            score += 2
        elif any(qw.startswith(tw) or tw.startswith(qw)
                 for tw in t_words if len(tw) > 2):
            score += 1
        if any(c.isdigit() for c in qw) and qw in title:
            score += 2

    if detail:
        attr_text = ""
        exact_values = set()
        for k, vs in (detail.get("attributes") or {}).items():
            attr_text += " " + k.replace("_", " ")
            for v in (vs if isinstance(vs, list) else [vs]):
                v_str = str(v).strip().lower()
                attr_text += " " + v_str
                exact_values.add(v_str)
        for sku_id, opts in (detail.get("sku_options") or {}).items():
            if isinstance(opts, dict):
                for k, v in opts.items():
                    v_str = str(v).strip().lower()
                    attr_text += " " + k.replace("_", " ") + " " + v_str
                    exact_values.add(v_str)
        attr_lower = attr_text.lower()
        attr_words = set(re.findall(r"\b\w+\b", attr_lower))

        for qw in q_words:
            if qw in exact_values:
                score += 3
            elif (qw + "#") in exact_values:
                score += 5
            elif qw in attr_words:
                score += 2

    return score


def _select_best_product(products, query_text, prefer_cheaper=False):
    pass
    if not products:
        return None
    products_scored = sorted(products, key=lambda p: _score_product_relevance(p, query_text), reverse=True)
    top = products_scored[:10]
    pids = [str(p.get("product_id", "")) for p in top if p.get("product_id")]
    details = _fetch_product_details(pids)
    def _final_score(p):
        s = _score_product_relevance(p, query_text, details.get(str(p.get("product_id", ""))))
        if prefer_cheaper:
            s -= (p.get("price", 0) or 0) / 100000
        return s
    return max(top, key=_final_score)


_EXTRACTION_PROMPT = """Extract search params as JSON. No markdown.
{"task_type":"product"|"shop"|"voucher","products":[{"keywords":"search query","price_range":"min-max"|null,"service":"official"|"freeShipping"|"COD"|"flashsale"|null}],"is_shop_voucher":bool}
- keywords: product type + brand + material + color + size + use. 3-6 words, include ALL qualifying terms.
- price_range: "100-500", "100-", "0-500". null if none.
- service: LazMall=official, free shipping=freeShipping, COD=COD, flash sale=flashsale. null if none.
- task_type: product=single, shop=same-shop multi, voucher=budget/discount.
- Multi-product: one entry per product, preserve order. Budget/voucher info are NOT products.
- is_shop_voucher: true if "same shop" voucher.
JSON only:"""


def _detect_task_type(query: str) -> str:
    pass
    q = query.lower()
    if "voucher" in q or "budget" in q or "discount" in q:
        return "voucher"
    if "shop" in q and any(w in q for w in ("both", "these", "offering", "sells", "same")):
        return "shop"
    return "product"


def _extract_search_params_with_llm(query: str) -> dict:
    pass
    model = getenv("SANDBOX_MODEL", "deepseek-ai/DeepSeek-V3.2-TEE")
    result = _inference_client.post("/inference/chat/completions", json_data={
        "model": model, "temperature": 0, "stream": False,
        "messages": [
            {"role": "system", "content": _EXTRACTION_PROMPT},
            {"role": "user", "content": query},
        ],
    })
    if result and "choices" in result and result["choices"]:
        content = result["choices"][0].get("message", {}).get("content", "")
        cleaned = re.sub(r"```json?\s*", "", content)
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
    return _extract_search_params_fallback(query)


def _extract_search_params_fallback(query: str) -> dict:
    pass
    task_type = _detect_task_type(query)
    stop = {"the", "and", "for", "with", "from", "that", "this", "are", "was", "can",
            "has", "have", "been", "will", "find", "looking", "show", "want", "need",
            "get", "buy", "product", "products", "search", "same", "shop", "within",
            "budget", "voucher", "discount", "price", "priced", "pesos", "php",
            "between", "than", "greater", "less", "more", "under", "over", "about",
            "also", "both", "these", "them", "each", "all", "any", "one", "two",
            "three", "four", "five", "offering", "sells", "using", "in", "is", "it",
            "its", "or", "at", "on", "by", "be", "do", "an", "my", "me", "im",
            "items", "item", "only", "just", "first", "second", "supports", "support",
            "compatible", "available", "made", "please", "like", "of", "above",
            "deals", "options", "option", "delivery", "shipping", "offers",
            "lazmall", "lazflash", "official", "cash", "payment", "pay",
            "cost", "costs", "via", "themed", "such", "those", "store", "stores",
            "focus", "category", "specifically", "guaranteed", "authenticity",
            "returns", "quick", "perks", "should", "help", "purchase", "type",
            "to", "named", "called", "family", "belongs", "comes", "another",
            "lastly", "benefits", "you", "weighing", "capacity",
            "size", "sized", "eu", "fits"}

    def _parse_product_spec(text):
        alpha_words = [w for w in re.findall(r"\b[a-zA-Z]{2,}\b", text.lower()) if w not in stop]
        alnum_tokens = re.findall(r"\b\d+[a-zA-Z]+\b|\b[a-zA-Z]+\d+[a-zA-Z]*\b", text.lower())
        words = alpha_words[:6]
        for t in alnum_tokens[:2]:
            if t not in words:
                words.append(t)
        shade_nums = re.findall(r'(\d+)#', text)
        for s in shade_nums[:2]:
            if s not in words:
                words.append(s)
        keywords = " ".join(words) or "product"
        price_range = None
        m = re.search(r"(?:greater|more|over|above|>|cost[s]?\s+more)\s*(?:than\s*)?(\d+)", text, re.I)
        if m:
            price_range = f"{m.group(1)}-"
        else:
            m = re.search(r"(\d{1,6})\s*(?:to|and|-)\s*(\d{1,6})\s*(?:pesos|php)", text, re.I)
            if m:
                price_range = f"{m.group(1)}-{m.group(2)}"
            elif re.search(r"(?:price|pesos|php|cost)", text, re.I):
                m = re.search(r"(\d{1,6})\s+(?:to|and)\s+(\d{1,6})", text)
                if m:
                    price_range = f"{m.group(1)}-{m.group(2)}"
        service = None
        tl = text.lower()
        if "lazmall" in tl or "official" in tl:
            service = "official"
        if "free shipping" in tl or "free delivery" in tl:
            service = "freeShipping" if not service else f"{service},freeShipping"
        if "lazflash" in tl or "flash sale" in tl or "flashsale" in tl:
            service = "flashsale" if not service else f"{service},flashsale"
        if "cash on delivery" in tl or "cod" in tl:
            service = "COD" if not service else f"{service},COD"
        return {"keywords": keywords, "price_range": price_range, "service": service}

    product_text = re.split(r"(?:My budget|budget is|I have a voucher)", query, flags=re.I)[0].strip()
    if not product_text or len(product_text) < 15:
        product_text = query

    parts = re.split(
        r"(?:,?\s*and\s+also\s+|,?\s*also,?\s+|Second(?:ly)?,\s*|Third(?:ly)?,\s*"
        r"|First,\s*|\(\d+\)\s*|\d+\.\s*|Additionally,\s*"
        r"|[.]\s*Next,\s*|[.]\s*Lastly,\s*|[.]\s*Finally,\s*|[.]\s*Last,\s*)",
        product_text, flags=re.I
    )
    parts = [p.strip() for p in parts if p and len(p.strip()) > 10]
    if not parts:
        parts = [query]

    products = [_parse_product_spec(p) for p in parts]
    products = [p for p in products if len(p["keywords"].split()) >= 2] or products
    is_shop = task_type == "shop" or (
        task_type == "voucher" and "same shop" in query.lower())

    return {
        "task_type": task_type,
        "products": products,
        "is_shop_voucher": is_shop,
    }


def _add_dialogue_step(think, tool_results, response, query, steps):
    pass
    steps.append(create_dialogue_step(think, tool_results, response, query, len(steps) + 1))


def _handle_single_product(params, query, steps):
    pass
    prods = params.get("products", [{}])
    p = prods[0] if prods else {}
    kw = p.get("keywords", "product")
    all_results = []

    search_p = {"q": kw}
    if p.get("price_range"):
        search_p["price"] = p["price_range"]
    if p.get("service"):
        search_p["service"] = p["service"]
    result = execute_tool_call("find_product", search_p)
    _add_dialogue_step("Processing.", [result], "", query, steps)
    all_results.extend(result["result"] or [])

    seen = set()
    unique = []
    for prod in all_results:
        pid = str(prod.get("product_id", ""))
        if pid and pid not in seen:
            seen.add(pid)
            unique.append(prod)

    best = _select_best_product(unique, query) if unique else None
    pid = str(best["product_id"]) if best else ""
    rec_str = _format_product_ids([pid])
    rec = execute_tool_call("recommend_product", {"product_ids": rec_str})
    term = execute_tool_call("terminate", {"status": "success"})
    _add_dialogue_step("Done.", [rec, term], "Done.", query, steps)


def _handle_same_shop_search(params, query, steps, is_voucher=False):
    pass
    queries = []
    for p in params.get("products", []):
        q = {"q": p.get("keywords", "product")}
        if p.get("price_range") and not is_voucher:
            q["price"] = p["price_range"]
        if p.get("service"):
            q["service"] = p["service"]
        queries.append(q)
    if not queries:
        queries = [{"q": "product"}]
    queries.append({"_original_query": query})

    result = execute_tool_call("find_products_in_same_shop",
                               {"product_queries": json.dumps(queries)})
    _add_dialogue_step("Processing.", [result], "", query, steps)

    shop_result = result["result"]
    pids = []
    if isinstance(shop_result, dict) and shop_result.get("found"):
        pids = [str(p["product_id"]) for p in shop_result["products"]]
    else:
        for p in params.get("products", []):
            try:
                kw = p.get("keywords", "product")
                sp = {"q": kw}
                if p.get("price_range") and not is_voucher:
                    sp["price"] = p["price_range"]
                r = execute_tool_call("find_product", sp)
                _add_dialogue_step("Processing.", [r], "", query, steps)
                if r["result"]:
                    best = _select_best_product(r["result"], query, prefer_cheaper=True)
                    if best:
                        pids.append(str(best["product_id"]))
            except Exception:
                pass

    rec_str = _format_product_ids(pids)
    rec = execute_tool_call("recommend_product", {"product_ids": rec_str})
    term = execute_tool_call("terminate", {"status": "success"})
    _add_dialogue_step("Done.", [rec, term], "Done.", query, steps)


def _handle_voucher_search(params, query, steps):
    pass
    is_shop = params.get("is_shop_voucher", False)
    if not is_shop and "same shop" in query.lower():
        is_shop = True
    products = params.get("products", [])

    if is_shop and len(products) > 1:
        _handle_same_shop_search(params, query, steps, is_voucher=True)
        return

    pids = []
    for p in products:
        kw = p.get("keywords", "product")
        sp = {"q": kw}
        if p.get("service"):
            sp["service"] = p["service"]
        result = execute_tool_call("find_product", sp)
        _add_dialogue_step("Processing.", [result], "", query, steps)
        found = result["result"] or []
        if found:
            score_q = kw if len(products) > 1 else query
            best = _select_best_product(found, score_q, prefer_cheaper=True)
            if best:
                pids.append(str(best["product_id"]))

    rec_str = _format_product_ids(pids)
    rec = execute_tool_call("recommend_product", {"product_ids": rec_str})
    term = execute_tool_call("terminate", {"status": "success"})
    _add_dialogue_step("Done.", [rec, term], "Done.", query, steps)


def agent_main(problem_data: dict) -> List[dict]:
    pass
    _product_detail_cache.clear()
    steps = []
    query = problem_data.get("query", "")

    try:
        params = _extract_search_params_with_llm(query)
        task_type = (params.get("task_type", "") or "").lower() or _detect_task_type(query)
        kw_task = _detect_task_type(query)
        if kw_task != "product" and task_type == "product":
            task_type = kw_task
        n_products = len(params.get("products", []))
        _add_dialogue_step("Processing.", [], "", query, steps)

        if task_type == "shop":
            _handle_same_shop_search(params, query, steps)
        elif task_type == "voucher":
            _handle_voucher_search(params, query, steps)
        else:
            _handle_single_product(params, query, steps)

    except Exception as e:
        try:
            rec = execute_tool_call("recommend_product", {"product_ids": ""})
            term = execute_tool_call("terminate", {"status": "failure"})
            _add_dialogue_step("Processing.", [rec, term], "Done.", query, steps)
        except Exception:
            steps.append(create_dialogue_step(
                "Done.", [], "Done.", query, len(steps) + 1))

    if not steps:
        steps.append(create_dialogue_step("Done.", [], "Done.", query, 1))

    return steps
