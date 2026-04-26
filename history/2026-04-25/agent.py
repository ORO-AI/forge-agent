"""Shopping agent: turn natural-language queries into search params, run tools, pick products.

Flow: extract structured params (LLM with JSON fallback) -> route by task (product / shop /
voucher) -> find_product and/or find_products_in_same_shop -> recommend_product + terminate.
Relevance scoring uses title/detail tokens plus light heuristics (material, color, size).
"""

import json
import logging
import re
from collections import defaultdict
from os import getenv
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set
from urllib.parse import quote_plus

from src.agent.agent_interface import Tool, create_dialogue_step, execute_tool_call
from src.agent.proxy_client import ProxyClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cap logged user query length (full query is still processed).
_LOG_QUERY_PREVIEW = 240

# --- Types & shared constants ---
Product = Dict[str, Any]
SearchSpec = Dict[str, Any]

DEFAULT_PRODUCT_QUERY = "product"
MAX_DETAIL_LOOKUP_PRODUCTS = 20
TOP_RELEVANCE_CANDIDATES = 20
CHEAPER_PRICE_TIEBREAK_DIVISOR = 100_000
MAX_SHOPS_WIDE_QUERY = 6
MAX_SHOPS_FOR_TWO_OR_FEWER_SPECS = 8

# --- API clients & compile-time patterns ---
_inference_client = ProxyClient(timeout=30, max_retries=3)
_search_client = ProxyClient(timeout=15, max_retries=1)

_PRODUCT_TEXT_SPLIT_RE = re.compile(
    r"(?:,?\s*and\s+also\s+|,?\s*also,?\s+|Second(?:ly)?,\s*|Third(?:ly)?,\s*"
    r"|First,\s*|\(\d+\)\s*|\d+\.\s*|Additionally,\s*"
    r"|[.]\s*Next,\s*|[.]\s*Lastly,\s*|[.]\s*Finally,\s*|[.]\s*Last,\s*)",
    re.I,
)
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

# Populated per agent_main run; avoids duplicate detail API calls while ranking candidates.
_product_detail_cache: Dict[str, Product] = {}


def _preview_query(q: str, limit: int = _LOG_QUERY_PREVIEW) -> str:
    """Return a safe one-line preview for logs (length-limited)."""
    text = " ".join(q.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


# ---------------------------------------------------------------------------
# Search helpers (params, fetch, dedupe, serialization)
# ---------------------------------------------------------------------------


def _normalize_service(service: Optional[str]) -> Optional[str]:
    """Map UI 'default' / comma lists to API service filter string."""
    if not service:
        return service
    if service == "default":
        return None

    services = [part.strip() for part in service.split(",") if part.strip() and part.strip() != "default"]
    return ",".join(services) or None


def _build_search_params(
    query: str,
    *,
    page: int = 1,
    shop_id: Optional[str] = None,
    price: Optional[str] = None,
    sort: Optional[str] = None,
    service: Optional[str] = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"q": quote_plus(query), "page": page}
    if shop_id:
        params["shop_id"] = shop_id
    if price:
        params["price"] = price
    if sort and sort != "default":
        params["sort"] = sort
    normalized_service = _normalize_service(service)
    if normalized_service:
        params["service"] = normalized_service
    return params


def _search_products(params: Dict[str, Any]) -> List[Product]:
    return _search_client.get("/search/find_product", params) or []


def _search_products_for_spec(
    spec: SearchSpec,
    *,
    shop_id: Optional[str] = None,
    include_price: bool = True,
    omit_service_from_api: bool = False,
) -> List[Product]:
    """Search using spec q/keywords/price; optionally drop service from the API call (widens results)."""
    price = None
    if include_price:
        price = spec.get("price")
        if price is None:
            price = spec.get("price_range")

    service = None if omit_service_from_api else spec.get("service")

    return _search_products(
        _build_search_params(
            spec.get("q") or spec.get("keywords") or DEFAULT_PRODUCT_QUERY,
            shop_id=shop_id,
            price=price,
            service=service,
        )
    )


def _product_matches_services(product: Product, service_spec: Optional[str]) -> bool:
    """True if product's service tags include every comma-separated requirement (e.g. official,COD)."""
    if not service_spec:
        return True
    required = [part.strip() for part in str(service_spec).split(",") if part.strip()]
    if not required:
        return True
    offered = product.get("service") or []
    if not isinstance(offered, list):
        offered = []
    return all(req in offered for req in required)


def _filter_products_by_spec_services(products: Sequence[Product], spec: SearchSpec) -> List[Product]:
    service_spec = spec.get("service")
    if not service_spec:
        return list(products)
    return [product for product in products if _product_matches_services(product, service_spec)]


def _serialize_products(products: Sequence[Product]) -> List[Dict[str, Any]]:
    return [
        {
            "product_id": product.get("product_id"),
            "title": product.get("title", ""),
            "price": product.get("price"),
            "shop_id": product.get("shop_id"),
        }
        for product in products
    ]


def _deduplicate_ids(ids: Iterable[Any]) -> List[str]:
    seen = set()
    unique_ids: List[str] = []
    for product_id in ids:
        normalized = str(product_id).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_ids.append(normalized)
    return unique_ids


def _deduplicate_products(products: Iterable[Product]) -> List[Product]:
    seen = set()
    unique_products: List[Product] = []
    for product in products:
        product_id = str(product.get("product_id", "")).strip()
        if product_id and product_id not in seen:
            seen.add(product_id)
            unique_products.append(product)
    return unique_products


def _format_product_ids(ids: Iterable[Any], expected_order: Optional[Sequence[str]] = None) -> str:
    normalized_ids = _deduplicate_ids(ids)
    if expected_order:
        known_positions = {str(product_id): index for index, product_id in enumerate(expected_order)}
        normalized_ids = sorted(
            normalized_ids,
            key=lambda product_id: known_positions.get(product_id, len(expected_order)),
        )
    return ",".join(normalized_ids)


def _fetch_product_details(product_ids: Sequence[str]) -> Dict[str, Product]:
    """Batch-fetch product detail for relevance; fills module cache (max MAX_DETAIL_LOOKUP_PRODUCTS)."""
    if not product_ids:
        return {}

    uncached_ids = [product_id for product_id in product_ids if product_id not in _product_detail_cache]
    if uncached_ids:
        result = _search_client.get(
            "/search/view_product_information",
            {"product_ids": ",".join(uncached_ids[:MAX_DETAIL_LOOKUP_PRODUCTS])},
        )
        if isinstance(result, list):
            for product in result:
                product_id = str(product.get("product_id", ""))
                if product_id:
                    _product_detail_cache[product_id] = product

    return {
        product_id: _product_detail_cache[product_id]
        for product_id in product_ids
        if product_id in _product_detail_cache
    }


# ---------------------------------------------------------------------------
# Relevance scoring
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "the a an for with from that this i me my looking show find want need get "
    "buy also and in is it am im priced pesos php price between than above below "
    "more less over under of to or on at by its be can has have will would should "
    "products product items both these offering sells shop budget voucher discount "
    "first second third made using available support supports compatible please "
    "looking".split()
)


def _extract_query_words(query_text: str) -> List[str]:
    return list(
        dict.fromkeys(
            word
            for word in re.findall(r"\b\w+\b", query_text.lower())
            if word not in _STOPWORDS and len(word) > 1
        )
    )


def _build_detail_search_text(detail: Product) -> tuple[str, set[str]]:
    tokens: List[str] = []
    exact_values: set[str] = set()

    for key, values in (detail.get("attributes") or {}).items():
        tokens.append(key.replace("_", " "))
        for value in values if isinstance(values, list) else [values]:
            value_str = str(value).strip().lower()
            tokens.append(value_str)
            exact_values.add(value_str)

    for options in (detail.get("sku_options") or {}).values():
        if isinstance(options, dict):
            for key, value in options.items():
                value_str = str(value).strip().lower()
                tokens.append(key.replace("_", " "))
                tokens.append(value_str)
                exact_values.add(value_str)

    return " ".join(tokens).lower(), exact_values


def _relevance_query_title_adjustments(query_lower: str, title_lower: str) -> float:
    """Heuristic bonuses/penalties (non-keyword) so obvious mismatches rank lower."""
    adj = 0.0
    if "deodorant" in query_lower and "pet" not in query_lower:
        if any(
            phrase in title_lower
            for phrase in (
                "pet ",
                " dog ",
                " cat ",
                "pipeline",
                "drain",
                "septic",
                "helmet",
                "sewage",
            )
        ):
            adj -= 18.0
    if "glass" in query_lower:
        if "glass" in title_lower:
            adj += 10.0
        elif "plastic" in title_lower and "glass" not in title_lower:
            adj -= 14.0
    color_words = (
        "blue",
        "red",
        "green",
        "black",
        "white",
        "grey",
        "gray",
        "pink",
        "violet",
        "brown",
        "yellow",
        "orange",
    )
    for color in color_words:
        if color in query_lower:
            if color in title_lower:
                adj += 5.0
            elif any(x in title_lower for x in ("clear", "transparent", "plating clear", "matte clear")):
                adj -= 8.0
    for match in re.finditer(r"iphone\s*(\d+\s*(?:pro\s*max|plus|pro|max|mini)?)", query_lower):
        slug = re.sub(r"\s+", "", match.group(0))
        if slug and slug in re.sub(r"\s+", "", title_lower):
            adj += 12.0
    for match in re.finditer(r"(\d+)\s*ml", query_lower):
        if match.group(1) in title_lower:
            adj += 6.0
    for match in re.finditer(r"eu[:\s]*(\d+)", query_lower, re.I):
        if match.group(1) in title_lower or f"size {match.group(1)}" in title_lower:
            adj += 8.0
    return adj


def _score_product_relevance(
    product: Product,
    query_text: str,
    detail: Optional[Product] = None,
) -> float:
    """Higher score = better match to query (title overlap + optional detail attributes)."""
    title = product.get("title", "").lower()
    title_words = set(re.findall(r"\b\w+\b", title))
    query_words = _extract_query_words(query_text)
    query_lower = query_text.lower()

    score = _relevance_query_title_adjustments(query_lower, title)
    for query_word in query_words:
        if query_word in title_words:
            score += 2
        elif query_word.endswith("s") and query_word[:-1] in title_words:
            score += 2
        elif not query_word.endswith("s") and f"{query_word}s" in title_words:
            score += 2
        elif len(query_word) >= 3 and any(
            title_word.startswith(query_word)
            for title_word in title_words
            if len(title_word) > len(query_word)
        ):
            score += 2
        elif any(
            query_word.startswith(title_word) or title_word.startswith(query_word)
            for title_word in title_words
            if len(title_word) > 2
        ):
            score += 1

        if any(char.isdigit() for char in query_word) and query_word in title:
            score += 2

    if detail:
        detail_text, exact_values = _build_detail_search_text(detail)
        detail_words = set(re.findall(r"\b\w+\b", detail_text))
        for query_word in query_words:
            if query_word in exact_values:
                score += 3
            elif f"{query_word}#" in exact_values:
                score += 5
            elif query_word in detail_words:
                score += 2

    return score


def _select_best_product(
    products: Sequence[Product],
    query_text: str,
    *,
    prefer_cheaper: bool = False,
    exclude_ids: Optional[Set[str]] = None,
) -> Optional[Product]:
    """Pick highest-scoring product; re-score top-N with detail fetch; optional price tie-break."""
    if not products:
        return None
    if exclude_ids:
        products = [product for product in products if str(product.get("product_id", "")) not in exclude_ids]
    if not products:
        return None

    scored_products = sorted(
        products,
        key=lambda product: _score_product_relevance(product, query_text),
        reverse=True,
    )
    top_candidates = scored_products[:TOP_RELEVANCE_CANDIDATES]
    details = _fetch_product_details(
        [str(product.get("product_id", "")) for product in top_candidates if product.get("product_id")]
    )

    def final_score(product: Product) -> float:
        score = _score_product_relevance(
            product,
            query_text,
            details.get(str(product.get("product_id", ""))),
        )
        if prefer_cheaper:
            price = product.get("price", 0) or 0
            score -= price / CHEAPER_PRICE_TIEBREAK_DIVISOR
        return score

    return max(top_candidates, key=final_score)


@Tool
def find_product(
    q: str,
    page: int = 1,
    shop_id: Optional[str] = None,
    price: Optional[str] = None,
    sort: Optional[str] = None,
    service: Optional[str] = None,
) -> List[Product]:
    """Search for products matching query."""
    params = _build_search_params(
        q,
        page=page,
        shop_id=shop_id,
        price=price,
        sort=sort,
        service=service,
    )
    result = _search_products(params)
    # Shop-scoped searches can be over-filtered by service; retry once without it.
    if shop_id and not result:
        logger.debug(
            "find_product: empty for shop_id=%s, retrying without service filter",
            shop_id,
        )
        retry_params = {**params}
        retry_params.pop("service", None)
        result = _search_products(retry_params)
    return result


# ---------------------------------------------------------------------------
# Same-shop search
# ---------------------------------------------------------------------------


def _parse_product_queries(product_queries: Any) -> tuple[Optional[List[SearchSpec]], str, Optional[str]]:
    """Parse JSON list of search specs; trailing dict may carry _original_query for scoring."""
    try:
        specs = json.loads(product_queries) if isinstance(product_queries, str) else product_queries
    except json.JSONDecodeError:
        return None, "", "Invalid JSON"

    if not specs or not isinstance(specs, list):
        return None, "", "Need non-empty list"

    original_query = ""
    if isinstance(specs[-1], dict) and specs[-1].get("_original_query"):
        original_query = specs.pop()["_original_query"]

    return specs, original_query, None


def _collect_broad_shop_results(
    specs: Sequence[SearchSpec], *, omit_service_from_api: bool = False
) -> List[List[Product]]:
    """One result list per spec index (parallel broad searches for same-shop discovery)."""
    return [
        _search_products_for_spec(spec, omit_service_from_api=omit_service_from_api) for spec in specs
    ]


def _group_products_by_shop(broad_results: Sequence[Sequence[Product]]) -> Dict[str, Dict[int, List[Product]]]:
    """Invert broad results: shop_id -> spec_index -> products from that search."""
    shop_coverage: Dict[str, Dict[int, List[Product]]] = defaultdict(lambda: defaultdict(list))
    for index, products in enumerate(broad_results):
        for product in products:
            shop_id = str(product.get("shop_id", ""))
            if shop_id:
                shop_coverage[shop_id][index].append(product)
    return shop_coverage


def _score_shop_coverage(
    shop_id: str,
    shop_coverage: Dict[str, Dict[int, List[Product]]],
    specs: Sequence[SearchSpec],
    original_query: str,
) -> tuple[int, float]:
    """Rank shops: sum of best per-spec relevance (prefer service-filtered pool when available)."""
    coverage = shop_coverage[shop_id]
    total_score = 0.0
    for index, products in coverage.items():
        query_text = original_query or specs[index].get("q", "")
        filtered = _filter_products_by_spec_services(products, specs[index])
        # If API omitted service, unfiltered products may include wrong tags—still use for ranking.
        pool = filtered or products
        total_score += max(
            (_score_product_relevance(product, query_text) for product in pool),
            default=0,
        )
    return len(coverage), total_score


def _pick_products_for_shop(
    shop_id: str,
    shop_coverage: Dict[str, Dict[int, List[Product]]],
    specs: Sequence[SearchSpec],
    original_query: str,
    *,
    broad_omit_service: bool,
) -> Optional[List[Product]]:
    """For one shop, pick one product per spec; enforce service tags; avoid duplicate product_ids."""
    selected_products: List[Product] = []
    used_ids: Set[str] = set()
    coverage = shop_coverage.get(shop_id, {})
    for index, spec in enumerate(specs):
        query_text = spec.get("q", "")
        score_query = original_query or query_text
        products = list(coverage.get(index) or [])
        if spec.get("service"):
            products = _filter_products_by_spec_services(products, spec)
        if not products:
            products = _search_products_for_spec(
                spec,
                shop_id=shop_id,
                omit_service_from_api=broad_omit_service,
            )
            products = _filter_products_by_spec_services(products, spec)
        if not products:
            products = _search_products_for_spec(spec, shop_id=shop_id, omit_service_from_api=True)
            products = _filter_products_by_spec_services(products, spec)
        best_product = _select_best_product(
            products,
            query_text or score_query,
            prefer_cheaper=True,
            exclude_ids=used_ids,
        )
        if not best_product:
            return None
        selected_products.append(best_product)
        product_id = str(best_product.get("product_id", ""))
        if product_id:
            used_ids.add(product_id)
    return selected_products


@Tool
def find_products_in_same_shop(product_queries: str) -> Dict[str, Any]:
    """Find multiple products from the SAME shop.

    Tries strict API service filters first, then omits service from search params and filters
    by product-level service tags—so LazMall+COD style constraints can still match one store.
    """
    specs, original_query, error = _parse_product_queries(product_queries)
    if error:
        logger.info("find_products_in_same_shop: parse error: %s", error)
        return {"found": False, "error": error}
    if specs is None:
        return {"found": False, "error": "Need non-empty list"}

    n_specs = len(specs)
    max_shops = (
        MAX_SHOPS_FOR_TWO_OR_FEWER_SPECS if n_specs <= 2 else MAX_SHOPS_WIDE_QUERY
    )
    shops_tried_total = 0

    for broad_omit_service in (False, True):
        broad_results = _collect_broad_shop_results(specs, omit_service_from_api=broad_omit_service)
        if not any(broad_results):
            logger.debug(
                "same_shop phase omit_service_from_api=%s: no broad results",
                broad_omit_service,
            )
            continue

        shop_coverage = _group_products_by_shop(broad_results)
        candidate_shop_ids = sorted(
            shop_coverage,
            key=lambda sid: _score_shop_coverage(sid, shop_coverage, specs, original_query),
            reverse=True,
        )
        logger.debug(
            "same_shop phase omit_service_from_api=%s: %d candidate shops, max_shops=%d",
            broad_omit_service,
            len(candidate_shop_ids),
            max_shops,
        )

        for shops_tried, shop_id in enumerate(candidate_shop_ids[:max_shops], start=1):
            shops_tried_total = shops_tried
            picked = _pick_products_for_shop(
                shop_id,
                shop_coverage,
                specs,
                original_query,
                broad_omit_service=broad_omit_service,
            )
            if picked is not None and len(picked) == n_specs:
                logger.info(
                    "find_products_in_same_shop: found shop_id=%s for %d specs (phase omit_api_service=%s, tried=%d)",
                    shop_id,
                    n_specs,
                    broad_omit_service,
                    shops_tried,
                )
                return {
                    "found": True,
                    "shop_id": shop_id,
                    "products": _serialize_products(picked),
                    "shops_tried": shops_tried,
                }

    logger.info(
        "find_products_in_same_shop: no single shop covers all %d product specs (shops_tried_total=%s)",
        n_specs,
        shops_tried_total,
    )
    return {
        "found": False,
        "error": f"No shop has all {n_specs} products",
        "shops_tried": min(shops_tried_total, max_shops),
    }


# ---------------------------------------------------------------------------
# Public tools (dialogue)
# ---------------------------------------------------------------------------


@Tool
def recommend_product(product_ids: str) -> str:
    """Recommend products to the user."""
    return f"Having recommended the products to the user: {product_ids}."


@Tool
def terminate(status: str = "success") -> str:
    """End the dialogue."""
    return f"The interaction has been completed with status: {status}"


# ---------------------------------------------------------------------------
# LLM extraction & rule-based fallback
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """Extract search params as JSON. No markdown.
{"task_type":"product"|"shop"|"voucher","products":[{"keywords":"search query","price_range":"min-max"|null,"service":"official"|"freeShipping"|"COD"|"flashsale"|null}],"is_shop_voucher":bool}
- keywords: 2-5 words: brand + specific model/type. Use exact model names and numbers verbatim. Include color/size only if essential to product identity. Omit filler words (quality, nice, good). NEVER start with "shops", "find stores", "look for selling".
- price_range: "100-500", "100-", "0-500". null if none. For voucher/budget tasks: ALWAYS compute price_range as "0-{ceil}" where ceil = round(budget / (1 - pct/100)) for percentage discounts (e.g. 1000 budget + 10% off → "0-1111"), or budget + fixed_amount for fixed discounts (e.g. 1000 budget + 100 off → "0-1100"). Divide ceil by number of products for multi-item tasks.
- service: LazMall=official, free shipping=freeShipping, COD=COD, flash sale=flashsale. null if none.
- task_type: product=single item only; shop=same store must sell MULTIPLE different items (numbered First/Second/Also, or "both", "these items"); voucher=budget/discount block present.
- shop tasks: products MUST have one object per DISTINCT item the user listed. If there are 4 items (board game; colander; toy; heater), products array length MUST be 4. NEVER merge several products into one keywords string.
- Multi-product: preserve user order. Budget/voucher sentences are NOT products.
- is_shop_voucher: true if "same shop" voucher.
JSON only:"""


_VOUCHER_HINTS = ("voucher", "budget", "discount")
_SHOP_HINTS = ("both", "these", "offering", "sells", "same", "items:")


def _detect_task_type(query: str) -> str:
    normalized = query.lower()
    if any(k in normalized for k in _VOUCHER_HINTS):
        return "voucher"
    if "shop" in normalized and any(k in normalized for k in _SHOP_HINTS):
        return "shop"
    return "product"


def _split_enumerated_shop_segments(product_text: str) -> Optional[List[str]]:
    segments = re.split(
        r"(?i)(?:^|\n)\s*(?:First|Second|Third|Fourth|Fifth|Additionally|Also|Next|Lastly)[,:]?\s+",
        product_text,
    )
    parts = [segment.strip() for segment in segments if segment and len(segment.strip()) > 18]
    if len(parts) >= 2:
        return parts
    if product_text.count(";") >= 2:
        semi = [segment.strip() for segment in product_text.split(";") if len(segment.strip()) > 18]
        if len(semi) >= 2:
            return semi
    return None


def _default_is_shop_voucher(task_type: str, query_lower: str) -> bool:
    return task_type == "shop" or (task_type == "voucher" and "same shop" in query_lower)


def _repair_shop_params_if_undersplit(params: Dict[str, Any], query: str) -> Dict[str, Any]:
    """If the model returned one merged product for a multi-item shop query, split on First/Second/; ."""
    if (params.get("task_type") or "").lower() != "shop":
        return params
    products = params.get("products") or []
    head = re.split(r"(?:My budget|budget is|I have a voucher)", query, flags=re.I)[0].strip() or query
    segments = _split_enumerated_shop_segments(head)
    if not segments or len(segments) <= len(products):
        return params
    repaired = [_parse_product_spec(segment) for segment in segments]
    if len(repaired) > len(products):
        logger.info(
            "Repaired shop extraction: %d product specs -> %d (enumerated segments)",
            len(products),
            len(repaired),
        )
        return {**params, "products": repaired}
    return params


def _extract_json_payload(content: str) -> Optional[Dict[str, Any]]:
    cleaned = re.sub(r"```json?\s*", "", content)
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = _JSON_BLOCK_RE.search(content)
        if not match:
            return None
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None


def _extract_search_params_with_llm(query: str) -> Dict[str, Any]:
    """Ask the LLM for structured JSON; on failure or invalid JSON, use rule-based fallback."""
    model = getenv("SANDBOX_MODEL", "deepseek-ai/DeepSeek-V3.2-TEE")
    result = _inference_client.post(
        "/inference/chat/completions",
        json_data={
            "model": model,
            "temperature": 0,
            "stream": False,
            "messages": [
                {"role": "system", "content": _EXTRACTION_PROMPT},
                {"role": "user", "content": query},
            ],
        },
    )
    if result and result.get("choices"):
        content = result["choices"][0].get("message", {}).get("content", "")
        parsed = _extract_json_payload(content)
        if parsed is not None:
            logger.debug("Extraction: using LLM JSON (model=%s)", model)
            return parsed
        logger.info("Extraction: LLM returned no parseable JSON; using fallback (model=%s)", model)
    else:
        logger.info("Extraction: no LLM choices; using fallback (model=%s)", model)
    return _extract_search_params_fallback(query)


_FALLBACK_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "can",
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
    "size", "sized", "eu", "fits",
}


def _parse_product_spec(text: str) -> Dict[str, Optional[str]]:
    lower_text = text.lower()
    alpha_words = [
        word
        for word in re.findall(r"\b[a-zA-Z]{2,}\b", lower_text)
        if word not in _FALLBACK_STOPWORDS
    ]
    alnum_tokens = re.findall(r"\b\d+[a-zA-Z]+\b|\b[a-zA-Z]+\d+[a-zA-Z]*\b", lower_text)

    keywords = alpha_words[:6]
    for token in alnum_tokens[:2]:
        if token not in keywords:
            keywords.append(token)

    for shade_number in re.findall(r"(\d+)#", text)[:2]:
        if shade_number not in keywords:
            keywords.append(shade_number)

    price_range = None
    min_only_match = re.search(
        r"(?:greater|more|over|above|>|cost[s]?\s+more)\s*(?:than\s*)?(\d+)",
        text,
        re.I,
    )
    if min_only_match:
        price_range = f"{min_only_match.group(1)}-"
    else:
        explicit_range_match = re.search(
            r"(\d{1,6})\s*(?:to|and|-)\s*(\d{1,6})\s*(?:pesos|php)",
            text,
            re.I,
        )
        if explicit_range_match:
            price_range = f"{explicit_range_match.group(1)}-{explicit_range_match.group(2)}"
        elif re.search(r"(?:price|pesos|php|cost)", text, re.I):
            loose_range_match = re.search(r"(\d{1,6})\s+(?:to|and)\s+(\d{1,6})", text)
            if loose_range_match:
                price_range = f"{loose_range_match.group(1)}-{loose_range_match.group(2)}"

    service_parts: List[str] = []
    if "lazmall" in lower_text or "official" in lower_text:
        service_parts.append("official")
    if "free shipping" in lower_text or "free delivery" in lower_text:
        service_parts.append("freeShipping")
    if "lazflash" in lower_text or "flash sale" in lower_text or "flashsale" in lower_text:
        service_parts.append("flashsale")
    if "cash on delivery" in lower_text or "cod" in lower_text:
        service_parts.append("COD")

    return {
        "keywords": " ".join(keywords) or DEFAULT_PRODUCT_QUERY,
        "price_range": price_range,
        "service": ",".join(service_parts) or None,
    }


def _extract_search_params_fallback(query: str) -> Dict[str, Any]:
    task_type = _detect_task_type(query)
    product_text = re.split(r"(?:My budget|budget is|I have a voucher)", query, flags=re.I)[0].strip()
    if not product_text or len(product_text) < 15:
        product_text = query

    if task_type == "shop":
        enum_parts = _split_enumerated_shop_segments(product_text)
        if enum_parts:
            products = [_parse_product_spec(part) for part in enum_parts]
            products = [product for product in products if len((product["keywords"] or "").split()) >= 2] or products
            qlow = query.lower()
            return {
                "task_type": task_type,
                "products": products,
                "is_shop_voucher": _default_is_shop_voucher(task_type, qlow),
            }

    parts = [part.strip() for part in _PRODUCT_TEXT_SPLIT_RE.split(product_text) if part and len(part.strip()) > 10]
    if not parts:
        parts = [query]

    products = [_parse_product_spec(part) for part in parts]
    products = [product for product in products if len((product["keywords"] or "").split()) >= 2] or products

    qlow = query.lower()
    return {
        "task_type": task_type,
        "products": products,
        "is_shop_voucher": _default_is_shop_voucher(task_type, qlow),
    }


# ---------------------------------------------------------------------------
# Agent orchestration (steps, handlers, entrypoint)
# ---------------------------------------------------------------------------


def _add_dialogue_step(
    think: str,
    tool_results: Sequence[Dict[str, Any]],
    response: str,
    query: str,
    steps: List[Dict[str, Any]],
) -> None:
    steps.append(create_dialogue_step(think, list(tool_results), response, query, len(steps) + 1))


def _execute_and_record(
    tool_name: str,
    payload: Dict[str, Any],
    query: str,
    steps: List[Dict[str, Any]],
    *,
    think: str = "Processing.",
    response: str = "",
) -> Dict[str, Any]:
    result = execute_tool_call(tool_name, payload)
    _add_dialogue_step(think, [result], response, query, steps)
    return result


def _build_tool_search_payload(product: Dict[str, Any], *, include_price: bool = True) -> Dict[str, Any]:
    payload = {"q": product.get("keywords", DEFAULT_PRODUCT_QUERY)}
    if include_price and product.get("price_range"):
        payload["price"] = product["price_range"]
    if product.get("service"):
        payload["service"] = product["service"]
    return payload


def _finalize_recommendation(product_ids: Iterable[Any], query: str, steps: List[Dict[str, Any]]) -> None:
    formatted = _format_product_ids(product_ids)
    logger.info("Recommending product_ids=%s", formatted or "(empty)")
    recommendation = execute_tool_call(
        "recommend_product",
        {"product_ids": formatted},
    )
    termination = execute_tool_call("terminate", {"status": "success"})
    _add_dialogue_step("Done.", [recommendation, termination], "Done.", query, steps)


def _handle_single_product(params: Dict[str, Any], query: str, steps: List[Dict[str, Any]]) -> None:
    """Single product task: one find_product, then best match by relevance."""
    products = params.get("products", [{}])
    product = products[0] if products else {}
    payload = _build_tool_search_payload(product)
    result = _execute_and_record(
        "find_product",
        payload,
        query,
        steps,
    )
    unique_products = _deduplicate_products(result.get("result") or [])
    # Fetch page 2 to widen candidate pool for relevance selection.
    if unique_products:
        page2_result = _execute_and_record(
            "find_product",
            {**payload, "page": 2},
            query,
            steps,
        )
        unique_products = _deduplicate_products(
            unique_products + (page2_result.get("result") or [])
        )
    if not unique_products:
        # Retry without price/service filters to widen the candidate pool.
        fallback_payload = {"q": payload.get("q", DEFAULT_PRODUCT_QUERY)}
        fallback_result = _execute_and_record(
            "find_product",
            fallback_payload,
            query,
            steps,
        )
        unique_products = _deduplicate_products(fallback_result.get("result") or [])
    if not unique_products:
        logger.info("Single product search returned no results")
    best_product = _select_best_product(unique_products, query) if unique_products else None
    if best_product:
        logger.debug(
            "Selected product_id=%s (from %d hits)",
            best_product.get("product_id"),
            len(unique_products),
        )
    _finalize_recommendation([best_product["product_id"]] if best_product else [], query, steps)


def _search_products_individually(
    params: Dict[str, Any],
    query: str,
    steps: List[Dict[str, Any]],
    *,
    include_price: bool,
    prefer_cheaper: bool,
    use_keyword_query_for_multi: bool = False,
    extra_pages: int = 0,
) -> List[str]:
    product_ids: List[str] = []
    products = params.get("products", [])
    for product in products:
        try:
            payload = _build_tool_search_payload(product, include_price=include_price)
            result = _execute_and_record(
                "find_product",
                payload,
                query,
                steps,
            )
            found_products = result.get("result") or []

            # Fetch additional pages to widen the candidate pool when requested.
            for page_num in range(2, 2 + extra_pages):
                extra_payload = {**payload, "page": page_num}
                extra_result = _execute_and_record(
                    "find_product",
                    extra_payload,
                    query,
                    steps,
                )
                found_products = _deduplicate_products(
                    found_products + (extra_result.get("result") or [])
                )

            if not found_products:
                continue

            score_query = (
                product.get("keywords", DEFAULT_PRODUCT_QUERY)
                if use_keyword_query_for_multi
                else query
            )
            best_product = _select_best_product(
                found_products,
                score_query,
                prefer_cheaper=prefer_cheaper,
            )
            if best_product:
                product_ids.append(str(best_product["product_id"]))
        except Exception:
            logger.exception("Fallback individual product search failed.")
    return product_ids


def _handle_same_shop_search(
    params: Dict[str, Any],
    query: str,
    steps: List[Dict[str, Any]],
    *,
    is_voucher: bool = False,
) -> None:
    """Try find_products_in_same_shop first; if not found, search each line item separately."""
    product_queries = [
        _build_tool_search_payload(product, include_price=not is_voucher)
        for product in params.get("products", [])
    ] or [{"q": DEFAULT_PRODUCT_QUERY}]
    product_queries.append({"_original_query": query})

    result = _execute_and_record(
        "find_products_in_same_shop",
        {"product_queries": json.dumps(product_queries)},
        query,
        steps,
    )

    same_shop_result = result.get("result")
    if isinstance(same_shop_result, dict) and same_shop_result.get("found"):
        product_ids = [str(product["product_id"]) for product in same_shop_result.get("products", [])]
    else:
        logger.info(
            "Same-shop search failed; falling back to per-product find_product (is_voucher=%s)",
            is_voucher,
        )
        product_ids = _search_products_individually(
            params,
            query,
            steps,
            include_price=not is_voucher,
            prefer_cheaper=True,
        )

    _finalize_recommendation(product_ids, query, steps)


def _derive_voucher_price_ceiling(query: str, n_products: int) -> Optional[str]:
    """Estimate per-product price ceiling from total budget and discount info in query."""
    budget = None
    for pattern in [
        r'budget\s+(?:is\s+|of\s+)?(?:php\s*|₱\s*)?(\d[\d,]+)',
        r'(?:php\s*|₱\s*)(\d[\d,]+)\s+(?:budget|total)',
        r'within\s+(?:a\s+)?(?:total\s+)?(?:budget\s+of\s+)?(?:php\s*|₱\s*)?(\d[\d,]+)',
    ]:
        m = re.search(pattern, query, re.I)
        if m:
            budget = float(m.group(1).replace(',', ''))
            break
    if budget is None:
        return None

    # Adjust pre-discount budget based on discount type so price filter isn't too tight.
    gross_budget = budget
    pct_m = re.search(r'(\d+)\s*%\s*(?:off|discount|voucher)', query, re.I)
    if pct_m:
        pct = float(pct_m.group(1))
        if 0 < pct < 100:
            gross_budget = budget / (1.0 - pct / 100.0)
    else:
        fixed_m = re.search(r'(?:php|₱)\s*(\d[\d,]+)\s*(?:off|discount)', query, re.I)
        if fixed_m:
            gross_budget = budget + float(fixed_m.group(1).replace(',', ''))

    max_per_product = gross_budget / max(n_products, 1)
    return f"0-{int(max_per_product)}"


def _enrich_voucher_product_params(params: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Add per-product price ceilings derived from budget for voucher tasks when not already set."""
    products = list(params.get("products") or [])
    if not products:
        return params
    if all(p.get("price_range") for p in products):
        return params  # LLM already extracted per-product price ranges

    ceiling = _derive_voucher_price_ceiling(query, len(products))
    if ceiling is None:
        return params

    logger.info("Voucher: applying derived price ceiling %s for %d products", ceiling, len(products))
    enriched = []
    for p in products:
        if p.get("price_range"):
            enriched.append(p)
        else:
            enriched.append({**p, "price_range": ceiling})
    return {**params, "products": enriched}


def _handle_voucher_search(params: Dict[str, Any], query: str, steps: List[Dict[str, Any]]) -> None:
    """Voucher path: optional same-shop bundle, else independent searches with budget awareness."""
    is_same_shop_voucher = params.get("is_shop_voucher", False) or "same shop" in query.lower()
    products = params.get("products", [])

    if is_same_shop_voucher and len(products) > 1:
        logger.info("Voucher: same-shop mode, %d products", len(products))
        _handle_same_shop_search(params, query, steps, is_voucher=True)
        return

    # Apply budget-derived price ceilings so searches stay within affordable range.
    enriched_params = _enrich_voucher_product_params(params, query)
    # Use include_price=True so the derived ceilings are applied; fallback to no-price if empty.
    product_ids = _search_products_individually(
        enriched_params,
        query,
        steps,
        include_price=True,
        prefer_cheaper=True,
        use_keyword_query_for_multi=True,
        extra_pages=1,
    )
    if not product_ids:
        # Retry without price constraints if budget-filtered search found nothing.
        product_ids = _search_products_individually(
            params,
            query,
            steps,
            include_price=False,
            prefer_cheaper=True,
            use_keyword_query_for_multi=True,
        )
    _finalize_recommendation(product_ids, query, steps)


def agent_main(problem_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Entry: extract params, route by task type, return dialogue steps for the evaluator."""
    _product_detail_cache.clear()
    steps: List[Dict[str, Any]] = []
    query = problem_data.get("query", "")
    logger.info("agent_main start query_preview=%r", _preview_query(query))

    try:
        params = _extract_search_params_with_llm(query)
        params = _repair_shop_params_if_undersplit(params, query)
        task_type = (params.get("task_type") or "").lower() or _detect_task_type(query)
        keyword_task_type = _detect_task_type(query)
        if keyword_task_type != "product" and task_type == "product":
            task_type = keyword_task_type

        n_products = len(params.get("products") or [])
        logger.info(
            "Resolved task_type=%s products=%d (keyword_hint=%s)",
            task_type,
            n_products,
            keyword_task_type,
        )

        _add_dialogue_step("Processing.", [], "", query, steps)

        if task_type == "shop":
            _handle_same_shop_search(params, query, steps)
        elif task_type == "voucher":
            _handle_voucher_search(params, query, steps)
        else:
            _handle_single_product(params, query, steps)
    except Exception:
        logger.exception("Agent execution failed.")
        try:
            recommendation = execute_tool_call("recommend_product", {"product_ids": ""})
            termination = execute_tool_call("terminate", {"status": "failure"})
            _add_dialogue_step("Processing.", [recommendation, termination], "Done.", query, steps)
        except Exception:
            steps.append(create_dialogue_step("Done.", [], "Done.", query, len(steps) + 1))

    if not steps:
        steps.append(create_dialogue_step("Done.", [], "Done.", query, 1))

    logger.info("agent_main done steps=%d", len(steps))
    return steps
