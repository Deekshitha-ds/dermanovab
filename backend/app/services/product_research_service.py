"""
DermaNova AI - Live Product Price Research

Searches Google Shopping through SerpApi and compares
supported Indian marketplaces internally.

Supported stores:
- Amazon
- Nykaa
- Purplle

The frontend receives ONE selected product offer.
Multiple stores are compared internally.
"""

import os
import re
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_KEY")

SERPAPI_URL = "https://serpapi.com/search.json"

SUPPORTED_STORES = {
    "amazon": "Amazon",
    "amazon.in": "Amazon",
    "amazon india": "Amazon",

    "nykaa": "Nykaa",
    "nykaa.com": "Nykaa",

    "purplle": "Purplle",
    "purplle.com": "Purplle",
}


# ============================================================
# TEXT HELPERS
# ============================================================

def _normalise(text: Optional[str]) -> str:
    """
    Converts text into a simple lowercase searchable format.
    """

    if not text:
        return ""

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\s%]+",
        " ",
        text,
    )

    return " ".join(text.split())


def _tokens(text: str) -> set[str]:
    """
    Converts text into useful searchable tokens.
    """

    stop_words = {
        "the",
        "for",
        "with",
        "and",
        "of",
        "to",
        "a",
        "an",
        "this",
        "that",
        "face",
        "skin",
        "product",
        "ml",
        "gm",
        "g",
        "oz",
        "online",
        "best",
        "beauty",
        "products",
        "india",
    }

    normalized = _normalise(text)

    return {
        token
        for token in normalized.split()
        if len(token) > 1
        and token not in stop_words
    }


def _store_name(source: Optional[str]) -> Optional[str]:
    """
    Converts a store/source name into our supported store name.
    """

    if not source:
        return None

    normalized = _normalise(source)

    if "amazon" in normalized:
        return "Amazon"

    if "nykaa" in normalized:
        return "Nykaa"

    if "purplle" in normalized:
        return "Purplle"

    return None


def _store_from_url(url: Optional[str]) -> Optional[str]:
    """
    Detects supported store from a merchant URL.
    """

    if not url:
        return None

    try:
        hostname = urlparse(url).netloc.lower()

        hostname = hostname.replace(
            "www.",
            "",
        )

        if "amazon.in" in hostname:
            return "Amazon"

        if "nykaa.com" in hostname:
            return "Nykaa"

        if "purplle.com" in hostname:
            return "Purplle"

    except Exception:
        pass

    return None


# ============================================================
# PRODUCT MATCHING
# ============================================================

def _brand_matches(
    brand: str,
    result_title: str,
) -> bool:
    """
    Checks whether the recommended brand appears in the
    shopping result title.
    """

    brand_normalized = _normalise(brand)
    title_normalized = _normalise(result_title)

    if not brand_normalized:
        return False

    return brand_normalized in title_normalized


def _product_match_score(
    product_name: str,
    brand: str,
    result_title: str,
) -> float:
    """
    Calculates product similarity.

    Brand is considered separately.
    Product tokens are compared against the shopping title.
    """

    product_tokens = _tokens(product_name)
    result_tokens = _tokens(result_title)

    if not product_tokens or not result_tokens:
        return 0.0

    matched = product_tokens & result_tokens

    return len(matched) / len(product_tokens)


def _has_bad_variant(
    product_name: str,
    result_title: str,
) -> bool:
    """
    Rejects obvious unwanted variants such as:

    - Duo
    - Pack
    - Combo
    - Set
    - Kit

    This helps prevent a single-product recommendation from
    being replaced with a bundle.
    """

    requested = _normalise(product_name)
    title = _normalise(result_title)

    variant_words = {
        "duo",
        "combo",
        "pack",
        "set",
        "kit",
        "bundle",
        "gift",
    }

    for word in variant_words:
        if word in title and word not in requested:
            return True

    return False


# ============================================================
# PRICE HELPER
# ============================================================

def _extract_price(
    value: Any,
) -> Optional[float]:
    """
    Converts a SerpApi price value into a positive float.
    """

    if value is None:
        return None

    try:

        if isinstance(value, str):

            value = re.sub(
                r"[^0-9.]+",
                "",
                value,
            )

        price = float(value)

        if price <= 0:
            return None

        return price

    except (
        TypeError,
        ValueError,
    ):
        return None


# ============================================================
# SERPAPI GOOGLE SHOPPING
# ============================================================

def _search_google_shopping(
    query: str,
) -> List[Dict[str, Any]]:
    """
    Searches Google Shopping through SerpApi.
    """

    if not SERPAPI_KEY:
        raise RuntimeError(
            "SERPAPI_KEY is not configured in the .env file."
        )

    params = {
        "engine": "google_shopping",
        "q": query,
        "api_key": SERPAPI_KEY,

        "google_domain": "google.co.in",
        "gl": "in",
        "hl": "en",

        "num": "40",
    }

    response = requests.get(
        SERPAPI_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    if data.get("error"):
        raise RuntimeError(
            f"SerpApi error: {data.get('error')}"
        )

    return data.get(
        "shopping_results",
        [],
    )


# ============================================================
# IMMERSIVE PRODUCT API
# ============================================================

def _get_immersive_product(
    page_token: str,
) -> Dict[str, Any]:
    """
    Gets detailed merchant offers from SerpApi's
    Google Immersive Product API.

    This is optional. If it fails, the main Google Shopping
    result is still used as a fallback.
    """

    if not SERPAPI_KEY:
        raise RuntimeError(
            "SERPAPI_KEY is not configured in the .env file."
        )

    params = {
        "engine": "google_immersive_product",
        "page_token": page_token,
        "api_key": SERPAPI_KEY,
        "more_stores": "true",
    }

    response = requests.get(
        SERPAPI_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    if data.get("error"):
        raise RuntimeError(
            f"SerpApi immersive product error: "
            f"{data.get('error')}"
        )

    return data


# ============================================================
# IMMERSIVE STORE OFFERS
# ============================================================

def _extract_store_offers(
    product_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Extracts Amazon, Nykaa and Purplle offers from the
    Immersive Product response.
    """

    product_results = product_data.get(
        "product_results",
        {},
    )

    stores = product_results.get(
        "stores",
        [],
    )

    offers: List[Dict[str, Any]] = []

    if not isinstance(stores, list):
        return offers

    for store_result in stores:

        if not isinstance(
            store_result,
            dict,
        ):
            continue

        link = store_result.get("link")

        if not isinstance(
            link,
            str,
        ):
            continue

        link = link.strip()

        if not link:
            continue

        store = None

        possible_store_fields = [
            "source",
            "store",
            "merchant",
            "seller",
            "name",
            "title",
        ]

        for field in possible_store_fields:

            value = store_result.get(field)

            store = _store_name(value)

            if store:
                break

        if not store:
            store = _store_from_url(link)

        if not store:
            continue

        price = store_result.get(
            "extracted_price"
        )

        if price is None:
            price = store_result.get(
                "price"
            )

        price = _extract_price(price)

        if price is None:
            continue

        offers.append({
            "store": store,
            "price": price,
            "purchase_link": link,
        })

    return offers


# ============================================================
# GOOGLE SHOPPING DIRECT OFFER
# ============================================================

def _extract_google_shopping_offer(
    product: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Uses the original Google Shopping result as a direct
    merchant fallback.

    This is the most important fallback when Immersive Product
    is unavailable.
    """

    link = product.get(
        "shopping_link"
    )

    source = product.get(
        "shopping_source"
    )

    if not link:
        return None

    if not isinstance(
        link,
        str,
    ):
        return None

    link = link.strip()

    if not link:
        return None

    store = _store_name(source)

    if not store:
        store = _store_from_url(link)

    if not store:
        return None

    price = _extract_price(
        product.get("price")
    )

    if price is None:
        return None

    return {
        "store": store,
        "price": price,
        "purchase_link": link,
    }


# ============================================================
# BUILD SEARCH QUERIES
# ============================================================

def _build_search_queries(
    product_name: str,
    brand: str,
    category: Optional[str],
) -> List[str]:
    """
    Creates multiple search queries.

    We do NOT depend on one Google Shopping query.
    """

    queries: List[str] = []

    clean_product = product_name.strip()
    clean_brand = brand.strip()

    # --------------------------------------------------------
    # Query 1: Brand + exact product
    # --------------------------------------------------------

    queries.append(
        f"{clean_brand} {clean_product}"
    )

    # --------------------------------------------------------
    # Query 2: Exact product phrase
    # --------------------------------------------------------

    queries.append(
        f'"{clean_brand} {clean_product}"'
    )

    # --------------------------------------------------------
    # Query 3: Brand + product + category
    # --------------------------------------------------------

    if category:
        queries.append(
            f"{clean_brand} {clean_product} {category}"
        )

    # --------------------------------------------------------
    # Store-specific queries
    # --------------------------------------------------------

    queries.append(
        f"{clean_brand} {clean_product} Amazon India"
    )

    queries.append(
        f"{clean_brand} {clean_product} Nykaa"
    )

    queries.append(
        f"{clean_brand} {clean_product} Purplle"
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    unique_queries = []

    seen = set()

    for query in queries:

        normalized = _normalise(query)

        if normalized in seen:
            continue

        seen.add(normalized)

        unique_queries.append(query)

    return unique_queries


# ============================================================
# BUILD CANDIDATES
# ============================================================

def _build_candidates(
    results: List[Dict[str, Any]],
    product_name: str,
    brand: str,
) -> List[Dict[str, Any]]:
    """
    Converts Google Shopping results into validated product
    candidates.
    """

    candidates: List[Dict[str, Any]] = []

    for result in results:

        if not isinstance(
            result,
            dict,
        ):
            continue

        title = result.get(
            "title",
            "",
        )

        if not title:
            continue

        # ----------------------------------------------------
        # Brand must match
        # ----------------------------------------------------

        if not _brand_matches(
            brand=brand,
            result_title=title,
        ):
            continue

        # ----------------------------------------------------
        # Reject obvious bundles/variants
        # ----------------------------------------------------

        if _has_bad_variant(
            product_name=product_name,
            result_title=title,
        ):
            continue

        # ----------------------------------------------------
        # Product similarity
        # ----------------------------------------------------

        match_score = _product_match_score(
            product_name=product_name,
            brand=brand,
            result_title=title,
        )

        # Keep a reasonably strict threshold.
        if match_score < 0.50:
            continue

        # ----------------------------------------------------
        # Price
        # ----------------------------------------------------

        price = _extract_price(
            result.get(
                "extracted_price"
            )
        )

        if price is None:
            continue

        candidates.append({
            "title": title,

            "price": price,

            "thumbnail": result.get(
                "thumbnail"
            ),

            "rating": result.get(
                "rating"
            ),

            "match_score": round(
                match_score,
                3,
            ),

            "immersive_token": result.get(
                "immersive_product_page_token"
            ),

            "shopping_link": result.get(
                "link"
            ),

            "shopping_source": result.get(
                "source"
            ),
        })

    return candidates


# ============================================================
# DEDUPLICATE CANDIDATES
# ============================================================

def _deduplicate_candidates(
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Removes duplicate shopping results.
    """

    unique = []

    seen = set()

    for candidate in candidates:

        key = (
            _normalise(
                candidate.get(
                    "title",
                    ""
                )
            ),
            candidate.get(
                "shopping_link"
            ),
        )

        if key in seen:
            continue

        seen.add(key)

        unique.append(candidate)

    return unique


# ============================================================
# LIVE PRODUCT RESEARCH
# ============================================================

def research_best_offer(
    product_name: str,
    brand: str,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Searches multiple Google Shopping queries, compares
    Amazon/Nykaa/Purplle results internally, and returns ONE
    selected offer.

    Immersive Product is optional.

    The returned object contains:

    found
    price
    store
    purchase_link
    title
    thumbnail
    rating
    match_score
    source
    """

    if not product_name or not brand:
        return {
            "found": False,
            "price": None,
            "store": None,
            "purchase_link": None,
            "title": None,
            "thumbnail": None,
            "rating": None,
            "match_score": None,
            "source": "live_research",
        }

    # ========================================================
    # BUILD MULTIPLE SEARCH QUERIES
    # ========================================================

    queries = _build_search_queries(
        product_name=product_name,
        brand=brand,
        category=category,
    )

    all_candidates: List[Dict[str, Any]] = []

    # ========================================================
    # RUN MULTIPLE GOOGLE SHOPPING SEARCHES
    # ========================================================

    for query in queries:

        try:

            print(
                f"Product research query: {query}"
            )

            results = _search_google_shopping(
                query
            )

            candidates = _build_candidates(
                results=results,
                product_name=product_name,
                brand=brand,
            )

            all_candidates.extend(
                candidates
            )

            # ------------------------------------------------
            # If this query produced a supported merchant
            # result, we still continue searching because
            # another store may have a cheaper offer.
            # ------------------------------------------------

        except Exception as exc:

            print(
                "Google Shopping query failed:",
                exc,
            )

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    all_candidates = _deduplicate_candidates(
        all_candidates
    )

    # ========================================================
    # NO MATCH
    # ========================================================

    if not all_candidates:

        print(
            "No matching shopping candidates found for:",
            brand,
            product_name,
        )

        return {
            "found": False,
            "price": None,
            "store": None,
            "purchase_link": None,
            "title": None,
            "thumbnail": None,
            "rating": None,
            "match_score": None,
            "source": "live_research",
        }

    # ========================================================
    # SORT PRODUCTS BY RELEVANCE
    # ========================================================

    all_candidates.sort(
        key=lambda item: (
            -item["match_score"],
            item["price"],
        )
    )

    # ========================================================
    # COLLECT SUPPORTED OFFERS
    # ========================================================

    supported_offers: List[Dict[str, Any]] = []

    # --------------------------------------------------------
    # Try each matching candidate
    # --------------------------------------------------------

    for candidate in all_candidates:

        # ====================================================
        # IMMERSIVE PRODUCT
        # ====================================================

        immersive_token = candidate.get(
            "immersive_token"
        )

        if immersive_token:

            try:

                immersive_data = _get_immersive_product(
                    immersive_token
                )

                offers = _extract_store_offers(
                    immersive_data
                )

                for offer in offers:

                    supported_offers.append({
                        "store": offer["store"],
                        "price": offer["price"],
                        "purchase_link": offer[
                            "purchase_link"
                        ],
                        "title": candidate[
                            "title"
                        ],
                        "thumbnail": candidate[
                            "thumbnail"
                        ],
                        "rating": candidate[
                            "rating"
                        ],
                        "match_score": candidate[
                            "match_score"
                        ],
                    })

            except Exception as exc:

                # Immersive failure is NOT fatal.
                print(
                    "Immersive product research failed:",
                    exc,
                )

        # ====================================================
        # ORIGINAL GOOGLE SHOPPING RESULT
        # ====================================================

        shopping_offer = _extract_google_shopping_offer(
            candidate
        )

        if shopping_offer:

            supported_offers.append({
                "store": shopping_offer[
                    "store"
                ],
                "price": shopping_offer[
                    "price"
                ],
                "purchase_link": shopping_offer[
                    "purchase_link"
                ],
                "title": candidate[
                    "title"
                ],
                "thumbnail": candidate[
                    "thumbnail"
                ],
                "rating": candidate[
                    "rating"
                ],
                "match_score": candidate[
                    "match_score"
                ],
            })

    # ========================================================
    # REMOVE DUPLICATE OFFERS
    # ========================================================

    unique_offers = []

    seen = set()

    for offer in supported_offers:

        key = (
            offer.get("store"),
            offer.get("price"),
            offer.get("purchase_link"),
        )

        if key in seen:
            continue

        seen.add(key)

        unique_offers.append(
            offer
        )

    supported_offers = unique_offers

    # ========================================================
    # SELECT ONE OFFER
    # ========================================================

    if supported_offers:

        # ----------------------------------------------------
        # Prefer the strongest product match first.
        #
        # If products have essentially the same relevance,
        # choose the cheaper merchant offer.
        # ----------------------------------------------------

        supported_offers.sort(
            key=lambda item: (
                -item["match_score"],
                item["price"],
            )
        )

        best_offer = supported_offers[0]

        print(
            "Selected live product:",
            best_offer["title"],
            "|",
            best_offer["store"],
            "|",
            best_offer["price"],
        )

        return {
            "found": True,
            "price": best_offer[
                "price"
            ],
            "store": best_offer[
                "store"
            ],
            "purchase_link": best_offer[
                "purchase_link"
            ],
            "title": best_offer[
                "title"
            ],
            "thumbnail": best_offer[
                "thumbnail"
            ],
            "rating": best_offer[
                "rating"
            ],
            "match_score": best_offer[
                "match_score"
            ],
            "source": "live_research",
        }

    # ========================================================
    # MATCHING PRODUCT FOUND BUT NO SUPPORTED STORE
    # ========================================================

    best_product = all_candidates[0]

    return {
        "found": True,
        "price": best_product[
            "price"
        ],
        "store": None,
        "purchase_link": None,
        "title": best_product[
            "title"
        ],
        "thumbnail": best_product[
            "thumbnail"
        ],
        "rating": best_product[
            "rating"
        ],
        "match_score": best_product[
            "match_score"
        ],
        "source": "live_research",
    }