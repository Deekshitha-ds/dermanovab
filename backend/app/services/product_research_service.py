"""
DermaNova AI - Live Product Price Research

Researches current Google Shopping results for a recommended
skincare product, compares supported marketplaces, and returns
one cheapest matching available result.

Supported stores:
- Amazon
- Nykaa
- Purplle

This service is separate from the product recommendation/scoring engine.
"""

import os
import re
from typing import Optional, Dict, Any, List

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

    # Keep letters, numbers, spaces and %
    text = re.sub(
        r"[^a-z0-9\s%]+",
        " ",
        text
    )

    return " ".join(text.split())


def _tokens(text: str) -> set[str]:
    """
    Converts text into meaningful searchable tokens.
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
    Converts SerpApi source names into our supported store names.
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

    This prevents DermaNova from selecting a cheaper
    product from a completely different brand.
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
    Calculates how closely a shopping result matches
    the recommended product.

    Brand is handled separately.
    """

    product_tokens = _tokens(product_name)
    result_tokens = _tokens(result_title)

    if not product_tokens or not result_tokens:
        return 0.0

    matched = product_tokens & result_tokens

    return len(matched) / len(product_tokens)


# ============================================================
# URL EXTRACTION
# ============================================================

def _extract_purchase_link(
    result: Dict[str, Any]
) -> Optional[str]:
    """
    Returns a direct merchant link when SerpApi provides one.

    Google Shopping can return a product_link that points
    back to Google's Shopping page instead of the merchant.
    That URL is NOT exposed as the purchase link.
    """

    possible_fields = [
        "link",
        "merchant_link",
        "source_link",
    ]

    for field in possible_fields:

        value = result.get(field)

        if not isinstance(value, str):
            continue

        value = value.strip()

        if not value:
            continue

        # Ignore Google Shopping URLs.
        if "google." in value and "/search" in value:
            continue

        return value

    return None


# ============================================================
# SERPAPI SEARCH
# ============================================================

def _search_google_shopping(
    query: str
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

        # India
        "google_domain": "google.co.in",
        "gl": "in",
        "hl": "en",

        # Return shopping results
        "num": "40",
    }

    response = requests.get(
        SERPAPI_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    # SerpApi may return an API error even with HTTP 200.
    if data.get("error"):
        raise RuntimeError(
            f"SerpApi error: {data.get('error')}"
        )

    return data.get(
        "shopping_results",
        []
    )


# ============================================================
# LIVE PRODUCT RESEARCH
# ============================================================

def research_best_offer(
    product_name: str,
    brand: str,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Searches Google Shopping, filters for the requested
    brand/product, compares supported stores, and returns
    the cheapest matching result.

    Returns one result only.
    """

    # --------------------------------------------------------
    # Build search query
    # --------------------------------------------------------

    query_parts = [
        brand,
        product_name,
    ]

    if category:
        query_parts.append(category)

    query = " ".join(
        part.strip()
        for part in query_parts
        if part and part.strip()
    )

    # --------------------------------------------------------
    # Search Google Shopping
    # --------------------------------------------------------

    results = _search_google_shopping(query)

    candidates: List[Dict[str, Any]] = []

    # --------------------------------------------------------
    # Process shopping results
    # --------------------------------------------------------

    for result in results:

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        source = result.get(
            "source",
            ""
        )

        store = _store_name(source)

        # Ignore unsupported stores
        if not store:
            continue

        # ----------------------------------------------------
        # Product title
        # ----------------------------------------------------

        title = result.get(
            "title",
            ""
        )

        if not title:
            continue

        # ----------------------------------------------------
        # Brand validation
        # ----------------------------------------------------

        if not _brand_matches(
            brand=brand,
            result_title=title,
        ):
            continue

        # ----------------------------------------------------
        # Product matching
        # ----------------------------------------------------

        match_score = _product_match_score(
            product_name=product_name,
            brand=brand,
            result_title=title,
        )

        # Require at least a reasonable product match.
        if match_score < 0.50:
            continue

        # ----------------------------------------------------
        # Price
        # ----------------------------------------------------

        price = result.get(
            "extracted_price"
        )

        if price is None:
            continue

        try:
            price = float(price)
        except (
            TypeError,
            ValueError,
        ):
            continue

        if price <= 0:
            continue

        # ----------------------------------------------------
        # Purchase URL
        # ----------------------------------------------------

        purchase_link = _extract_purchase_link(
            result
        )

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        thumbnail = result.get(
            "thumbnail"
        )

        # ----------------------------------------------------
        # Rating
        # ----------------------------------------------------

        rating = result.get(
            "rating"
        )

        # ----------------------------------------------------
        # Save candidate
        # ----------------------------------------------------

        candidates.append({
            "store": store,
            "price": price,
            "purchase_link": purchase_link,
            "title": title,
            "thumbnail": thumbnail,
            "rating": rating,
            "match_score": round(
                match_score,
                3
            ),
        })

    # ========================================================
    # NO MATCH FOUND
    # ========================================================

    if not candidates:
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
    # SELECT CHEAPEST MATCHING PRODUCT
    # ========================================================

    candidates.sort(
        key=lambda item: item["price"]
    )

    best = candidates[0]

    # ========================================================
    # RETURN ONE PRODUCT
    # ========================================================

    return {
        "found": True,
        "price": best["price"],
        "store": best["store"],
        "purchase_link": best["purchase_link"],
        "title": best["title"],
        "thumbnail": best["thumbnail"],
        "rating": best["rating"],
        "match_score": best["match_score"],
        "source": "live_research",
    }