"""
DermaNova AI - Live Product Price Research

Researches current Google Shopping results and then uses
SerpApi's Immersive Product API to obtain direct merchant
purchase links.

Supported stores:
- Amazon
- Nykaa
- Purplle

The service compares supported stores internally and returns
ONE selected product offer to the frontend.
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
            ""
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
    Calculates how closely a shopping result matches
    the recommended product.
    """

    product_tokens = _tokens(product_name)
    result_tokens = _tokens(result_title)

    if not product_tokens or not result_tokens:
        return 0.0

    matched = product_tokens & result_tokens

    return len(matched) / len(product_tokens)


# ============================================================
# SERPAPI GOOGLE SHOPPING
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
        []
    )


# ============================================================
# IMMERSIVE PRODUCT API
# ============================================================

def _get_immersive_product(
    page_token: str
) -> Dict[str, Any]:
    """
    Gets detailed product information including merchant
    stores and direct purchase links.
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
# DIRECT MERCHANT OFFER EXTRACTION
# ============================================================

def _extract_store_offers(
    product_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Extracts supported merchant offers from the immersive
    product response.

    Only Amazon, Nykaa and Purplle are returned.
    """

    product_results = product_data.get(
        "product_results",
        {}
    )

    stores = product_results.get(
        "stores",
        []
    )

    offers: List[Dict[str, Any]] = []

    for store_result in stores:

        if not isinstance(
            store_result,
            dict
        ):
            continue

        # ----------------------------------------------------
        # Direct merchant link
        # ----------------------------------------------------

        link = store_result.get(
            "link"
        )

        if not isinstance(link, str):
            continue

        link = link.strip()

        if not link:
            continue

        # ----------------------------------------------------
        # Detect store
        # ----------------------------------------------------

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

            value = store_result.get(
                field
            )

            store = _store_name(value)

            if store:
                break

        # If the store name isn't present,
        # identify it from the URL.
        if not store:
            store = _store_from_url(
                link
            )

        # Ignore unsupported stores.
        if not store:
            continue

        # ----------------------------------------------------
        # Price
        # ----------------------------------------------------

        price = store_result.get(
            "extracted_price"
        )

        if price is None:
            price = store_result.get(
                "price"
            )

        if price is None:
            continue

        try:
            if isinstance(price, str):
                price = re.sub(
                    r"[^0-9.]+",
                    "",
                    price
                )

            price = float(price)

        except (
            TypeError,
            ValueError,
        ):
            continue

        if price <= 0:
            continue

        # ----------------------------------------------------
        # Save offer
        # ----------------------------------------------------

        offers.append({
            "store": store,
            "price": price,
            "purchase_link": link,
        })

    return offers


# ============================================================
# LIVE PRODUCT RESEARCH
# ============================================================

def research_best_offer(
    product_name: str,
    brand: str,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Searches Google Shopping, finds the matching product,
    opens the immersive product page, compares supported
    merchant offers, and returns ONE selected offer.
    """

    # --------------------------------------------------------
    # Build query
    # --------------------------------------------------------

    query_parts = [
        brand,
        product_name,
    ]

    if category:
        query_parts.append(
            category
        )

    query = " ".join(
        part.strip()
        for part in query_parts
        if part and part.strip()
    )

    # --------------------------------------------------------
    # Google Shopping search
    # --------------------------------------------------------

    results = _search_google_shopping(
        query
    )

    candidates: List[Dict[str, Any]] = []

    # --------------------------------------------------------
    # Find matching products
    # --------------------------------------------------------

    for result in results:

        title = result.get(
            "title",
            ""
        )

        if not title:
            continue

        # Brand must match.
        if not _brand_matches(
            brand=brand,
            result_title=title,
        ):
            continue

        # Product similarity.
        match_score = _product_match_score(
            product_name=product_name,
            brand=brand,
            result_title=title,
        )

        if match_score < 0.50:
            continue

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
                3
            ),
            "immersive_token": result.get(
                "immersive_product_page_token"
            ),
        })

    # --------------------------------------------------------
    # No matching product
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Pick strongest product match
    #
    # We use match score first so that the immersive API
    # is opened for the most relevant product.
    # --------------------------------------------------------

    candidates.sort(
        key=lambda item: (
            -item["match_score"],
            item["price"],
        )
    )

    best_product = candidates[0]

    # --------------------------------------------------------
    # Get direct merchant offers
    # --------------------------------------------------------

    immersive_token = best_product.get(
        "immersive_token"
    )

    offers: List[Dict[str, Any]] = []

    if immersive_token:

        try:
            immersive_data = _get_immersive_product(
                immersive_token
            )

            offers = _extract_store_offers(
                immersive_data
            )

        except Exception as exc:

            print(
                "Immersive product research failed:",
                exc
            )

    # --------------------------------------------------------
    # If direct merchant offers are available
    # --------------------------------------------------------

    if offers:

        # Cheapest supported merchant offer.
        offers.sort(
            key=lambda item: item["price"]
        )

        best_offer = offers[0]

        return {
            "found": True,
            "price": best_offer["price"],
            "store": best_offer["store"],
            "purchase_link": best_offer[
                "purchase_link"
            ],
            "title": best_product["title"],
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

    # --------------------------------------------------------
    # Fallback
    #
    # If immersive research does not return a supported
    # merchant link, still return the matching product,
    # but don't expose a fake Google Shopping URL.
    # --------------------------------------------------------

    return {
        "found": True,
        "price": best_product["price"],
        "store": None,
        "purchase_link": None,
        "title": best_product["title"],
        "thumbnail": best_product["thumbnail"],
        "rating": best_product["rating"],
        "match_score": best_product[
            "match_score"
        ],
        "source": "live_research",
    }