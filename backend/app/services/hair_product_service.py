
from typing import Any
from app.services.product_research_service import research_best_offer


# Candidate products for live price research.
# These are search candidates, not guaranteed available offers.
HAIR_PRODUCTS = [
    {
        "name": "Anti-Dandruff Shampoo",
        "brand": "Head & Shoulders",
        "category": "Anti-dandruff shampoo",
        "concerns": ["Dandruff"],
    },
    {
        "name": "Scalp Advanced Anti-Dandruff Shampoo",
        "brand": "L'Oreal Professionnel",
        "category": "Anti-dandruff shampoo",
        "concerns": ["Dandruff"],
    },
    {
        "name": "Smooth and Manageable Conditioner",
        "brand": "Dove",
        "category": "Conditioner",
        "concerns": ["Frizz", "Dryness", "Breakage", "Dullness"],
    },
    {
        "name": "Daily Moisture Shampoo",
        "brand": "Dove",
        "category": "Gentle shampoo",
        "concerns": ["Dryness", "Oily scalp", "Hair fall"],
    },
]



def get_hair_products(
    concerns: list[str],
    product_categories: list[str],
    max_products: int = 3,
) -> dict[str, Any]:
    """Research relevant hair products without a budget restriction."""

    matched = [
        product for product in HAIR_PRODUCTS
        if set(product["concerns"]) & set(concerns)
        or product["category"] in product_categories
    ]

    matched = matched[:max_products]
    products = []

    for candidate in matched:
        try:
            offer = research_best_offer(
                product_name=candidate["name"],
                brand=candidate["brand"],
                category=candidate["category"],
            )

            if not offer.get("found"):
                continue

            price = offer.get("price")
            if price is None:
                continue

            price = float(price)

            if not offer.get("store") or not offer.get("purchase_link"):
                continue

            products.append({
                "name": offer.get("title") or candidate["name"],
                "brand": candidate["brand"],
                "category": candidate["category"],
                "price": price,
                "store": offer["store"],
                "purchase_link": offer["purchase_link"],
                "image": offer.get("thumbnail"),
                "rating": offer.get("rating"),
                "source": offer.get("source"),
            })

        except Exception as exc:
            print("Hair product research failed:", exc)

    return {"products": products}
