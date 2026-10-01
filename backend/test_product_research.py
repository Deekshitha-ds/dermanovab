
from app.services.product_research_service import research_best_offer


result = research_best_offer(
    product_name="Niacinamide 10% Serum",
    brand="The Ordinary",
    category="face serum",
)


print("\nRESULT")
print("=" * 60)

print("Found:", result.get("found"))
print("Product:", result.get("title"))
print("Price:", result.get("price"))
print("Store:", result.get("store"))
print("Rating:", result.get("rating"))
print("Image:", result.get("thumbnail"))
print("Link:", result.get("purchase_link"))
print("Match Score:", result.get("match_score"))

print("=" * 60)