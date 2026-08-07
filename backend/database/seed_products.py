"""
Run with: python -m database.seed_products
Populates the products table with 60 sample products across the 10 brands
specified in the brief. Mirrors the same catalog used in the frontend mock
so backend and frontend stay consistent once wired together.
"""
import random
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal, Base, engine
from app.models.orm_models import Product

BRANDS = ["Cetaphil", "CeraVe", "Minimalist", "The Derma Co", "Dot & Key", "Simple", "Foxtale", "Bioderma", "Neutrogena", "La Roche-Posay"]
SKIN_CATEGORIES = ["Cleanser", "Face Wash", "Moisturizer", "Sunscreen", "Serum", "Toner", "Face Mask", "Lip Balm"]
HAIR_CATEGORIES = ["Shampoo", "Conditioner", "Hair Oil", "Hair Serum"]
CONCERNS = ["Acne", "Dark Spots", "Large Pores", "Uneven Tone", "Dryness", "Oiliness", "Sensitivity", "Ageing", "Hair Fall", "Dandruff", "Frizz"]
SKIN_TYPES = ["Oily", "Dry", "Combination", "Normal", "Sensitive"]
HAIR_TYPES = ["Straight", "Wavy", "Curly", "Coily"]
INGREDIENTS = ["Niacinamide", "Salicylic Acid", "Hyaluronic Acid", "Ceramides", "Zinc PCA", "Vitamin C", "Retinol", "Centella Asiatica", "Aloe Vera", "Glycerin", "Panthenol", "Peptides", "Squalane", "Green Tea Extract", "Caffeine"]
PRICES = [149, 199, 249, 299, 349, 399, 449, 499, 599, 699, 799, 899, 999, 1199, 1499]


def seed():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    rng = random.Random(42)
    if db.query(Product).count() > 0:
        print("Products already seeded, skipping.")
        return
    for i in range(60):
        is_hair = i % 5 == 0
        brand = BRANDS[i % len(BRANDS)]
        category = rng.choice(HAIR_CATEGORIES if is_hair else SKIN_CATEGORIES)
        concern = rng.choice(CONCERNS)
        ingredients = list({rng.choice(INGREDIENTS) for _ in range(3)})
        skin_type = "All" if is_hair else rng.choice(SKIN_TYPES)
        hair_type = rng.choice(HAIR_TYPES) if is_hair else "All"
        product = Product(
            name=f"{brand} {concern.split()[0]} {category}",
            brand=brand,
            category=category,
            price=rng.choice(PRICES),
            ingredients=ingredients,
            skin_type=skin_type,
            hair_type=hair_type,
            concern=concern,
            dermatologist_tested=rng.random() > 0.3,
            fragrance_free=rng.random() > 0.4,
            paraben_free=rng.random() > 0.2,
            cruelty_free=rng.random() > 0.25,
            vegan=rng.random() > 0.5,
            rating=round(3.5 + rng.random() * 1.5, 1),
            description=f"A {skin_type.lower()}-friendly {category.lower()} formulated with {', '.join(ingredients)} to help with {concern.lower()}.",
            purchase_link="#",
        )
        db.add(product)
    db.commit()
    print("Seeded 60 products.")


if __name__ == "__main__":
    seed()
