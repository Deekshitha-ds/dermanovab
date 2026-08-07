from typing import List, Optional
from sqlalchemy.orm import Session
from app.models.orm_models import Product


def recommend_products(
    db: Session,
    budget: float,
    skin_type: Optional[str] = None,
    hair_type: Optional[str] = None,
    concerns: Optional[List[str]] = None,
    dermatologist_only: bool = False,
    sensitive: bool = False,
    weather_condition: Optional[str] = None,
) -> List[Product]:
    concerns = concerns or []
    query = db.query(Product).filter(Product.price <= budget)
    if dermatologist_only:
        query = query.filter(Product.dermatologist_tested.is_(True))
    if sensitive:
        query = query.filter(Product.fragrance_free.is_(True))
    products = query.all()

    def score(p: Product) -> float:
        s = p.rating * 10
        if skin_type and p.skin_type == skin_type:
            s += 15
        if hair_type and p.hair_type == hair_type:
            s += 15
        if p.concern in concerns:
            s += 25
        if weather_condition == "Sunny" and p.category == "Sunscreen":
            s += 30
        if weather_condition == "Rainy" and p.category == "Moisturizer":
            s += 10
        if weather_condition == "Winter" and p.category == "Moisturizer":
            s += 20
        return s

    products.sort(key=score, reverse=True)
    return products


def build_routine(products: List[Product]):
    def pick(categories, used):
        for p in products:
            if p.category in categories and p.id not in used:
                return p
        return None

    used_ids = set()
    morning_cats = [["Face Wash", "Cleanser"], ["Moisturizer"], ["Sunscreen"]]
    night_cats = [["Cleanser", "Face Wash"], ["Serum"], ["Moisturizer"]]

    morning, night = [], []
    for cats in morning_cats:
        item = pick(cats, used_ids)
        if item:
            used_ids.add(item.id)
            morning.append(item)
    for cats in night_cats:
        item = pick(cats, used_ids)
        if item:
            used_ids.add(item.id)
            night.append(item)

    return {
        "morning": morning,
        "night": night,
        "morning_total": sum(p.price for p in morning),
        "night_total": sum(p.price for p in night),
        "best_choice": products[0] if products else None,
        "budget_choice": min(products, key=lambda p: p.price) if products else None,
        "premium_choice": max(products, key=lambda p: p.price) if products else None,
    }
