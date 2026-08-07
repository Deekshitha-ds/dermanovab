import os
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth import get_current_user, require_admin
from app.models.orm_models import User, WishlistItem, Product

wishlist_router = APIRouter(prefix="/api/wishlist", tags=["wishlist"])
weather_router = APIRouter(prefix="/api/weather", tags=["weather"])
chatbot_router = APIRouter(prefix="/api/chatbot", tags=["chatbot"])
admin_router = APIRouter(prefix="/api/admin", tags=["admin"])


@wishlist_router.post("/{product_id}")
def add_to_wishlist(product_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if not db.query(Product).filter(Product.id == product_id).first():
        raise HTTPException(status_code=404, detail="Product not found.")
    item = WishlistItem(user_id=current_user.id, product_id=product_id)
    db.add(item)
    db.commit()
    return {"message": "Added to wishlist."}


@wishlist_router.get("")
def get_wishlist(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    items = db.query(WishlistItem).filter(WishlistItem.user_id == current_user.id).all()
    product_ids = [i.product_id for i in items]
    return db.query(Product).filter(Product.id.in_(product_ids)).all()


@wishlist_router.delete("/{product_id}")
def remove_from_wishlist(product_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db.query(WishlistItem).filter(WishlistItem.user_id == current_user.id, WishlistItem.product_id == product_id).delete()
    db.commit()
    return {"message": "Removed from wishlist."}


@weather_router.get("")
def get_weather(lat: float, lon: float):
    """
    Placeholder weather endpoint using simulated seasonal data so the app is
    fully functional without a paid weather API key.
    TO USE REAL WEATHER: sign up for a provider (OpenWeatherMap, WeatherAPI,
    Open-Meteo) and replace the body below with an HTTP call using the API
    key from an environment variable, e.g. os.getenv("WEATHER_API_KEY").
    """
    month = datetime.utcnow().month
    if month in (6, 7, 8, 9):
        condition, temp, humidity, uv = "Rainy", 27, 82, 4
    elif month in (12, 1, 2):
        condition, temp, humidity, uv = "Winter", 19, 40, 3
    else:
        condition, temp, humidity, uv = "Sunny", 33, 50, 8
    return {"condition": condition, "temperature_c": temp, "humidity_pct": humidity, "uv_index": uv}


CHATBOT_RULES = [
    (["niacinamide"], "Niacinamide is generally safe for daily use, morning and night, and suits most skin types including oily and acne-prone skin."),
    (["salicylic"], "Salicylic acid suits oily and acne-prone skin. Start 2-3 times a week and always follow with sunscreen the next morning."),
    (["sunscreen", "spf"], "Apply sunscreen every morning as the last step, and reapply every 2-3 hours outdoors."),
    (["retinol"], "Introduce retinol at night, 2-3 times a week initially, alongside a moisturizer."),
    (["dandruff"], "An anti-fungal shampoo with zinc pyrithione or ketoconazole, used 2-3 times weekly, is a good starting point."),
]


@chatbot_router.post("/ask")
def ask_chatbot(message: str):
    m = message.lower()
    for keywords, answer in CHATBOT_RULES:
        if any(k in m for k in keywords):
            return {"reply": answer, "disclaimer": "Informational only, not a medical diagnosis."}
    return {
        "reply": "Introduce one new active at a time and always patch test first.",
        "disclaimer": "Informational only, not a medical diagnosis.",
    }


@admin_router.get("/users")
def admin_list_users(db: Session = Depends(get_db), _: User = Depends(require_admin)):
    users = db.query(User).all()
    return [{"id": u.id, "name": u.name, "email": u.email, "created_at": u.created_at.isoformat()} for u in users]


@admin_router.delete("/products/{product_id}")
def admin_delete_product(product_id: int, db: Session = Depends(get_db), _: User = Depends(require_admin)):
    db.query(Product).filter(Product.id == product_id).delete()
    db.commit()
    return {"message": "Product deleted."}
