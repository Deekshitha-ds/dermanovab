from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth import get_current_user
from app.models.orm_models import User
from app.models.schemas import RecommendationRequest, ProductOut
from app.services.recommendation_service import recommend_products, build_routine

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])


@router.post("/routine")
def get_routine(payload: RecommendationRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    products = recommend_products(
        db,
        budget=payload.budget,
        skin_type=payload.skin_type,
        hair_type=payload.hair_type,
        concerns=payload.concerns,
        dermatologist_only=payload.dermatologist_only,
        sensitive=payload.sensitive,
        weather_condition=payload.weather_condition,
    )
    routine = build_routine(products)

    def ser(p):
        return ProductOut.model_validate(p).model_dump() if p else None

    return {
        "morning": [ser(p) for p in routine["morning"]],
        "night": [ser(p) for p in routine["night"]],
        "morning_total": routine["morning_total"],
        "night_total": routine["night_total"],
        "best_choice": ser(routine["best_choice"]),
        "budget_choice": ser(routine["budget_choice"]),
        "premium_choice": ser(routine["premium_choice"]),
    }
