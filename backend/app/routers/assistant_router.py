
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth import get_current_user
from app.models.orm_models import User, Analysis, SavedProduct
from app.services.chatbot_service import generate_nova_response


router = APIRouter(
    prefix="/api/assistant",
    tags=["Assistant"],
)


class AssistantChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=3000)
    consultation_type: Optional[str] = "general"
    conversation_history: Optional[List[dict]] = None


@router.post("/chat")
def assistant_chat(
    payload: AssistantChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Get the latest five analyses belonging ONLY to this user.
    analyses = (
        db.query(Analysis)
        .filter(Analysis.user_id == current_user.id)
        .order_by(Analysis.created_at.desc())
        .limit(5)
        .all()
    )

    latest_analysis = None

    if analyses:
        latest = analyses[0]

        latest_analysis = {
            "analysis_id": latest.id,
            "date": (
                latest.created_at.isoformat()
                if latest.created_at else None
            ),
            "skin_type": latest.detected_type,
            "detected_issues": latest.detected_issues or [],
            "scores": latest.scores or {},
            "result_data": latest.result_data or {},
        }

    # Include recent scans for progress-related questions.
    analysis_history = [
        {
            "analysis_id": item.id,
            "date": (
                item.created_at.isoformat()
                if item.created_at else None
            ),
            "skin_type": item.detected_type,
            "detected_issues": item.detected_issues or [],
            "scores": item.scores or {},
        }
        for item in analyses
    ]

    # Retrieve saved products belonging ONLY to this user.
    saved_products = (
        db.query(SavedProduct)
        .filter(SavedProduct.user_id == current_user.id)
        .order_by(SavedProduct.created_at.desc())
        .limit(20)
        .all()
    )

    saved_product_context = [
        {
            "name": item.product_name,
            "brand": item.brand,
            "category": item.category,
            "description": item.description,
            "price": item.price,
        }
        for item in saved_products
    ]

    # Build context from authenticated user's own database records.
    skin_context = {
        "user_profile": {
            "skin_type": current_user.skin_type,
            "hair_type": current_user.hair_type,
            "sensitive_skin": current_user.sensitive_skin,
            "allergies": current_user.allergies,
            "monthly_budget": current_user.monthly_budget,
        },
        "latest_skin_analysis": latest_analysis,
        "recent_skin_analyses": analysis_history,
        "saved_products": saved_product_context,
        "consultation_type": payload.consultation_type,
    }

    # Keep only the latest 10 previous messages.
    history = []

    for item in (payload.conversation_history or [])[-10:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        message_text = item.get("text")

        if role in ("user", "model") and isinstance(message_text, str):
            history.append({
                "role": role,
                "text": message_text[:3000],
            })

    try:
        reply = generate_nova_response(
            message=payload.message.strip(),
            skin_analysis=skin_context,
            conversation_history=history,
        )

        if not reply:
            raise HTTPException(
                status_code=502,
                detail="The AI returned an empty response.",
            )

        return {
            "reply": reply,
            "consultation_type": payload.consultation_type,
        }

    except HTTPException:
        raise

    except Exception as exc:
        print("Assistant Gemini error:", str(exc))

        raise HTTPException(
            status_code=502,
            detail="The Assistant could not generate a response. Please try again.",
        )
