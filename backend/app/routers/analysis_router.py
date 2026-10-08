from fastapi import APIRouter, Depends, UploadFile, File, HTTPException,Form
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth import get_current_user
from app.models.orm_models import User, Analysis
from app.services.ai_service import analyze_skin, analyze_hair

from app.services.recommendation_service import (
    recommend_products,
    build_routine,
)

from app.services.product_research_service import (
    research_best_offer,
)

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


def _research_product(product):
    """
    Get live shopping information for one recommended product.
    """

    try:
        result = research_best_offer(
            product_name=product.name,
            brand=product.brand,
            category=product.category,
        )

        return result

    except Exception as exc:
        print("Live product research failed:", exc)

        return {
            "found": False,
            "price": None,
            "store": None,
            "purchase_link": None,
            "title": None,
            "thumbnail": None,
            "rating": None,
            "match_score": None,
        }


def _serialize_recommended_product(product):
    """
    Convert a database Product into frontend-friendly data
    and attach live shopping information.
    """

    if not product:
        return None

    live = _research_product(product)

    return {
        "id": product.id,
        "name": product.name,
        "brand": product.brand,
        "category": product.category,

        "price": (
            live["price"]
            if live.get("price") is not None
            else product.price
        ),

        "image_url": (
            live["thumbnail"]
            if live.get("thumbnail")
            else product.image_url
        ),

        "rating": (
            live["rating"]
            if live.get("rating") is not None
            else product.rating
        ),

        "description": product.description,

        "store": live.get("store"),

        "purchase_link": live.get(
            "purchase_link"
        ),

        "live_research": {
            "found": live.get("found", False),
            "store": live.get("store"),
            "price": live.get("price"),
            "match_score": live.get("match_score"),
        },
    }
@router.post("/skin")
async def run_skin_analysis(
    file: UploadFile = File(...),
    mode: str = Form("upload"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty image upload."
        )

    if mode not in {"live", "upload"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid analysis mode. Use 'live' or 'upload'."
        )

    # -----------------------------------------
    # 1. Run AI / YOLO skin analysis
    # -----------------------------------------
    result = analyze_skin(
        image_bytes,
        mode=mode
    )

    detected_issues = result.get("detected_issues", [])
    skin_type = result.get("detected_type")

    # -----------------------------------------
    # 2. Save analysis result to database
    # -----------------------------------------
    record = Analysis(
        user_id=current_user.id,
        mode="skin",
        detected_type=result["detected_type"],
        detected_issues=result["detected_issues"],
        scores=result["scores"],
        face_detected=result["face_detected"],
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    # -----------------------------------------
    # 3. Find products based on detected issues
    # -----------------------------------------
    routine_products = recommend_products(
        db,
        budget=current_user.monthly_budget or 800,
        skin_type=skin_type,
        hair_type=None,
        concerns=detected_issues,
        dermatologist_only=False,
        sensitive=current_user.sensitive_skin or False,
        weather_condition=None,
    )

    # -----------------------------------------
    # 4. Build morning + evening routine
    # -----------------------------------------
    routine = build_routine(routine_products)

    morning_routine = [
        _serialize_recommended_product(product)
        for product in routine["morning"]
        if product
    ]

    evening_routine = [
        _serialize_recommended_product(product)
        for product in routine["night"]
        if product
    ]

    # -----------------------------------------
    # 5. Find one major concern product
    # -----------------------------------------
    major_product = None

    treatment_categories = {
        "serum",
        "treatment",
        "spot treatment",
        "cream",
    }

    for product in routine_products:
        category = (product.category or "").strip().lower()
        concern_text = (product.concern or "").lower()

        issue_match = any(
            issue.lower() in concern_text
            for issue in detected_issues
        )

        category_match = category in treatment_categories

        if issue_match and category_match:
            major_product = product
            break

    if major_product is None:
        major_product = routine.get("best_choice")

    major_recommendation = (
        _serialize_recommended_product(major_product)
        if major_product
        else None
    )

    # -----------------------------------------
    # 6. Return complete scan result
    # -----------------------------------------
    return {
        "analysis_id": record.id,
        **result,
        "major_recommendation": major_recommendation,
        "morning_routine": morning_routine,
        "evening_routine": evening_routine,
    }

@router.post("/hair")
async def run_hair_analysis(file: UploadFile = File(...), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload.")
    result = analyze_hair(image_bytes)
    record = Analysis(
        user_id=current_user.id,
        mode="hair",
        detected_type=result["detected_type"],
        detected_issues=result["detected_issues"],
        scores=result["scores"],
        face_detected=result["face_detected"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {"analysis_id": record.id, **result}


@router.get("/history")
def get_history(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    records = db.query(Analysis).filter(Analysis.user_id == current_user.id).order_by(Analysis.created_at.desc()).all()
    return [
        {
            "id": r.id,
            "mode": r.mode,
            "detected_type": r.detected_type,
            "detected_issues": r.detected_issues,
            "scores": r.scores,
            "created_at": r.created_at.isoformat(),
        }
        for r in records
    ]

# ============================================================
# GET ONE SKIN ANALYSIS REPORT
# ============================================================

@router.get("/history/{analysis_id}")
def get_skin_report(
    analysis_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    record = (
        db.query(Analysis)
        .filter(
            Analysis.id == analysis_id,
            Analysis.user_id == current_user.id,
        )
        .first()
    )

    if not record:
        raise HTTPException(
            status_code=404,
            detail="Saved scan not found."
        )

    return {
        "analysis_id": record.id,
        "mode": record.mode,
        "detected_type": record.detected_type,
        "detected_issues": record.detected_issues,
        "scores": record.scores,
        "face_detected": record.face_detected,
        "created_at": (
            record.created_at.isoformat()
            if record.created_at
            else None
        ),
    }


# ============================================================
# DELETE ONE SKIN ANALYSIS
# ============================================================

@router.delete("/history/{analysis_id}")
def delete_skin_report(
    analysis_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    record = (
        db.query(Analysis)
        .filter(
            Analysis.id == analysis_id,
            Analysis.user_id == current_user.id,
        )
        .first()
    )

    if not record:
        raise HTTPException(
            status_code=404,
            detail="Saved scan not found."
        )

    db.delete(record)
    db.commit()

    return {
        "message": "Saved scan deleted successfully.",
        "analysis_id": analysis_id,
    }