from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth import get_current_user
from app.models.orm_models import User, Analysis
from app.services.ai_service import analyze_skin, analyze_hair

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


@router.post("/skin")
async def run_skin_analysis(file: UploadFile = File(...), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload.")
    result = analyze_skin(image_bytes)
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
    return {"analysis_id": record.id, **result}


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
    records = db.query(Analysis).filter(Analysis.user_id == current_user.id).order_by(Analysis.created_at).all()
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
