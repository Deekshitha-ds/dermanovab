from fastapi import APIRouter, Depends, UploadFile, File
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth import get_current_user
from app.models.orm_models import User
from app.models.schemas import QuestionnaireIn

router = APIRouter(prefix="/api/users", tags=["users"])


@router.put("/questionnaire")
def save_questionnaire(payload: QuestionnaireIn, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    for field, value in payload.model_dump().items():
        setattr(current_user, field if field != "monthly_budget" else "monthly_budget", value)
    db.commit()
    return {"message": "Preferences saved."}


@router.post("/profile-picture")
async def upload_profile_picture(file: UploadFile = File(...), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # TODO: replace with a real Cloudinary (or S3) upload and store the returned URL.
    # from app.services.image_storage import upload_image
    # url = upload_image(await file.read())
    fake_url = f"https://cdn.example.com/avatars/{current_user.id}_{file.filename}"
    current_user.profile_picture_url = fake_url
    db.commit()
    return {"profile_picture_url": fake_url}
