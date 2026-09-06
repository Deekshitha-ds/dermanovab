from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.orm_models import User
from app.models.schemas import UserRegister, UserLogin, Token
from app.auth import hash_password, verify_password, create_access_token, get_current_user

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=Token)
def register(payload: UserRegister, db: Session = Depends(get_db)):
    
    print(payload)
    print(type(payload.password))
    print(payload.password)
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=400, detail="An account with this email already exists.")
    user = User(name=payload.name, email=payload.email, hashed_password=hash_password(payload.password))
    db.add(user)
    db.commit()
    token = create_access_token({"sub": user.email})
    return {"access_token": token}


@router.post("/login", response_model=Token)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    token = create_access_token({"sub": user.email})
    return {"access_token": token}

@router.post("/token", response_model=Token)
def token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(
        User.email == form_data.username
    ).first()

    if not user or not verify_password(
        form_data.password,
        user.hashed_password
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password."
        )

    access_token = create_access_token({
        "sub": user.email
    })

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }
@router.post("/forgot-password")
def forgot_password(email: str, db: Session = Depends(get_db)):
    # Always return a generic response to avoid leaking which emails are registered.
    # TODO: integrate a real email provider (e.g. SendGrid, SES) to send a reset link.
    return {"message": "If an account exists for this email, a reset link has been sent."}


@router.get("/me")
def me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "name": current_user.name,
        "email": current_user.email,
        "profile_picture_url": current_user.profile_picture_url,
        "is_admin": current_user.is_admin,
    }
