from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict


class UserRegister(BaseModel):
    name: str
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class QuestionnaireIn(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    skin_type: Optional[str] = None
    hair_type: Optional[str] = None
    sensitive_skin: bool = False
    pregnant: Optional[bool] = None
    allergies: Optional[str] = None
    preferred_brands: Optional[str] = None
    monthly_budget: float = 800


class AnalysisResult(BaseModel):
    mode: str
    detected_type: str
    detected_issues: List[str]
    scores: Dict[str, float]
    face_detected: bool


class ProductOut(BaseModel):
    id: int
    name: str
    brand: str
    category: str
    price: float
    ingredients: List[str]
    skin_type: Optional[str]
    hair_type: Optional[str]
    concern: Optional[str]
    dermatologist_tested: bool
    fragrance_free: bool
    paraben_free: bool
    cruelty_free: bool
    vegan: bool
    rating: float
    description: Optional[str]
    purchase_link: Optional[str]

    class Config:
        from_attributes = True


class RecommendationRequest(BaseModel):
    budget: float
    skin_type: Optional[str] = None
    hair_type: Optional[str] = None
    concerns: List[str] = []
    dermatologist_only: bool = False
    sensitive: bool = False
    weather_condition: Optional[str] = None
