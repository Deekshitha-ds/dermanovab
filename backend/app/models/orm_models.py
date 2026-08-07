from sqlalchemy import (
    Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey, JSON
)
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), nullable=False)
    email = Column(String(180), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    profile_picture_url = Column(String(500), nullable=True)
    age = Column(Integer, nullable=True)
    gender = Column(String(30), nullable=True)
    skin_type = Column(String(30), nullable=True)
    hair_type = Column(String(30), nullable=True)
    sensitive_skin = Column(Boolean, default=False)
    pregnant = Column(Boolean, nullable=True)
    allergies = Column(String(300), nullable=True)
    preferred_brands = Column(String(300), nullable=True)
    monthly_budget = Column(Float, default=800)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    analyses = relationship("Analysis", back_populates="user")
    wishlist_items = relationship("WishlistItem", back_populates="user")


class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    brand = Column(String(80), nullable=False)
    category = Column(String(60), nullable=False)
    price = Column(Float, nullable=False)
    ingredients = Column(JSON, nullable=False, default=list)
    skin_type = Column(String(30), nullable=True)
    hair_type = Column(String(30), nullable=True)
    concern = Column(String(60), nullable=True)
    dermatologist_tested = Column(Boolean, default=False)
    fragrance_free = Column(Boolean, default=False)
    paraben_free = Column(Boolean, default=False)
    cruelty_free = Column(Boolean, default=False)
    vegan = Column(Boolean, default=False)
    rating = Column(Float, default=4.0)
    image_url = Column(String(500), nullable=True)
    description = Column(Text, nullable=True)
    purchase_link = Column(String(500), nullable=True)


class Analysis(Base):
    __tablename__ = "analyses"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    mode = Column(String(10), nullable=False)  # 'skin' or 'hair'
    image_url = Column(String(500), nullable=True)
    detected_type = Column(String(30), nullable=True)
    detected_issues = Column(JSON, nullable=False, default=list)
    scores = Column(JSON, nullable=False, default=dict)  # {"health":.., "hydration":.., "oiliness":.., "confidence":..}
    face_detected = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="analyses")


class Recommendation(Base):
    __tablename__ = "recommendations"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    analysis_id = Column(Integer, ForeignKey("analyses.id"), nullable=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    routine_slot = Column(String(20), nullable=True)  # morning / night / weekly / monthly
    created_at = Column(DateTime, default=datetime.utcnow)


class WishlistItem(Base):
    __tablename__ = "wishlist"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="wishlist_items")


class History(Base):
    __tablename__ = "history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    action = Column(String(60), nullable=False)
    details = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
