
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth import get_current_user
from app.models.orm_models import SavedProduct, Analysis


router = APIRouter(
    prefix="/api/saved-products",
    tags=["Saved Products"],
)


class SavedProductCreate(BaseModel):
    product_key: str = Field(min_length=1, max_length=500)
    product_name: str = Field(min_length=1, max_length=200)

    brand: Optional[str] = Field(default=None, max_length=120)
    category: Optional[str] = Field(default=None, max_length=100)
    price: Optional[float] = Field(default=None, ge=0)
    image_url: Optional[str] = Field(default=None, max_length=1000)
    rating: Optional[float] = None
    description: Optional[str] = None
    store: Optional[str] = Field(default=None, max_length=120)
    purchase_link: Optional[str] = Field(default=None, max_length=1000)
    analysis_id: Optional[int] = None


def serialize_saved_product(item: SavedProduct):
    return {
        "id": item.id,
        "product_key": item.product_key,
        "product_name": item.product_name,
        "name": item.product_name,
        "brand": item.brand,
        "category": item.category,
        "price": item.price,
        "image_url": item.image_url,
        "rating": item.rating,
        "description": item.description,
        "store": item.store,
        "purchase_link": item.purchase_link,
        "analysis_id": item.analysis_id,
        "created_at": (
            item.created_at.isoformat()
            if item.created_at
            else None
        ),
    }


@router.post("")
def save_product(
    payload: SavedProductCreate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    # If linked to a scan, verify that the scan belongs to this user.
    if payload.analysis_id is not None:
        analysis = (
            db.query(Analysis)
            .filter(
                Analysis.id == payload.analysis_id,
                Analysis.user_id == current_user.id,
            )
            .first()
        )

        if not analysis:
            raise HTTPException(
                status_code=404,
                detail="Scan not found for this account.",
            )

    # Avoid saving the same product more than once for this user.
    existing = (
        db.query(SavedProduct)
        .filter(
            SavedProduct.user_id == current_user.id,
            SavedProduct.product_key == payload.product_key,
        )
        .first()
    )

    if existing:
        return {
            "message": "Product is already saved.",
            "product": serialize_saved_product(existing),
        }

    item = SavedProduct(
        user_id=current_user.id,
        product_key=payload.product_key,
        product_name=payload.product_name,
        brand=payload.brand,
        category=payload.category,
        price=payload.price,
        image_url=payload.image_url,
        rating=payload.rating,
        description=payload.description,
        store=payload.store,
        purchase_link=payload.purchase_link,
        analysis_id=payload.analysis_id,
    )

    db.add(item)
    db.commit()
    db.refresh(item)

    return {
        "message": "Product saved successfully.",
        "product": serialize_saved_product(item),
    }


@router.get("")
def get_saved_products(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    items = (
        db.query(SavedProduct)
        .filter(SavedProduct.user_id == current_user.id)
        .order_by(SavedProduct.created_at.desc())
        .all()
    )

    return {
        "products": [
            serialize_saved_product(item)
            for item in items
        ],
        "total": len(items),
    }


@router.delete("/{saved_product_id}")
def remove_saved_product(
    saved_product_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    item = (
        db.query(SavedProduct)
        .filter(
            SavedProduct.id == saved_product_id,
            SavedProduct.user_id == current_user.id,
        )
        .first()
    )

    if not item:
        raise HTTPException(
            status_code=404,
            detail="Saved product not found.",
        )

    db.delete(item)
    db.commit()

    return {"message": "Product removed from saved products."}
