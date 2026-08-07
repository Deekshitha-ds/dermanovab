from typing import Optional, List
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.orm_models import Product
from app.models.schemas import ProductOut

router = APIRouter(prefix="/api/products", tags=["products"])


@router.get("", response_model=List[ProductOut])
def list_products(
    db: Session = Depends(get_db),
    budget: Optional[float] = None,
    brand: Optional[str] = None,
    skin_type: Optional[str] = None,
    hair_type: Optional[str] = None,
    dermatologist_tested: Optional[bool] = None,
    cruelty_free: Optional[bool] = None,
    fragrance_free: Optional[bool] = None,
    paraben_free: Optional[bool] = None,
    vegan: Optional[bool] = None,
    min_rating: Optional[float] = None,
):
    query = db.query(Product)
    if budget is not None:
        query = query.filter(Product.price <= budget)
    if brand:
        query = query.filter(Product.brand == brand)
    if skin_type:
        query = query.filter(Product.skin_type == skin_type)
    if hair_type:
        query = query.filter(Product.hair_type == hair_type)
    if dermatologist_tested is not None:
        query = query.filter(Product.dermatologist_tested == dermatologist_tested)
    if cruelty_free is not None:
        query = query.filter(Product.cruelty_free == cruelty_free)
    if fragrance_free is not None:
        query = query.filter(Product.fragrance_free == fragrance_free)
    if paraben_free is not None:
        query = query.filter(Product.paraben_free == paraben_free)
    if vegan is not None:
        query = query.filter(Product.vegan == vegan)
    if min_rating is not None:
        query = query.filter(Product.rating >= min_rating)
    return query.order_by(Product.rating.desc()).all()


@router.get("/{product_id}", response_model=ProductOut)
def get_product(product_id: int, db: Session = Depends(get_db)):
    return db.query(Product).filter(Product.id == product_id).first()
