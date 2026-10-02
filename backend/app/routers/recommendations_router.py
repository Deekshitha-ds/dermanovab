from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth import get_current_user
from app.models.orm_models import User
from app.models.schemas import RecommendationRequest, ProductOut
from app.services.recommendation_service import (
    recommend_products,
    build_routine,
)
from app.services.product_research_service import (
    research_best_offer,
)


router = APIRouter(
    prefix="/api/recommendations",
    tags=["recommendations"],
)


def _live_product_data(product):
    """
    Researches the currently selected database product
    using live shopping data.

    Returns one selected merchant offer.
    """

    try:

        result = research_best_offer(
            product_name=product.name,
            brand=product.brand,
            category=product.category,
        )

        return result

    except Exception as exc:

        print(
            "Live product research failed:",
            exc,
        )

        return {
            "found": False,
            "price": None,
            "store": None,
            "purchase_link": None,
            "title": None,
            "thumbnail": None,
            "rating": None,
            "match_score": None,
            "source": "live_research",
        }


@router.post("/routine")
def get_routine(
    payload: RecommendationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    # ========================================================
    # EXISTING PERSONALIZED RECOMMENDATION ENGINE
    # ========================================================

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

    # ========================================================
    # SERIALIZE PRODUCT + LIVE SHOPPING DATA
    # ========================================================

    def ser(p):

        if not p:
            return None

        # Existing database product data
        data = ProductOut.model_validate(
            p
        ).model_dump()

        # ----------------------------------------------------
        # LIVE PRODUCT RESEARCH
        # ----------------------------------------------------

        live = _live_product_data(p)

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        if live.get("thumbnail"):

            data["image_url"] = live[
                "thumbnail"
            ]

        elif getattr(
            p,
            "image_url",
            None,
        ):

            data["image_url"] = p.image_url

        else:

            data["image_url"] = None

        # ----------------------------------------------------
        # LIVE PRICE
        # ----------------------------------------------------

        if live.get("price") is not None:

            data["price"] = live[
                "price"
            ]

        # ----------------------------------------------------
        # LIVE RATING
        # ----------------------------------------------------

        if live.get("rating") is not None:

            data["rating"] = live[
                "rating"
            ]

        # ----------------------------------------------------
        # LIVE PURCHASE LINK
        # ----------------------------------------------------

        if live.get("purchase_link"):

            data["purchase_link"] = live[
                "purchase_link"
            ]

        # ----------------------------------------------------
        # ONE SELECTED STORE OFFER
        # ----------------------------------------------------

        if (
            live.get("store")
            and live.get("price") is not None
            and live.get("purchase_link")
        ):

            data["offers"] = [
                {
                    "id": None,
                    "store": live[
                        "store"
                    ],
                    "price": live[
                        "price"
                    ],
                    "purchase_link": live[
                        "purchase_link"
                    ],
                    "availability": True,
                }
            ]

        else:

            # No live merchant offer found.
            # Do not expose multiple old store links.
            data["offers"] = []

        # ----------------------------------------------------
        # LIVE RESEARCH STATUS
        # ----------------------------------------------------

        data["live_research"] = {
            "found": live.get(
                "found",
                False,
            ),
            "store": live.get(
                "store"
            ),
            "price": live.get(
                "price"
            ),
            "match_score": live.get(
                "match_score"
            ),
        }

        return data

    # ========================================================
    # RESPONSE
    # ========================================================

    return {
        "morning": [
            ser(p)
            for p in routine["morning"]
        ],

        "night": [
            ser(p)
            for p in routine["night"]
        ],

        "morning_total": routine[
            "morning_total"
        ],

        "night_total": routine[
            "night_total"
        ],

        "best_choice": ser(
            routine["best_choice"]
        ),

        "budget_choice": ser(
            routine["budget_choice"]
        ),

        "premium_choice": ser(
            routine["premium_choice"]
        ),
    }