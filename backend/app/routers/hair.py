
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

from app.services.hair_product_service import get_hair_products

router = APIRouter(
    prefix="/hair",
    tags=["Hair Care Intelligence"],
)


class HairProfile(BaseModel):
    hair_type: Literal["Straight", "Wavy", "Curly", "Coily"]
    hair_length: Literal["Short", "Medium", "Long"]
    hair_texture: Literal["Fine", "Medium", "Thick"]
    scalp_condition: Literal[
        "Oily", "Dry", "Normal", "Combination", "Unsure"
    ]
    hair_concerns: list[str] = Field(min_length=1)


def build_routine(profile: HairProfile) -> list[dict]:
    routine = []

    if profile.scalp_condition == "Oily":
        routine.append({
            "step": "Scalp cleansing",
            "recommendation": (
                "Use a gentle shampoo as needed. Avoid applying "
                "heavy conditioner directly to the scalp."
            ),
            "product_category": "Shampoo",
        })
    elif profile.scalp_condition == "Dry":
        routine.append({
            "step": "Gentle cleansing",
            "recommendation": (
                "Choose a gentle shampoo and avoid excessive washing "
                "if it worsens scalp dryness."
            ),
            "product_category": "Gentle shampoo",
        })
    elif profile.scalp_condition == "Combination":
        routine.append({
            "step": "Balanced scalp care",
            "recommendation": (
                "Cleanse according to scalp oiliness and condition "
                "the hair lengths separately."
            ),
            "product_category": "Balanced shampoo",
        })
    else:
        routine.append({
            "step": "Regular cleansing",
            "recommendation": (
                "Choose a gentle shampoo suited to your scalp and "
                "adjust washing frequency to your needs."
            ),
            "product_category": "Gentle shampoo",
        })

    concern_rules = {
        "Frizz": (
            "Frizz management",
            "Condition the lengths and ends; consider a lightweight "
            "leave-in conditioner.",
            "Conditioner",
        ),
        "Dryness": (
            "Moisture care",
            "Use a conditioning product on dry lengths and ends.",
            "Conditioner or hair mask",
        ),
        "Breakage": (
            "Breakage prevention",
            "Handle wet hair gently, reduce heat styling, and use "
            "conditioner to improve manageability.",
            "Conditioner",
        ),
        "Dandruff": (
            "Flake care",
            "Consider an evidence-based anti-dandruff shampoo and "
            "follow its label. Seek medical advice if symptoms persist.",
            "Anti-dandruff shampoo",
        ),
        "Hair fall": (
            "Hair shedding care",
            "Avoid tight hairstyles and harsh treatments. Persistent "
            "or sudden hair loss should be assessed by a healthcare professional.",
            "Gentle hair care",
        ),
        "Oily scalp": (
            "Oil management",
            "Wash when needed and avoid heavy products on the scalp "
            "if they make it feel oilier.",
            "Lightweight shampoo",
        ),
        "Dullness": (
            "Shine and manageability",
            "Use conditioner on the hair lengths and limit excessive "
            "heat styling.",
            "Conditioner",
        ),
    }

    for concern in profile.hair_concerns:
        title, recommendation, category = concern_rules[concern]
        routine.append({
            "step": title,
            "recommendation": recommendation,
            "product_category": category,
        })

    return routine


@router.post("/recommendations")
def get_hair_recommendations(profile: HairProfile):
    allowed_concerns = {
        "Hair fall",
        "Dandruff",
        "Frizz",
        "Dryness",
        "Breakage",
        "Oily scalp",
        "Dullness",
    }

    invalid = set(profile.hair_concerns) - allowed_concerns
    if invalid:
        raise HTTPException(
            status_code=422,
            detail="One or more hair concerns are invalid.",
        )

    routine = build_routine(profile)

    try:
        product_data = get_hair_products(
            concerns=profile.hair_concerns,
            product_categories=[
                item["product_category"] for item in routine
            ],
            max_products=3,
        )
    except Exception as exc:
        print("Hair product recommendations unavailable:", exc)
        product_data = {"products": []}

    return {
        "success": True,
        "module": "DermaNova Hair Care Intelligence",
        "hair_profile": profile.model_dump(),
        "routine": routine,
        "product_categories": list(dict.fromkeys(
            item["product_category"] for item in routine
        )),
        "products": product_data.get("products", []),
        "message": "Your personalized hair-care plan has been generated.",
    }
