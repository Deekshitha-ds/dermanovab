"""
DermaNova AI recommendation engine.

Generates general skincare guidance from the visual skin
assessment and detected concerns.

This is informational guidance, not medical diagnosis.
"""


def _has_issue(issues, keyword):
    return any(
        keyword.lower() in str(issue).lower()
        for issue in issues
    )


def generate_recommendations(
    skin_type,
    issues,
    hydration,
    oiliness,
):
    recommendations = []

    # --------------------------------------------------
    # BASIC ROUTINE
    # --------------------------------------------------

    recommendations.append({
        "category": "Daily Routine",
        "title": "Gentle cleansing",
        "description": (
            "Use a gentle cleanser and avoid aggressive scrubbing "
            "that may irritate the skin."
        ),
    })

    recommendations.append({
        "category": "Sun Protection",
        "title": "Daily sunscreen",
        "description": (
            "Use a broad-spectrum sunscreen during the day, "
            "especially when pigmentation or dark spots are present."
        ),
    })

    # --------------------------------------------------
    # SKIN TYPE
    # --------------------------------------------------

    if skin_type == "Oily":

        recommendations.append({
            "category": "Skin Type",
            "title": "Manage excess oil",
            "description": (
                "Choose lightweight, non-comedogenic products "
                "and avoid overly heavy moisturizers."
            ),
        })

    elif skin_type == "Dry":

        recommendations.append({
            "category": "Skin Type",
            "title": "Support the skin barrier",
            "description": (
                "Use a gentle cleanser and a moisturizing product "
                "to help maintain the skin barrier."
            ),
        })

    elif skin_type == "Combination":

        recommendations.append({
            "category": "Skin Type",
            "title": "Balance different areas",
            "description": (
                "Use lightweight products and avoid excessively "
                "drying the oilier areas of the face."
            ),
        })

    else:

        recommendations.append({
            "category": "Skin Type",
            "title": "Maintain your routine",
            "description": (
                "Continue with a gentle cleanser, moisturizer, "
                "and daily sun protection."
            ),
        })

    # --------------------------------------------------
    # HYDRATION
    # --------------------------------------------------

    if hydration < 45:

        recommendations.append({
            "category": "Hydration",
            "title": "Increase hydration support",
            "description": (
                "Your visual hydration indicator is relatively low. "
                "Consider a gentle moisturizer and avoid excessive "
                "use of drying products."
            ),
        })

    elif hydration >= 70:

        recommendations.append({
            "category": "Hydration",
            "title": "Maintain hydration",
            "description": (
                "Your visual hydration indicator is relatively good. "
                "Continue consistent moisturizing and gentle cleansing."
            ),
        })

    # --------------------------------------------------
    # OILINESS
    # --------------------------------------------------

    if oiliness >= 70:

        recommendations.append({
            "category": "Oil Control",
            "title": "Control excess surface oil",
            "description": (
                "Use lightweight, non-comedogenic skincare and "
                "avoid repeatedly stripping the skin."
            ),
        })

    # --------------------------------------------------
    # ACNE-RELATED CONCERNS
    # --------------------------------------------------

    if (
        _has_issue(issues, "papule")
        or _has_issue(issues, "pustule")
        or _has_issue(issues, "nodule")
        or _has_issue(issues, "blackhead")
        or _has_issue(issues, "whitehead")
        or _has_issue(issues, "acne")
    ):

        recommendations.append({
            "category": "Acne Care",
            "title": "Avoid picking or squeezing",
            "description": (
                "Avoid squeezing or picking detected areas, "
                "as this can increase irritation and the risk "
                "of marks."
            ),
        })

    # --------------------------------------------------
    # NODULES
    # --------------------------------------------------

    if _has_issue(issues, "nodule"):

        recommendations.append({
            "category": "Professional Care",
            "title": "Consider dermatological evaluation",
            "description": (
                "Persistent, painful, or deep skin nodules may "
                "require professional evaluation by a dermatologist."
            ),
        })

    # --------------------------------------------------
    # DARK SPOTS
    # --------------------------------------------------

    if _has_issue(issues, "dark spot"):

        recommendations.append({
            "category": "Pigmentation",
            "title": "Prioritize sun protection",
            "description": (
                "Consistent broad-spectrum sunscreen can help "
                "protect against worsening visible dark spots."
            ),
        })

    # --------------------------------------------------
    # PIGMENTATION
    # --------------------------------------------------

    if _has_issue(issues, "pigmentation"):

        recommendations.append({
            "category": "Pigmentation",
            "title": "Support even-looking skin tone",
            "description": (
                "Consider gentle skincare ingredients such as "
                "niacinamide and avoid unnecessary skin irritation."
            ),
        })

    # --------------------------------------------------
    # UNEVEN SKIN TONE
    # --------------------------------------------------

    if _has_issue(issues, "uneven skin tone"):

        recommendations.append({
            "category": "Skin Tone",
            "title": "Focus on consistent skin protection",
            "description": (
                "Daily sunscreen and a consistent gentle routine "
                "can support a more even-looking skin tone."
            ),
        })

    # --------------------------------------------------
    # REDNESS
    # --------------------------------------------------

    if _has_issue(issues, "redness"):

        recommendations.append({
            "category": "Redness",
            "title": "Use gentle products",
            "description": (
                "Prefer fragrance-free, gentle skincare and avoid "
                "products that cause burning or irritation."
            ),
        })

    return recommendations

# ============================================================
# PRODUCT RECOMMENDATIONS
# ============================================================

def recommend_products(skin_type, issues):
    """
    Returns general skincare product-category recommendations.

    These are cosmetic product categories, not medical prescriptions.
    """

    products = []

    issues_text = " ".join(
        str(issue).lower()
        for issue in issues
    )

    # --------------------------------------------------------
    # CLEANSER
    # --------------------------------------------------------

    if skin_type == "Oily":
        products.append({
            "category": "Cleanser",
            "recommendation": "Gentle foaming or gel cleanser",
            "reason": "Suitable for removing excess surface oil."
        })

    elif skin_type == "Dry":
        products.append({
            "category": "Cleanser",
            "recommendation": "Gentle hydrating cleanser",
            "reason": "Helps cleanse without excessive dryness."
        })

    else:
        products.append({
            "category": "Cleanser",
            "recommendation": "Gentle low-irritation cleanser",
            "reason": "Suitable for maintaining a simple daily routine."
        })

    # --------------------------------------------------------
    # MOISTURIZER
    # --------------------------------------------------------

    if skin_type == "Dry":
        moisturizer = "Barrier-supporting moisturizer"
    else:
        moisturizer = "Lightweight non-comedogenic moisturizer"

    products.append({
        "category": "Moisturizer",
        "recommendation": moisturizer,
        "reason": "Helps maintain the skin barrier."
    })

    # --------------------------------------------------------
    # SUNSCREEN
    # --------------------------------------------------------

    products.append({
        "category": "Sunscreen",
        "recommendation": "Broad-spectrum SPF 30+ sunscreen",
        "reason": "Helps protect skin from UV exposure."
    })

    # --------------------------------------------------------
    # PIGMENTATION
    # --------------------------------------------------------

    if (
        "pigmentation" in issues_text
        or "dark spot" in issues_text
        or "uneven skin tone" in issues_text
    ):
        products.append({
            "category": "Targeted Care",
            "recommendation": "Niacinamide-based serum",
            "reason": "Can support a more even-looking skin tone."
        })

    # --------------------------------------------------------
    # CLOGGED PORES / ACNE-RELATED CONCERNS
    # --------------------------------------------------------

    if (
        "blackhead" in issues_text
        or "whitehead" in issues_text
        or "papule" in issues_text
        or "pustule" in issues_text
    ):
        products.append({
            "category": "Targeted Care",
            "recommendation": "Salicylic-acid product",
            "reason": "Can help with clogged pores when tolerated."
        })

    # --------------------------------------------------------
    # REDNESS
    # --------------------------------------------------------

    if "redness" in issues_text:
        products.append({
            "category": "Sensitive Skin Support",
            "recommendation": "Fragrance-free gentle moisturizer",
            "reason": "Helps minimize unnecessary irritation."
        })

    return products


# ============================================================
# ROUTINE BUILDER
# ============================================================

def build_routine(skin_type, issues):
    """
    Builds a simple morning and evening skincare routine.
    """

    issues_text = " ".join(
        str(issue).lower()
        for issue in issues
    )

    morning = [
        "Gentle cleanser",
        "Lightweight moisturizer",
        "Broad-spectrum SPF 30+ sunscreen"
    ]

    evening = [
        "Gentle cleanser",
        "Moisturizer"
    ]

    # --------------------------------------------------------
    # PIGMENTATION
    # --------------------------------------------------------

    if (
        "pigmentation" in issues_text
        or "dark spot" in issues_text
        or "uneven skin tone" in issues_text
    ):
        morning.insert(
            1,
            "Optional niacinamide serum"
        )

    # --------------------------------------------------------
    # CLOGGED PORES / ACNE
    # --------------------------------------------------------

    if (
        "blackhead" in issues_text
        or "whitehead" in issues_text
        or "papule" in issues_text
        or "pustule" in issues_text
    ):
        evening.insert(
            1,
            "Optional salicylic-acid treatment"
        )

    # --------------------------------------------------------
    # DRY SKIN
    # --------------------------------------------------------

    if skin_type == "Dry":
        morning[1] = "Barrier-supporting moisturizer"
        evening[-1] = "Rich barrier-supporting moisturizer"

    return {
        "morning": morning,
        "evening": evening
    }

def build_full_recommendation(
    skin_type,
    issues,
    hydration,
    oiliness
):
    """
    Builds the complete recommendation response
    used by the DermaNova skin analysis screen.
    """

    routine = build_routine(
        skin_type,
        issues
    )

    products = recommend_products(
        skin_type,
        issues
    )

    issue_text = " ".join(
        str(issue).lower()
        for issue in issues
    )

    focus = []
    notes = []

    # --------------------------------------------------------
    # SKIN CONCERNS
    # --------------------------------------------------------

    if (
        "dark spot" in issue_text
        or "pigmentation" in issue_text
        or "uneven skin tone" in issue_text
    ):
        focus.append("Pigmentation and uneven skin tone")

    if (
        "blackhead" in issue_text
        or "whitehead" in issue_text
        or "papule" in issue_text
        or "pustule" in issue_text
    ):
        focus.append("Clogged pores and acne-related concerns")

    if "nodule" in issue_text:
        focus.append("Deep or persistent acne-like lesions")

    if "redness" in issue_text:
        focus.append("Skin sensitivity and redness")

    if oiliness >= 70:
        focus.append("Excess surface oil")

    if hydration < 45:
        focus.append("Skin hydration")

    # --------------------------------------------------------
    # IMPORTANT NOTES
    # --------------------------------------------------------

    if (
        "blackhead" in issue_text
        or "whitehead" in issue_text
        or "papule" in issue_text
        or "pustule" in issue_text
        or "nodule" in issue_text
    ):
        notes.append(
            "Avoid picking or squeezing detected areas."
        )

    if (
        "dark spot" in issue_text
        or "pigmentation" in issue_text
    ):
        notes.append(
            "Consistent sun protection is important when "
            "addressing visible pigmentation."
        )

    if "nodule" in issue_text:
        notes.append(
            "Persistent, painful, or deep nodules should be "
            "evaluated by a dermatologist."
        )

    notes.append(
        "Introduce new active skincare products gradually "
        "and stop if significant irritation occurs."
    )

    return {
        "routine": routine,
        "products": products,
        "focus": focus,
        "notes": notes
    }