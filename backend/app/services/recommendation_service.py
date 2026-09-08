"""
DermaNova AI Recommendation Engine

Generates personalized cosmetic skincare guidance from
the visual skin assessment and detected concerns.

This system provides informational skincare guidance.
It is NOT a medical diagnosis or prescription.
"""


# ============================================================
# HELPERS
# ============================================================

def _has_issue(issues, keyword):
    return any(
        keyword.lower() in str(issue).lower()
        for issue in issues
    )


def _issue_text(issues):
    return " ".join(
        str(issue).lower()
        for issue in issues
    )


# ============================================================
# PERSONALIZED INSIGHTS
# ============================================================

def generate_recommendations(
    skin_type,
    issues,
    hydration,
    oiliness,
):
    recommendations = []

    # --------------------------------------------------------
    # SKIN TYPE
    # --------------------------------------------------------

    if skin_type == "Oily":
        recommendations.append({
            "category": "Skin Type",
            "title": "Balance excess oil",
            "description": (
                "Your assessment indicates an oilier skin profile. "
                "Focus on lightweight, non-comedogenic products "
                "without repeatedly stripping the skin."
            ),
        })

    elif skin_type == "Dry":
        recommendations.append({
            "category": "Skin Type",
            "title": "Strengthen the skin barrier",
            "description": (
                "Your assessment indicates a drier skin profile. "
                "Prioritize gentle cleansing and consistent "
                "moisturization to support the skin barrier."
            ),
        })

    elif skin_type == "Combination":
        recommendations.append({
            "category": "Skin Type",
            "title": "Balance different areas",
            "description": (
                "Your assessment indicates combination characteristics. "
                "Use lightweight hydration while avoiding excessive "
                "drying of oilier areas."
            ),
        })

    else:
        recommendations.append({
            "category": "Skin Type",
            "title": "Maintain your skin balance",
            "description": (
                "Your assessment indicates a relatively balanced "
                "skin profile. A simple, consistent routine can "
                "help maintain it."
            ),
        })

    # --------------------------------------------------------
    # HYDRATION
    # --------------------------------------------------------

    if hydration < 45:
        recommendations.append({
            "category": "Hydration",
            "title": "Prioritize hydration",
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

    # --------------------------------------------------------
    # OILINESS
    # --------------------------------------------------------

    if oiliness >= 70:
        recommendations.append({
            "category": "Oil Control",
            "title": "Manage surface oil",
            "description": (
                "Your visual oiliness indicator is relatively high. "
                "Lightweight, non-comedogenic skincare may help "
                "maintain a comfortable skin balance."
            ),
        })

    # --------------------------------------------------------
    # ACNE / CLOGGED PORES
    # --------------------------------------------------------

    if (
        _has_issue(issues, "papule")
        or _has_issue(issues, "pustule")
        or _has_issue(issues, "blackhead")
        or _has_issue(issues, "whitehead")
        or _has_issue(issues, "acne")
    ):
        recommendations.append({
            "category": "Acne Care",
            "title": "Keep congestion under control",
            "description": (
                "Detected acne-like or clogged-pore concerns suggest "
                "keeping the routine gentle and avoiding picking, "
                "squeezing, or aggressive exfoliation."
            ),
        })

    # --------------------------------------------------------
    # NODULES
    # --------------------------------------------------------

    if _has_issue(issues, "nodule"):
        recommendations.append({
            "category": "Professional Care",
            "title": "Consider professional evaluation",
            "description": (
                "Deep, persistent, or painful nodules should be "
                "evaluated by a qualified dermatologist rather "
                "than managed only with cosmetic products."
            ),
        })

    # --------------------------------------------------------
    # PIGMENTATION
    # --------------------------------------------------------

    if (
        _has_issue(issues, "dark spot")
        or _has_issue(issues, "pigmentation")
        or _has_issue(issues, "uneven skin tone")
    ):
        recommendations.append({
            "category": "Pigmentation",
            "title": "Protect and even-looking skin tone",
            "description": (
                "Consistent sun protection and a gentle routine "
                "can help support a more even-looking complexion "
                "and reduce additional visible pigmentation."
            ),
        })

    # --------------------------------------------------------
    # REDNESS
    # --------------------------------------------------------

    if _has_issue(issues, "redness"):
        recommendations.append({
            "category": "Sensitivity",
            "title": "Minimize irritation",
            "description": (
                "Choose gentle, fragrance-free products where "
                "possible and avoid products that cause burning, "
                "stinging, or persistent irritation."
            ),
        })

    return recommendations


# ============================================================
# PRODUCT CATEGORY ENGINE
# ============================================================

def recommend_products(skin_type, issues):
    """
    Returns cosmetic product-category recommendations.

    These are NOT medical prescriptions.
    """

    products = []
    issues_text = _issue_text(issues)

    # --------------------------------------------------------
    # CLEANSER
    # --------------------------------------------------------

    if skin_type == "Oily":
        products.append({
            "category": "Cleanser",
            "recommendation": "Gentle foaming or gel cleanser",
            "reason": "Helps remove excess surface oil without aggressive scrubbing.",
            "priority": "High",
        })

    elif skin_type == "Dry":
        products.append({
            "category": "Cleanser",
            "recommendation": "Gentle hydrating cleanser",
            "reason": "Cleanses while minimizing unnecessary dryness.",
            "priority": "High",
        })

    elif skin_type == "Combination":
        products.append({
            "category": "Cleanser",
            "recommendation": "Gentle balancing cleanser",
            "reason": "Supports cleansing without excessively drying different areas.",
            "priority": "High",
        })

    else:
        products.append({
            "category": "Cleanser",
            "recommendation": "Gentle low-irritation cleanser",
            "reason": "Suitable for maintaining a simple daily routine.",
            "priority": "High",
        })

    # --------------------------------------------------------
    # MOISTURIZER
    # --------------------------------------------------------

    if skin_type == "Dry":
        moisturizer = "Barrier-supporting moisturizer"
        moisturizer_reason = (
            "Supports hydration and helps maintain the skin barrier."
        )

    elif skin_type == "Oily":
        moisturizer = "Lightweight non-comedogenic moisturizer"
        moisturizer_reason = (
            "Provides hydration without relying on a heavy texture."
        )

    else:
        moisturizer = "Lightweight daily moisturizer"
        moisturizer_reason = (
            "Helps maintain comfortable and consistent hydration."
        )

    products.append({
        "category": "Moisturizer",
        "recommendation": moisturizer,
        "reason": moisturizer_reason,
        "priority": "High",
    })

    # --------------------------------------------------------
    # SUNSCREEN
    # --------------------------------------------------------

    products.append({
        "category": "Sun Protection",
        "recommendation": "Broad-spectrum SPF 30+ sunscreen",
        "reason": (
            "Helps protect skin from UV exposure and is especially "
            "important when visible pigmentation or dark spots are present."
        ),
        "priority": "Essential",
    })

    # --------------------------------------------------------
    # NIACINAMIDE
    # --------------------------------------------------------

    if (
        "pigmentation" in issues_text
        or "dark spot" in issues_text
        or "uneven skin tone" in issues_text
        or skin_type == "Oily"
    ):
        products.append({
            "category": "Targeted Care",
            "recommendation": "Niacinamide-based serum",
            "reason": (
                "Can support an even-looking skin tone and may be "
                "useful for routines focused on visible oiliness."
            ),
            "priority": "Targeted",
        })

    # --------------------------------------------------------
    # SALICYLIC ACID
    # --------------------------------------------------------

    if (
        "blackhead" in issues_text
        or "whitehead" in issues_text
        or "papule" in issues_text
        or "pustule" in issues_text
    ):
        products.append({
            "category": "Pore Care",
            "recommendation": "Salicylic-acid product",
            "reason": (
                "Can help manage clogged pores when tolerated "
                "and used according to product directions."
            ),
            "priority": "Targeted",
        })

    # --------------------------------------------------------
    # REDNESS SUPPORT
    # --------------------------------------------------------

    if "redness" in issues_text:
        products.append({
            "category": "Sensitive Skin Support",
            "recommendation": "Fragrance-free gentle moisturizer",
            "reason": (
                "Helps keep the routine simple and reduce unnecessary "
                "irritation."
            ),
            "priority": "Targeted",
        })

    return products


# ============================================================
# ROUTINE BUILDER
# ============================================================

def build_routine(skin_type, issues):
    """
    Builds a personalized morning and evening routine.
    """

    issues_text = _issue_text(issues)

    morning = [
        {
            "step": 1,
            "product": "Gentle cleanser",
            "purpose": "Cleanse the skin without unnecessary irritation."
        },
        {
            "step": 2,
            "product": "Moisturizer",
            "purpose": "Maintain comfortable hydration and support the skin barrier."
        },
        {
            "step": 3,
            "product": "Broad-spectrum SPF 30+ sunscreen",
            "purpose": "Protect the skin from daily UV exposure."
        }
    ]

    evening = [
        {
            "step": 1,
            "product": "Gentle cleanser",
            "purpose": "Remove daily buildup and prepare the skin for nighttime care."
        },
        {
            "step": 2,
            "product": "Moisturizer",
            "purpose": "Support overnight hydration and the skin barrier."
        }
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
            {
                "step": 2,
                "product": "Optional niacinamide serum",
                "purpose": "Support a more even-looking skin tone."
            }
        )

        for index, item in enumerate(morning, start=1):
            item["step"] = index

    # --------------------------------------------------------
    # CLOGGED PORES
    # --------------------------------------------------------

    if (
        "blackhead" in issues_text
        or "whitehead" in issues_text
        or "papule" in issues_text
        or "pustule" in issues_text
    ):
        evening.insert(
            1,
            {
                "step": 2,
                "product": "Optional salicylic-acid treatment",
                "purpose": "Support management of clogged pores when tolerated."
            }
        )

        for index, item in enumerate(evening, start=1):
            item["step"] = index

    # --------------------------------------------------------
    # DRY SKIN
    # --------------------------------------------------------

    if skin_type == "Dry":

        morning[1 if len(morning) > 3 else 1] = {
            "step": 2,
            "product": "Barrier-supporting moisturizer",
            "purpose": "Provide additional hydration and barrier support."
        }

        for index, item in enumerate(morning, start=1):
            item["step"] = index

        evening[-1] = {
            "step": len(evening),
            "product": "Rich barrier-supporting moisturizer",
            "purpose": "Support overnight hydration."
        }

    return {
        "morning": morning,
        "evening": evening
    }


# ============================================================
# FOCUS AREA ENGINE
# ============================================================

def build_focus_areas(
    skin_type,
    issues,
    hydration,
    oiliness
):
    focus = []

    issues_text = _issue_text(issues)

    if (
        "dark spot" in issues_text
        or "pigmentation" in issues_text
        or "uneven skin tone" in issues_text
    ):
        focus.append({
            "title": "Pigmentation",
            "priority": "High",
            "description": "Support an even-looking complexion and protect against further visible pigmentation."
        })

    if (
        "blackhead" in issues_text
        or "whitehead" in issues_text
        or "papule" in issues_text
        or "pustule" in issues_text
    ):
        focus.append({
            "title": "Clogged pores",
            "priority": "High",
            "description": "Keep the routine gentle while supporting clearer-looking pores."
        })

    if "nodule" in issues_text:
        focus.append({
            "title": "Deep lesions",
            "priority": "High",
            "description": "Persistent or painful deep lesions deserve professional assessment."
        })

    if "redness" in issues_text:
        focus.append({
            "title": "Skin comfort",
            "priority": "Medium",
            "description": "Minimize potential irritation and keep the routine gentle."
        })

    if oiliness >= 70:
        focus.append({
            "title": "Oil balance",
            "priority": "Medium",
            "description": "Use lightweight products while avoiding excessive stripping."
        })

    if hydration < 45:
        focus.append({
            "title": "Hydration",
            "priority": "High",
            "description": "Increase hydration support with consistent moisturizing."
        })

    if not focus:
        focus.append({
            "title": "Skin maintenance",
            "priority": "Medium",
            "description": "Maintain a consistent, gentle skincare routine."
        })

    return focus


# ============================================================
# AVOID LIST
# ============================================================

def build_avoid_list(skin_type, issues):
    avoid = []

    issues_text = _issue_text(issues)

    avoid.append("Aggressive scrubbing")

    if (
        "blackhead" in issues_text
        or "whitehead" in issues_text
        or "papule" in issues_text
        or "pustule" in issues_text
        or "nodule" in issues_text
    ):
        avoid.append("Picking or squeezing detected areas")

    if "redness" in issues_text:
        avoid.append("Products that cause burning or persistent irritation")

    if skin_type == "Oily":
        avoid.append("Repeatedly stripping the skin to remove oil")

    if skin_type == "Dry":
        avoid.append("Excessive use of drying cleansers or treatments")

    return avoid


# ============================================================
# IMPORTANT NOTES
# ============================================================

def build_notes(issues):
    notes = []

    issues_text = _issue_text(issues)

    if (
        "blackhead" in issues_text
        or "whitehead" in issues_text
        or "papule" in issues_text
        or "pustule" in issues_text
        or "nodule" in issues_text
    ):
        notes.append(
            "Avoid picking or squeezing detected areas."
        )

    if (
        "dark spot" in issues_text
        or "pigmentation" in issues_text
    ):
        notes.append(
            "Consistent sun protection is especially important "
            "when visible pigmentation is present."
        )

    if "nodule" in issues_text:
        notes.append(
            "Persistent, painful, or deep nodules should be "
            "evaluated by a dermatologist."
        )

    notes.append(
        "Introduce new active skincare products gradually "
        "and stop if significant irritation occurs."
    )

    notes.append(
        "DermaNova's visual analysis is informational and "
        "should not be treated as a medical diagnosis."
    )

    return notes


# ============================================================
# MASTER RECOMMENDATION
# ============================================================

def build_full_recommendation(
    skin_type,
    issues,
    hydration,
    oiliness
):
    """
    Builds the complete personalized skincare response.
    """

    routine = build_routine(
        skin_type,
        issues
    )

    products = recommend_products(
        skin_type,
        issues
    )

    focus = build_focus_areas(
        skin_type,
        issues,
        hydration,
        oiliness
    )

    avoid = build_avoid_list(
        skin_type,
        issues
    )

    notes = build_notes(issues)

    # --------------------------------------------------------
    # PERSONALIZED SUMMARY
    # --------------------------------------------------------

    if skin_type == "Oily":
        profile = "Your skin profile appears oil-prone."
    elif skin_type == "Dry":
        profile = "Your skin profile appears hydration-focused."
    elif skin_type == "Combination":
        profile = "Your skin profile shows combination characteristics."
    else:
        profile = "Your skin profile appears relatively balanced."

    if focus:
        primary_focus = focus[0]["title"]
        summary = (
            f"{profile} Your current routine should primarily "
            f"focus on {primary_focus.lower()} while maintaining "
            f"gentle cleansing, hydration, and daily sun protection."
        )
    else:
        summary = (
            f"{profile} Maintain a simple, consistent routine "
            "with gentle cleansing, hydration, and daily sun protection."
        )

    return {
        "profile": {
            "skin_type": skin_type,
            "summary": summary
        },

        "routine": routine,

        "products": products,

        "focus": focus,

        "avoid": avoid,

        "notes": notes
    }