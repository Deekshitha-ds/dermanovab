"""
DermaNova AI Recommendation Engine

Supports two recommendation flows:

1. AI skin-analysis flow
   - generate_recommendations()
   - build_full_recommendation()

2. Database product recommendation flow
   - recommend_products()
   - build_routine()

This system provides informational cosmetic skincare guidance.
It is NOT a medical diagnosis or prescription.
"""

from sqlalchemy.orm import Session

from app.models.orm_models import Product


# ============================================================
# GENERAL HELPERS
# ============================================================

def _normalise(value):
    if value is None:
        return ""

    return str(value).strip().lower()


def _normalise_list(values):
    if not values:
        return []

    return [
        _normalise(value)
        for value in values
        if value is not None and str(value).strip()
    ]


def _has_issue(issues, keyword):
    keyword = _normalise(keyword)

    return any(
        keyword in _normalise(issue)
        for issue in (issues or [])
    )


def _issue_text(issues):
    return " ".join(
        _normalise(issue)
        for issue in (issues or [])
    )


def _product_ingredients(product):
    ingredients = product.ingredients or []

    if isinstance(ingredients, list):
        return [_normalise(item) for item in ingredients]

    if isinstance(ingredients, str):
        return [
            _normalise(item)
            for item in ingredients.split(",")
        ]

    return []


# ============================================================
# AI SKIN ANALYSIS RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    skin_type,
    issues,
    hydration,
    oiliness,
):
    """
    Generates personalized cosmetic guidance from the AI scan.

    This function is used by ai_service.py.
    """

    recommendations = []

    issues = issues or []

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
# DATABASE PRODUCT HELPERS
# ============================================================

def _matches_skin_type(product, skin_type):

    requested = _normalise(skin_type)
    product_type = _normalise(product.skin_type)

    if not requested:
        return True

    if not product_type:
        return True

    if product_type in (
        "all",
        "all skin types",
        "any",
    ):
        return True

    return requested in product_type


def _matches_hair_type(product, hair_type):

    requested = _normalise(hair_type)
    product_type = _normalise(product.hair_type)

    if not requested:
        return True

    if not product_type:
        return True

    if product_type in (
        "all",
        "all hair types",
        "any",
    ):
        return True

    return requested in product_type


def _matches_concern(product, concerns):

    if not concerns:
        return True

    product_concern = _normalise(product.concern)

    if not product_concern:
        return True

    for concern in concerns:

        concern_text = _normalise(concern)

        if (
            concern_text in product_concern
            or product_concern in concern_text
        ):
            return True

    return False


def _is_sensitive_friendly(product):

    if product.fragrance_free:
        return True

    ingredients = _product_ingredients(product)

    irritating_keywords = [
        "fragrance",
        "parfum",
        "essential oil",
        "denatured alcohol",
    ]

    for ingredient in ingredients:

        if any(
            keyword in ingredient
            for keyword in irritating_keywords
        ):
            return False

    return True


def _concern_matches_ingredient(
    product,
    concerns,
):
    """
    Gives additional ranking points when ingredients
    are relevant to detected concerns.
    """

    if not concerns:
        return 0

    ingredients = _product_ingredients(product)

    if not ingredients:
        return 0

    ingredient_text = " ".join(ingredients)

    score = 0

    for concern in concerns:

        concern = _normalise(concern)

        # ----------------------------------------------------
        # ACNE / CLOGGED PORES
        # ----------------------------------------------------

        if any(
            keyword in concern
            for keyword in [
                "blackhead",
                "whitehead",
                "papule",
                "pustule",
                "acne",
                "clogged pore",
                "clogged pores",
            ]
        ):

            if (
                "salicylic acid" in ingredient_text
                or "bha" in ingredient_text
            ):
                score += 12

        # ----------------------------------------------------
        # PIGMENTATION
        # ----------------------------------------------------

        if any(
            keyword in concern
            for keyword in [
                "pigmentation",
                "dark spot",
                "dark spots",
                "uneven skin tone",
            ]
        ):

            if (
                "niacinamide" in ingredient_text
                or "azelaic acid" in ingredient_text
                or "vitamin c" in ingredient_text
            ):
                score += 12

        # ----------------------------------------------------
        # HYDRATION
        # ----------------------------------------------------

        if any(
            keyword in concern
            for keyword in [
                "dry",
                "dehydration",
                "hydration",
            ]
        ):

            if (
                "hyaluronic acid" in ingredient_text
                or "glycerin" in ingredient_text
                or "ceramide" in ingredient_text
                or "squalane" in ingredient_text
            ):
                score += 10

        # ----------------------------------------------------
        # REDNESS / SENSITIVITY
        # ----------------------------------------------------

        if any(
            keyword in concern
            for keyword in [
                "redness",
                "sensitive",
                "irritation",
            ]
        ):

            if (
                "ceramide" in ingredient_text
                or "panthenol" in ingredient_text
                or "centella" in ingredient_text
                or "aloe" in ingredient_text
            ):
                score += 10

    return score


def _category_score(product):

    category = _normalise(product.category)

    preferred_categories = {
        "cleanser": 8,
        "face wash": 8,
        "facewash": 8,
        "moisturizer": 8,
        "moisturiser": 8,
        "sunscreen": 10,
        "sun protection": 10,
        "serum": 5,
        "treatment": 5,
    }

    return preferred_categories.get(
        category,
        0
    )


def _budget_score(product, budget):

    if budget is None or budget <= 0:
        return 0

    price = float(product.price or 0)

    if price <= 0:
        return 0

    if price <= budget * 0.15:
        return 10

    if price <= budget * 0.25:
        return 8

    if price <= budget * 0.35:
        return 6

    if price <= budget * 0.50:
        return 3

    return 0


def _rating_score(product):

    rating = float(product.rating or 0)

    if rating >= 4.7:
        return 10

    if rating >= 4.5:
        return 8

    if rating >= 4.2:
        return 6

    if rating >= 4.0:
        return 4

    return 0


# ============================================================
# DATABASE PRODUCT SCORING
# ============================================================

def _score_product(
    product,
    budget,
    skin_type,
    hair_type,
    concerns,
    dermatologist_only,
    sensitive,
):

    score = 0

    # --------------------------------------------------------
    # SKIN TYPE
    # --------------------------------------------------------

    requested_skin = _normalise(skin_type)
    product_skin = _normalise(product.skin_type)

    if requested_skin and product_skin:

        if product_skin in (
            "all",
            "all skin types",
            "any",
        ):
            score += 15

        elif requested_skin in product_skin:
            score += 25

    elif requested_skin and not product_skin:

        score += 5

    # --------------------------------------------------------
    # HAIR TYPE
    # --------------------------------------------------------

    requested_hair = _normalise(hair_type)
    product_hair = _normalise(product.hair_type)

    if requested_hair and product_hair:

        if product_hair in (
            "all",
            "all hair types",
            "any",
        ):
            score += 15

        elif requested_hair in product_hair:
            score += 25

    elif requested_hair and not product_hair:

        score += 5

    # --------------------------------------------------------
    # CONCERNS
    # --------------------------------------------------------

    product_concern = _normalise(product.concern)

    if concerns:

        if product_concern:

            for concern in concerns:

                concern_text = _normalise(concern)

                if (
                    concern_text in product_concern
                    or product_concern in concern_text
                ):
                    score += 30
                    break

        score += _concern_matches_ingredient(
            product,
            concerns
        )

    else:

        score += 5

    # --------------------------------------------------------
    # DERMATOLOGIST TESTED
    # --------------------------------------------------------

    if dermatologist_only:

        if product.dermatologist_tested:
            score += 25
        else:
            score -= 100

    elif product.dermatologist_tested:

        score += 5

    # --------------------------------------------------------
    # SENSITIVE SKIN
    # --------------------------------------------------------

    if sensitive:

        if product.fragrance_free:
            score += 20

        if product.dermatologist_tested:
            score += 5

        if product.paraben_free:
            score += 3

        if not _is_sensitive_friendly(product):
            score -= 25

    # --------------------------------------------------------
    # CRUELTY FREE / VEGAN
    # --------------------------------------------------------

    if product.cruelty_free:
        score += 2

    if product.vegan:
        score += 2

    # --------------------------------------------------------
    # RATING
    # --------------------------------------------------------

    score += _rating_score(product)

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    score += _category_score(product)

    # --------------------------------------------------------
    # BUDGET
    # --------------------------------------------------------

    score += _budget_score(
        product,
        budget
    )

    return score


# ============================================================
# DATABASE PRODUCT RECOMMENDATION
# ============================================================

def recommend_products(
    db: Session,
    budget: float,
    skin_type=None,
    hair_type=None,
    concerns=None,
    dermatologist_only=False,
    sensitive=False,
    weather_condition=None,
):
    """
    Database-backed product recommendation engine.

    Matches recommendations_router.py exactly.
    """

    concerns = _normalise_list(
        concerns or []
    )

    # --------------------------------------------------------
    # QUERY
    # --------------------------------------------------------

    query = db.query(Product)

    if dermatologist_only:

        query = query.filter(
            Product.dermatologist_tested == True
        )

    products = query.all()

    if not products:
        return []

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    scored_products = []

    for product in products:

        score = _score_product(
            product=product,
            budget=budget,
            skin_type=skin_type,
            hair_type=hair_type,
            concerns=concerns,
            dermatologist_only=dermatologist_only,
            sensitive=sensitive,
        )

        scored_products.append(
            (product, score)
        )

    # --------------------------------------------------------
    # WEATHER
    # --------------------------------------------------------

    weather = _normalise(
        weather_condition
    )

    if weather:

        adjusted_products = []

        for product, score in scored_products:

            weather_bonus = 0

            category = _normalise(
                product.category
            )

            name = _normalise(
                product.name
            )

            description = _normalise(
                product.description
            )

            ingredients = " ".join(
                _product_ingredients(product)
            )

            # ------------------------------------------------
            # HOT / HUMID
            # ------------------------------------------------

            if weather in (
                "hot",
                "humid",
                "summer",
            ):

                if (
                    "gel" in name
                    or "gel" in category
                    or "lightweight" in description
                ):
                    weather_bonus += 5

                if (
                    "niacinamide" in ingredients
                    or "non-comedogenic" in ingredients
                ):
                    weather_bonus += 3

            # ------------------------------------------------
            # DRY / COLD
            # ------------------------------------------------

            elif weather in (
                "dry",
                "cold",
                "winter",
            ):

                if (
                    "ceramide" in ingredients
                    or "hyaluronic acid" in ingredients
                    or "glycerin" in ingredients
                ):
                    weather_bonus += 8

            adjusted_products.append(
                (
                    product,
                    score + weather_bonus
                )
            )

        scored_products = adjusted_products

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    scored_products.sort(
        key=lambda item: (
            item[1],
            float(item[0].rating or 0),
            -float(item[0].price or 0),
        ),
        reverse=True,
    )

    return [
        product
        for product, score in scored_products
    ]


# ============================================================
# DATABASE ROUTINE BUILDER
# ============================================================

def build_routine(products):
    """
    Builds morning/night routines from ranked database products.

    Matches recommendations_router.py exactly.
    """

    if not products:

        return {
            "morning": [],
            "night": [],
            "morning_total": 0,
            "night_total": 0,
            "best_choice": None,
            "budget_choice": None,
            "premium_choice": None,
        }

    # --------------------------------------------------------
    # FIND CATEGORY
    # --------------------------------------------------------

    def find_category(*keywords):

        for product in products:

            category = _normalise(
                product.category
            )

            name = _normalise(
                product.name
            )

            combined = (
                f"{category} {name}"
            )

            if any(
                keyword in combined
                for keyword in keywords
            ):
                return product

        return None

    # --------------------------------------------------------
    # CORE PRODUCTS
    # --------------------------------------------------------

    cleanser = find_category(
        "cleanser",
        "face wash",
        "facewash",
    )

    moisturizer = find_category(
        "moisturizer",
        "moisturiser",
    )

    sunscreen = find_category(
        "sunscreen",
        "sun protection",
        "spf",
    )

    serum = find_category(
        "serum",
        "treatment",
    )

    # --------------------------------------------------------
    # MORNING
    # --------------------------------------------------------

    morning = []

    if cleanser:
        morning.append(cleanser)

    if serum and serum not in morning:
        morning.append(serum)

    if moisturizer and moisturizer not in morning:
        morning.append(moisturizer)

    if sunscreen and sunscreen not in morning:
        morning.append(sunscreen)

    # --------------------------------------------------------
    # NIGHT
    # --------------------------------------------------------

    night = []

    if cleanser:
        night.append(cleanser)

    if serum and serum not in night:
        night.append(serum)

    if moisturizer and moisturizer not in night:
        night.append(moisturizer)

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if not morning:

        morning = products[
            :min(3, len(products))
        ]

    if not night:

        night = products[
            :min(3, len(products))
        ]

    # --------------------------------------------------------
    # TOTALS
    # --------------------------------------------------------

    morning_total = round(
        sum(
            float(product.price or 0)
            for product in morning
        ),
        2,
    )

    night_total = round(
        sum(
            float(product.price or 0)
            for product in night
        ),
        2,
    )

    # --------------------------------------------------------
    # BEST CHOICE
    # --------------------------------------------------------

    best_choice = products[0]

    # --------------------------------------------------------
    # BUDGET CHOICE
    # --------------------------------------------------------

    budget_choice = min(
        products,
        key=lambda product:
        float(product.price or 0),
    )

    # --------------------------------------------------------
    # PREMIUM CHOICE
    # --------------------------------------------------------

    premium_choice = max(
        products,
        key=lambda product: (
            float(product.rating or 0),
            float(product.price or 0),
        ),
    )

    return {
        "morning": morning,
        "night": night,
        "morning_total": morning_total,
        "night_total": night_total,
        "best_choice": best_choice,
        "budget_choice": budget_choice,
        "premium_choice": premium_choice,
    }


# ============================================================
# STRUCTURED AI RECOMMENDATION HELPERS
# ============================================================

def _build_ai_products(
    skin_type,
    issues,
):
    """
    Category-level products for the ScanResult page.

    This is intentionally separate from the DB product engine.
    """

    products = []

    issues_text = _issue_text(
        issues
    )

    # --------------------------------------------------------
    # CLEANSER
    # --------------------------------------------------------

    if skin_type == "Oily":

        products.append({
            "category": "Cleanser",
            "recommendation": (
                "Gentle foaming or gel cleanser"
            ),
            "reason": (
                "Helps remove excess surface oil "
                "without aggressive scrubbing."
            ),
            "priority": "High",
        })

    elif skin_type == "Dry":

        products.append({
            "category": "Cleanser",
            "recommendation": (
                "Gentle hydrating cleanser"
            ),
            "reason": (
                "Cleanses while minimizing "
                "unnecessary dryness."
            ),
            "priority": "High",
        })

    elif skin_type == "Combination":

        products.append({
            "category": "Cleanser",
            "recommendation": (
                "Gentle balancing cleanser"
            ),
            "reason": (
                "Supports cleansing without "
                "excessively drying different areas."
            ),
            "priority": "High",
        })

    else:

        products.append({
            "category": "Cleanser",
            "recommendation": (
                "Gentle low-irritation cleanser"
            ),
            "reason": (
                "Suitable for maintaining "
                "a simple daily routine."
            ),
            "priority": "High",
        })

    # --------------------------------------------------------
    # MOISTURIZER
    # --------------------------------------------------------

    if skin_type == "Dry":

        moisturizer = (
            "Barrier-supporting moisturizer"
        )

        moisturizer_reason = (
            "Supports hydration and helps "
            "maintain the skin barrier."
        )

    elif skin_type == "Oily":

        moisturizer = (
            "Lightweight non-comedogenic moisturizer"
        )

        moisturizer_reason = (
            "Provides hydration without "
            "relying on a heavy texture."
        )

    else:

        moisturizer = (
            "Lightweight daily moisturizer"
        )

        moisturizer_reason = (
            "Helps maintain comfortable "
            "and consistent hydration."
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
        "recommendation": (
            "Broad-spectrum SPF 30+ sunscreen"
        ),
        "reason": (
            "Helps protect skin from UV exposure "
            "and is especially important when "
            "visible pigmentation or dark spots "
            "are present."
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
            "recommendation": (
                "Niacinamide-based serum"
            ),
            "reason": (
                "Can support an even-looking "
                "skin tone and may be useful "
                "for routines focused on "
                "visible oiliness."
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
            "recommendation": (
                "Salicylic-acid product"
            ),
            "reason": (
                "Can help manage clogged pores "
                "when tolerated and used "
                "according to product directions."
            ),
            "priority": "Targeted",
        })

    # --------------------------------------------------------
    # REDNESS
    # --------------------------------------------------------

    if "redness" in issues_text:

        products.append({
            "category": "Sensitive Skin Support",
            "recommendation": (
                "Fragrance-free gentle moisturizer"
            ),
            "reason": (
                "Helps keep the routine simple "
                "and reduce unnecessary irritation."
            ),
            "priority": "Targeted",
        })

    return products


# ============================================================
# AI ROUTINE
# ============================================================

def _build_ai_routine(
    skin_type,
    issues,
):
    """
    Routine used by the existing ScanResult UI.
    """

    issues_text = _issue_text(
        issues
    )

    morning = [
        {
            "step": 1,
            "product": "Gentle cleanser",
            "purpose": (
                "Cleanse the skin without "
                "unnecessary irritation."
            ),
        },
        {
            "step": 2,
            "product": "Moisturizer",
            "purpose": (
                "Maintain comfortable hydration "
                "and support the skin barrier."
            ),
        },
        {
            "step": 3,
            "product": (
                "Broad-spectrum SPF 30+ sunscreen"
            ),
            "purpose": (
                "Protect the skin from daily "
                "UV exposure."
            ),
        },
    ]

    evening = [
        {
            "step": 1,
            "product": "Gentle cleanser",
            "purpose": (
                "Remove daily buildup and prepare "
                "the skin for nighttime care."
            ),
        },
        {
            "step": 2,
            "product": "Moisturizer",
            "purpose": (
                "Support overnight hydration "
                "and the skin barrier."
            ),
        },
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
                "product": (
                    "Optional niacinamide serum"
                ),
                "purpose": (
                    "Support a more even-looking "
                    "skin tone."
                ),
            },
        )

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
                "product": (
                    "Optional salicylic-acid treatment"
                ),
                "purpose": (
                    "Support management of clogged "
                    "pores when tolerated."
                ),
            },
        )

    # --------------------------------------------------------
    # DRY SKIN
    # --------------------------------------------------------

    if skin_type == "Dry":

        for item in morning:

            if item["product"] == "Moisturizer":

                item["product"] = (
                    "Barrier-supporting moisturizer"
                )

                item["purpose"] = (
                    "Provide additional hydration "
                    "and barrier support."
                )

        evening[-1] = {
            "step": len(evening),
            "product": (
                "Rich barrier-supporting moisturizer"
            ),
            "purpose": (
                "Support overnight hydration."
            ),
        }

    # --------------------------------------------------------
    # RENUMBER
    # --------------------------------------------------------

    for index, item in enumerate(
        morning,
        start=1
    ):
        item["step"] = index

    for index, item in enumerate(
        evening,
        start=1
    ):
        item["step"] = index

    return {
        "morning": morning,
        "evening": evening,
    }


# ============================================================
# AI FOCUS AREAS
# ============================================================

def _build_focus_areas(
    skin_type,
    issues,
    hydration,
    oiliness,
):
    focus = []

    issues_text = _issue_text(
        issues
    )

    if (
        "dark spot" in issues_text
        or "pigmentation" in issues_text
        or "uneven skin tone" in issues_text
    ):

        focus.append({
            "title": "Pigmentation",
            "priority": "High",
            "description": (
                "Support an even-looking complexion "
                "and protect against further "
                "visible pigmentation."
            ),
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
            "description": (
                "Keep the routine gentle while "
                "supporting clearer-looking pores."
            ),
        })

    if "nodule" in issues_text:

        focus.append({
            "title": "Deep lesions",
            "priority": "High",
            "description": (
                "Persistent or painful deep lesions "
                "deserve professional assessment."
            ),
        })

    if "redness" in issues_text:

        focus.append({
            "title": "Skin comfort",
            "priority": "Medium",
            "description": (
                "Minimize potential irritation "
                "and keep the routine gentle."
            ),
        })

    if oiliness >= 70:

        focus.append({
            "title": "Oil balance",
            "priority": "Medium",
            "description": (
                "Use lightweight products while "
                "avoiding excessive stripping."
            ),
        })

    if hydration < 45:

        focus.append({
            "title": "Hydration",
            "priority": "High",
            "description": (
                "Increase hydration support with "
                "consistent moisturizing."
            ),
        })

    if not focus:

        focus.append({
            "title": "Skin maintenance",
            "priority": "Medium",
            "description": (
                "Maintain a consistent, gentle "
                "skincare routine."
            ),
        })

    return focus


# ============================================================
# AI AVOID LIST
# ============================================================

def _build_avoid_list(
    skin_type,
    issues,
):
    avoid = []

    issues_text = _issue_text(
        issues
    )

    avoid.append(
        "Aggressive scrubbing"
    )

    if (
        "blackhead" in issues_text
        or "whitehead" in issues_text
        or "papule" in issues_text
        or "pustule" in issues_text
        or "nodule" in issues_text
    ):

        avoid.append(
            "Picking or squeezing detected areas"
        )

    if "redness" in issues_text:

        avoid.append(
            "Products that cause burning "
            "or persistent irritation"
        )

    if skin_type == "Oily":

        avoid.append(
            "Repeatedly stripping the skin "
            "to remove oil"
        )

    if skin_type == "Dry":

        avoid.append(
            "Excessive use of drying cleansers "
            "or treatments"
        )

    return avoid


# ============================================================
# AI NOTES
# ============================================================

def _build_notes(issues):

    notes = []

    issues_text = _issue_text(
        issues
    )

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
            "Consistent sun protection is especially "
            "important when visible pigmentation "
            "is present."
        )

    if "nodule" in issues_text:

        notes.append(
            "Persistent, painful, or deep nodules "
            "should be evaluated by a dermatologist."
        )

    notes.append(
        "Introduce new active skincare products "
        "gradually and stop if significant "
        "irritation occurs."
    )

    notes.append(
        "DermaNova's visual analysis is informational "
        "and should not be treated as a medical diagnosis."
    )

    return notes


# ============================================================
# MASTER AI RECOMMENDATION
# ============================================================

def build_full_recommendation(
    skin_type,
    issues,
    hydration,
    oiliness,
):
    """
    Builds the complete structured recommendation
    consumed by ai_service.py and ScanResult.jsx.
    """

    issues = issues or []

    routine = _build_ai_routine(
        skin_type,
        issues,
    )

    products = _build_ai_products(
        skin_type,
        issues,
    )

    focus = _build_focus_areas(
        skin_type,
        issues,
        hydration,
        oiliness,
    )

    avoid = _build_avoid_list(
        skin_type,
        issues,
    )

    notes = _build_notes(
        issues
    )

    # --------------------------------------------------------
    # PROFILE SUMMARY
    # --------------------------------------------------------

    if skin_type == "Oily":

        profile_text = (
            "Your skin profile appears oil-prone."
        )

    elif skin_type == "Dry":

        profile_text = (
            "Your skin profile appears "
            "hydration-focused."
        )

    elif skin_type == "Combination":

        profile_text = (
            "Your skin profile shows "
            "combination characteristics."
        )

    else:

        profile_text = (
            "Your skin profile appears "
            "relatively balanced."
        )

    if focus:

        primary_focus = focus[0]["title"]

        summary = (
            f"{profile_text} Your current routine "
            f"should primarily focus on "
            f"{primary_focus.lower()} while maintaining "
            f"gentle cleansing, hydration, and daily "
            f"sun protection."
        )

    else:

        summary = (
            f"{profile_text} Maintain a simple, "
            "consistent routine with gentle cleansing, "
            "hydration, and daily sun protection."
        )

    return {
        "profile": {
            "skin_type": skin_type,
            "summary": summary,
        },

        "routine": routine,

        "products": products,

        "focus": focus,

        "avoid": avoid,

        "notes": notes,
    }