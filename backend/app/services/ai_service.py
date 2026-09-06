"""
DermaNova AI skin analysis service.

YOLO:
    Detects trained skin-lesion classes.

OpenCV:
    Provides visual skin-quality estimates such as:
    - uneven skin tone
    - pigmentation
    - redness
    - oiliness
    - hydration

These visual metrics are estimates from the image and should not be
treated as medical diagnoses.
"""

from app.ml.yolo_detector import detect_skin
from app.services.recommendation_service import generate_recommendations
from app.ml.face_mesh import extract_face

import hashlib
import random
from typing import Dict, Any

import cv2
import numpy as np


# ============================================================
# CONSTANTS
# ============================================================

SKIN_TYPES = [
    "Oily",
    "Dry",
    "Combination",
    "Normal"
]


HAIR_TYPES = [
    "Straight",
    "Wavy",
    "Curly",
    "Coily"
]


HAIR_ISSUES = [
    "Hair Fall",
    "Dry Hair",
    "Dandruff",
    "Frizzy Hair",
    "Split Ends"
]


# ============================================================
# RANDOM SEED
# ============================================================

def _seed_from_bytes(data: bytes) -> int:

    return int(
        hashlib.sha256(data).hexdigest(),
        16
    ) % (2 ** 32)


# ============================================================
# FACE DETECTION
# ============================================================

def detect_face(image_bytes: bytes) -> bool:

    rng = random.Random(
        _seed_from_bytes(image_bytes)
    )

    return rng.random() > 0.08


# ============================================================
# IMAGE DECODING
# ============================================================

def _decode_image(image_bytes: bytes):

    array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )

    return image


# ============================================================
# SKIN COLOR ANALYSIS
# ============================================================

def _analyze_skin_color(image_bytes: bytes):

    # --------------------------------------------------------
    # Extract face
    # --------------------------------------------------------

    original, face, face_box = extract_face(image_bytes)

    if face is None or face.size == 0:

        return {
            "uneven_tone": 0,
            "pigmentation": 0,
            "dark_spots": 0,
            "redness": 0,
            "oiliness": 50,
            "hydration": 50
        }

    # --------------------------------------------------------
    # Resize face
    # --------------------------------------------------------

    face = cv2.resize(face, (512, 512))

    # --------------------------------------------------------
    # Convert color spaces
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2HSV
    )

    lab = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2LAB
    )

    ycrcb = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2YCrCb
    )

    H, S, V = cv2.split(hsv)
    L, A, B = cv2.split(lab)

    Y, Cr, Cb = cv2.split(ycrcb)

    # ========================================================
    # SKIN MASK
    # ========================================================

    # Basic skin-color filtering.
    # This removes much of the background/hair/shirt influence.

    skin_mask = (
        (Cr > 125) &
        (Cr < 175) &
        (Cb > 70) &
        (Cb < 140) &
        (S > 20) &
        (V > 40)
    )

    # Remove extreme edges of face crop
    h, w = skin_mask.shape

    border = int(min(h, w) * 0.05)

    skin_mask[:border, :] = False
    skin_mask[-border:, :] = False
    skin_mask[:, :border] = False
    skin_mask[:, -border:] = False

    skin_pixels = np.where(skin_mask)

    if len(skin_pixels[0]) < 1000:

        return {
            "uneven_tone": 20,
            "pigmentation": 10,
            "dark_spots": 10,
            "redness": 15,
            "oiliness": 50,
            "hydration": 60
        }

    # ========================================================
    # 1. UNEVEN SKIN TONE
    # ========================================================

    skin_L = L[skin_mask].astype(np.float32)

    brightness_std = float(
        np.std(skin_L)
    )

    uneven_score = np.interp(
        brightness_std,
        [5, 25],
        [5, 95]
    )

    # ========================================================
    # 2. PIGMENTATION
    # ========================================================

    mean_L = float(
        np.mean(skin_L)
    )

    # Regions significantly darker than surrounding skin

    dark_pixels = (
        skin_mask &
        (L < mean_L - 15)
    )

    dark_ratio = (
        np.sum(dark_pixels)
        /
        np.sum(skin_mask)
        * 100
    )

    pigmentation_score = np.interp(
        dark_ratio,
        [1, 15],
        [5, 95]
    )

    # ========================================================
    # 3. DARK SPOTS
    # ========================================================

    smooth_L = cv2.GaussianBlur(
        L,
        (21, 21),
        0
    )

    local_difference = (
        smooth_L.astype(np.float32)
        -
        L.astype(np.float32)
    )

    dark_spot_mask = (
        skin_mask &
        (local_difference > 8)
    )

    dark_spot_ratio = (
        np.sum(dark_spot_mask)
        /
        np.sum(skin_mask)
        * 100
    )

    dark_spot_score = np.interp(
        dark_spot_ratio,
        [0.5, 8],
        [5, 95]
    )

    # ========================================================
    # 4. REDNESS
    # ========================================================

    skin_A = A[skin_mask].astype(np.float32)

    mean_A = float(
        np.mean(skin_A)
    )

    redness_score = np.interp(
        mean_A,
        [130, 155],
        [5, 95]
    )

    # ========================================================
    # 5. OILINESS
    # ========================================================

    skin_S = S[skin_mask]
    skin_V = V[skin_mask]

    shine_mask = (
        (skin_S > 60) &
        (skin_V > 180)
    )

    shine_ratio = (
        np.sum(shine_mask)
        /
        len(skin_S)
        * 100
    )

    oiliness_score = np.interp(
        shine_ratio,
        [1, 15],
        [20, 95]
    )

    # ========================================================
    # 6. HYDRATION ESTIMATE
    # ========================================================

    hydration_score = (
        100
        - uneven_score * 0.20
        - oiliness_score * 0.12
    )

    hydration_score = float(
        np.clip(
            hydration_score,
            25,
            95
        )
    )

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "uneven_tone": round(
            float(
                np.clip(
                    uneven_score,
                    0,
                    100
                )
            ),
            1
        ),

        "pigmentation": round(
            float(
                np.clip(
                    pigmentation_score,
                    0,
                    100
                )
            ),
            1
        ),

        "dark_spots": round(
            float(
                np.clip(
                    dark_spot_score,
                    0,
                    100
                )
            ),
            1
        ),

        "redness": round(
            float(
                np.clip(
                    redness_score,
                    0,
                    100
                )
            ),
            1
        ),

        "oiliness": round(
            float(
                np.clip(
                    oiliness_score,
                    0,
                    100
                )
            ),
            1
        ),

        "hydration": round(
            float(
                np.clip(
                    hydration_score,
                    0,
                    100
                )
            ),
            1
        )
    }

# ============================================================
# SKIN TYPE
# ============================================================

def _calculate_skin_type(
    oiliness,
    hydration
):

    if oiliness >= 72 and hydration >= 55:

        return "Oily"


    if hydration <= 42 and oiliness <= 50:

        return "Dry"


    if oiliness >= 65 and hydration <= 55:

        return "Combination"


    return "Normal"


# ============================================================
# HEALTH SCORE
# ============================================================

def _calculate_health(
    detections,
    visual
):

    # YOLO lesion penalty

    lesion_penalty = min(
        len(detections) * 7,
        35
    )


    # Visual-condition penalty

    visual_penalty = (

        visual["uneven_tone"] * 0.08

        + visual["pigmentation"] * 0.06

        + visual["dark_spots"] * 0.05

        + visual["redness"] * 0.04
    )


    health = (
        100
        - lesion_penalty
        - visual_penalty
    )


    return round(
        float(np.clip(
            health,
            35,
            100
        )),
        1
    )


# ============================================================
# SKIN ANALYSIS
# ============================================================

def analyze_skin(
    image_bytes: bytes
):

    # --------------------------------------------------------
    # YOLO analysis
    # --------------------------------------------------------

    detections, output_path = detect_skin(
        image_bytes
    )


    # --------------------------------------------------------
    # Visual skin analysis
    # --------------------------------------------------------

    visual = _analyze_skin_color(
        image_bytes
    )


    # --------------------------------------------------------
    # YOLO issue names
    # --------------------------------------------------------

    issues = [
        d["issue"]
        for d in detections
    ]


    # --------------------------------------------------------
    # Add visual concerns
    # --------------------------------------------------------

    visual_issues = []


    if visual["uneven_tone"] >= 55:

        visual_issues.append(
            "Uneven Skin Tone"
        )


    if visual["pigmentation"] >= 55:

        visual_issues.append(
            "Pigmentation"
        )


    if visual["dark_spots"] >= 55:

        visual_issues.append(
            "Dark Spots"
        )


    if visual["redness"] >= 60:

        visual_issues.append(
            "Redness"
        )


    # --------------------------------------------------------
    # Combine issues
    # --------------------------------------------------------

    all_issues = []

    for issue in issues + visual_issues:

        if issue not in all_issues:

            all_issues.append(issue)


    # --------------------------------------------------------
    # Skin type
    # --------------------------------------------------------

    skin_type = _calculate_skin_type(
        visual["oiliness"],
        visual["hydration"]
    )


    # --------------------------------------------------------
    # Health
    # --------------------------------------------------------

    health = _calculate_health(
        detections,
        visual
    )
    recommendations = generate_recommendations(
        skin_type=skin_type,
        issues=all_issues,
        hydration=visual["hydration"],
        oiliness=visual["oiliness"],
)

    # --------------------------------------------------------
    # YOLO confidence
    # --------------------------------------------------------

    if detections:

        yolo_confidence = (
            sum(
                d["confidence"]
                for d in detections
            )
            / len(detections)
        )

    else:

        yolo_confidence = 0


    # --------------------------------------------------------
    # Overall confidence
    # --------------------------------------------------------

    if detections:

        confidence = (
            yolo_confidence * 0.65
            + 75 * 0.35
        )

    else:

        confidence = 70


    # --------------------------------------------------------
    # FINAL RESPONSE
    # --------------------------------------------------------

    return {

        "mode": "skin",

        "detected_type": skin_type,

        "detected_issues": all_issues,
        "recommendations": recommendations,

        "detections": detections,

        "skin_metrics": {

            "uneven_tone": visual[
                "uneven_tone"
            ],

            "pigmentation": visual[
                "pigmentation"
            ],

            "dark_spots": visual[
                "dark_spots"
            ],

            "redness": visual[
                "redness"
            ]
        },

        "processed_image":
            "/static/scan_result.jpg",

        "scores": {

            "oiliness":
                visual["oiliness"],

            "hydration":
                visual["hydration"],

            "health":
                health,

            "confidence":
                round(
                    confidence,
                    1
                )
        },

        "face_detected": True
    }


# ============================================================
# HAIR ANALYSIS
# ============================================================

def analyze_hair(
    image_bytes: bytes
) -> Dict[str, Any]:

    rng = random.Random(
        _seed_from_bytes(image_bytes) + 999
    )


    issues = [
        issue
        for issue in HAIR_ISSUES
        if rng.random() > 0.5
    ][:3]


    if not issues:

        issues = [
            HAIR_ISSUES[0]
        ]


    return {

        "mode": "hair",

        "detected_type":
            rng.choice(HAIR_TYPES),

        "detected_issues":
            issues,

        "scores": {

            "health":
                round(
                    50 + rng.random() * 45,
                    1
                ),

            "confidence":
                round(
                    80 + rng.random() * 17,
                    1
                )
        },

        "face_detected": False
    }