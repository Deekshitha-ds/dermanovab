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

from unittest import result

from app.ml.yolo_detector import detect_skin
from app.services.recommendation_service import (
    generate_recommendations,
    build_full_recommendation
)
from app.ml.face_mesh import (
    extract_face,
    get_under_eye_regions,
)

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

def _analyze_dark_circles(image_bytes):
    """
    Estimate dark-circle intensity specifically in the
    under-eye regions detected by MediaPipe Face Mesh.

    Returns scores and an overlay image.
    """

    original, left_region, right_region = get_under_eye_regions(
        image_bytes
    )

    if original is None or left_region is None or right_region is None:
        return {
            "left": 0,
            "right": 0,
            "overall": 0,
            "overlay_path": None,
        }

    h, w = original.shape[:2]

    mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    cv2.fillPoly(
        mask,
        [left_region],
        255
    )

    cv2.fillPoly(
        mask,
        [right_region],
        255
    )

    # Convert image to LAB.
    lab = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2LAB
    )

    luminance = lab[:, :, 0]

    # Analyze each eye independently.
    def region_score(region):
        region_mask = np.zeros(
            (h, w),
            dtype=np.uint8
        )

        cv2.fillPoly(
            region_mask,
            [region],
            255
        )

        pixels = luminance[
            region_mask > 0
        ]

        if len(pixels) < 20:
            return 0

        # Lower L value = darker region.
        mean_l = float(
            np.mean(pixels)
        )

        # Convert darkness into a 0-100 score.
        score = 100 - (
            mean_l / 255 * 100
        )

        return float(
            np.clip(
                score,
                0,
                100
            )
        )

    left_score = region_score(
        left_region
    )

    right_score = region_score(
        right_region
    )

    overall_score = (
        left_score + right_score
    ) / 2

    # ---------------------------------------------------------
    # Create a soft visual overlay.
    # ---------------------------------------------------------

    overlay = original.copy()

    # Dark-circle visualization color.
    # BGR = blue/purple.
    overlay_color = (
        180,
        80,
        180
    )

    colored = np.zeros_like(
        original
    )

    colored[:, :] = overlay_color

    blended = cv2.addWeighted(
        original,
        0.70,
        colored,
        0.30,
        0
    )

    # Only apply the color inside the eye masks.
    mask_3d = cv2.cvtColor(
        mask,
        cv2.COLOR_GRAY2BGR
    )

    highlighted = np.where(
        mask_3d > 0,
        blended,
        original
    )

    # Draw a soft contour around both regions.
    cv2.polylines(
        highlighted,
        [left_region],
        True,
        overlay_color,
        2
    )

    cv2.polylines(
        highlighted,
        [right_region],
        True,
        overlay_color,
        2
    )

    output_path = (
        "app/static/dark_circles_result.jpg"
    )

    cv2.imwrite(
        output_path,
        highlighted
    )

    return {
        "left": round(
            left_score,
            2
        ),
        "right": round(
            right_score,
            2
        ),
        "overall": round(
            overall_score,
            2
        ),
        "overlay_path": (
            "/static/dark_circles_result.jpg"
        ),
    }

def _analyze_pigmentation(image_bytes: bytes):
    """
    Detect localized darker pigmentation-like regions.

    The algorithm compares each skin region with its local
    surrounding brightness instead of using one global
    face brightness value.

    This is a visual estimate and is not a medical diagnosis.
    """

    original, face, face_box = extract_face(
        image_bytes
    )

    if face is None or face.size == 0:
        return {
            "score": 0,
            "overlay_path": None
        }

    # --------------------------------------------------------
    # Resize for analysis
    # --------------------------------------------------------

    face_resized = cv2.resize(
        face,
        (512, 512),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # Color spaces
    # --------------------------------------------------------

    lab = cv2.cvtColor(
        face_resized,
        cv2.COLOR_BGR2LAB
    )

    ycrcb = cv2.cvtColor(
        face_resized,
        cv2.COLOR_BGR2YCrCb
    )

    L, A, B = cv2.split(lab)
    Y, Cr, Cb = cv2.split(ycrcb)

    # --------------------------------------------------------
    # Skin mask
    # --------------------------------------------------------

    skin_mask = (
        (Cr > 125) &
        (Cr < 175) &
        (Cb > 70) &
        (Cb < 140) &
        (L > 35)
    ).astype(np.uint8) * 255

    # Clean skin mask.
    skin_mask = cv2.morphologyEx(
        skin_mask,
        cv2.MORPH_OPEN,
        np.ones((5, 5), np.uint8)
    )

    skin_mask = cv2.morphologyEx(
        skin_mask,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), np.uint8)
    )

    # Shrink the mask slightly so hairline/face-edge shadows
    # do not become pigmentation.
    skin_mask = cv2.erode(
        skin_mask,
        np.ones((13, 13), np.uint8),
        iterations=1
    )

    skin_pixels = L[
        skin_mask > 0
    ]

    if len(skin_pixels) < 500:
        return {
            "score": 0,
            "overlay_path": None
        }

    # --------------------------------------------------------
    # LOCAL brightness comparison
    # --------------------------------------------------------

    L_float = L.astype(
        np.float32
    )

    # Estimate the surrounding skin brightness.
    local_average = cv2.GaussianBlur(
        L_float,
        (41, 41),
        0
    )

    # Positive value means the pixel is darker than
    # its surrounding area.
    local_darkness = (
        local_average - L_float
    )

    # --------------------------------------------------------
    # Candidate pigmentation
    # --------------------------------------------------------

    raw_mask = (
        (skin_mask > 0) &
        (local_darkness > 15)
    ).astype(np.uint8) * 255

    # Clean isolated noise.
    raw_mask = cv2.morphologyEx(
        raw_mask,
        cv2.MORPH_OPEN,
        np.ones((5, 5), np.uint8)
    )

    raw_mask = cv2.morphologyEx(
        raw_mask,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), np.uint8)
    )

    # --------------------------------------------------------
    # Keep only localized regions
    # --------------------------------------------------------

    binary = (
        raw_mask > 0
    ).astype(np.uint8)

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    pigmentation_mask = np.zeros(
        (512, 512),
        dtype=np.uint8
    )

    for i in range(1, count):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        # Ignore huge regions.
        #
        # Huge regions are usually:
        # - shadows
        # - lighting gradients
        # - eyelid/eye regions
        # - general skin-tone differences
        #
        # Keep only reasonably localized areas.

        if 80 <= area <= 2500:

            component = (
                labels == i
            ).astype(np.uint8) * 255

            pigmentation_mask = cv2.bitwise_or(
                pigmentation_mask,
                component
            )

    # --------------------------------------------------------
    # Smooth only the boundaries
    # --------------------------------------------------------

    pigmentation_mask = cv2.GaussianBlur(
        pigmentation_mask,
        ( nine := 9, nine ),
        0
    )

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    skin_area = np.sum(
        skin_mask > 0
    )

    affected_area = np.sum(
        pigmentation_mask > 80
    )

    ratio = (
        affected_area
        /
        max(skin_area, 1)
        * 100
    )

    # Much more conservative scoring.
    score = np.interp(
        ratio,
        [0.2, 5],
        [5, 85]
    )

    score = float(
        np.clip(
            score,
            0,
            100
        )
    )

    # --------------------------------------------------------
    # CREATE NATURAL-LOOKING OVERLAY
    # --------------------------------------------------------

    purple = np.zeros_like(
        face_resized
    )

    purple[:, :] = (
        175,
        85,
        210
    )

    # Very subtle blend.
    blended = cv2.addWeighted(
        face_resized,
        0.84,
        purple,
        0.16,
        0
    )

    mask_soft = (
        pigmentation_mask.astype(
            np.float32
        )
        / 255.0
    )

    mask_soft = (
        mask_soft
        * 0.70
    )

    mask_soft = mask_soft[
        :,
        :,
        None
    ]

    highlighted = (
        face_resized.astype(
            np.float32
        )
        * (1.0 - mask_soft)
        +
        blended.astype(
            np.float32
        )
        * mask_soft
    )

    highlighted = np.clip(
        highlighted,
        0,
        255
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Put result back into original image
    # --------------------------------------------------------

    x1, y1, x2, y2 = face_box

    face_width = max(
        x2 - x1,
        1
    )

    face_height = max(
        y2 - y1,
        1
    )

    highlighted_original = cv2.resize(
        highlighted,
        (
            face_width,
            face_height
        ),
        interpolation=cv2.INTER_LINEAR
    )

    result = original.copy()

    result[
        y1:y2,
        x1:x2
    ] = highlighted_original

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        "app/static/pigmentation_result.jpg"
    )

    cv2.imwrite(
        output_path,
        result,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    return {
        "score": round(
            score,
            1
        ),
        "overlay_path":
            "/static/pigmentation_result.jpg"
    }

def _analyze_uneven_tone(image_bytes: bytes):
    """
    Estimate and highlight regions of uneven skin tone.

    This is a visual computer-vision estimate and is not
    a medical diagnosis.
    """

    original, face, face_box = extract_face(image_bytes)

    if face is None or face.size == 0:
        return {
            "score": 0,
            "overlay_path": None
        }

    # --------------------------------------------------------
    # Resize face for analysis
    # --------------------------------------------------------

    face_resized = cv2.resize(
        face,
        (512, 512)
    )

    # --------------------------------------------------------
    # Convert to LAB
    # --------------------------------------------------------

    lab = cv2.cvtColor(
        face_resized,
        cv2.COLOR_BGR2LAB
    )

    L, A, B = cv2.split(lab)

    # --------------------------------------------------------
    # Build skin mask
    # --------------------------------------------------------

    ycrcb = cv2.cvtColor(
        face_resized,
        cv2.COLOR_BGR2YCrCb
    )

    Y, Cr, Cb = cv2.split(ycrcb)

    skin_mask = (
        (Cr > 125) &
        (Cr < 175) &
        (Cb > 70) &
        (Cb < 140) &
        (L > 35)
    ).astype(np.uint8) * 255

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    skin_mask = cv2.morphologyEx(
        skin_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    skin_mask = cv2.morphologyEx(
        skin_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    skin_pixels = L[
        skin_mask > 0
    ]

    if len(skin_pixels) < 500:
        return {
            "score": 0,
            "overlay_path": None
        }

    # --------------------------------------------------------
    # Measure local tone variation
    # --------------------------------------------------------

    L_float = L.astype(
        np.float32
    )

    local_average = cv2.GaussianBlur(
        L_float,
        (31, 31),
        0
    )

    local_difference = cv2.absdiff(
        L_float,
        local_average
    )

    # Only look at skin.
    uneven_mask = (
        (skin_mask > 0) &
        (local_difference > 7)
    ).astype(np.uint8) * 255

    # Remove tiny isolated areas.
    uneven_mask = cv2.morphologyEx(
        uneven_mask,
        cv2.MORPH_OPEN,
        np.ones((5, 5), np.uint8)
    )

    uneven_mask = cv2.morphologyEx(
        uneven_mask,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), np.uint8)
    )

    # Smooth the mask so the result looks premium.
    uneven_mask = cv2.GaussianBlur(
        uneven_mask,
        (15, 15),
        0
    )

    # --------------------------------------------------------
    # Calculate score
    # --------------------------------------------------------

    skin_area = np.sum(
        skin_mask > 0
    )

    uneven_area = np.sum(
        uneven_mask > 60
    )

    ratio = (
        uneven_area
        /
        max(skin_area, 1)
        * 100
    )

    score = np.interp(
        ratio,
        [2, 20],
        [5, 95]
    )

    score = float(
        np.clip(
            score,
            0,
            100
        )
    )

    # --------------------------------------------------------
    # Create visual overlay
    # --------------------------------------------------------

    tone_color = np.zeros_like(
        face_resized
    )

    # BGR yellow/golden tone
    tone_color[:, :] = (
        40,
        190,
        230
    )

    blended = cv2.addWeighted(
        face_resized,
        0.76,
        tone_color,
        0.24,
        0
    )

    mask_3d = cv2.cvtColor(
        uneven_mask,
        cv2.COLOR_GRAY2BGR
    )

    highlighted = np.where(
        mask_3d > 35,
        blended,
        face_resized
    )

    # --------------------------------------------------------
    # Put result back onto original image
    # --------------------------------------------------------

    x1, y1, x2, y2 = face_box

    face_width = max(
        x2 - x1,
        1
    )

    face_height = max(
        y2 - y1,
        1
    )

    highlighted_original_size = cv2.resize(
        highlighted,
        (
            face_width,
            face_height
        )
    )

    result = original.copy()

    result[
        y1:y2,
        x1:x2
    ] = highlighted_original_size

    output_path = (
        "app/static/uneven_tone_result.jpg"
    )

    cv2.imwrite(
        output_path,
        result
    )

    return {
        "score": round(
            score,
            1
        ),
        "overlay_path":
            "/static/uneven_tone_result.jpg"
    }

def _build_combined_skin_overlay(
    image_bytes: bytes,
    detections
):
    """
    Create one clean facial-analysis image containing:

    - Acne / YOLO detections
    - Pigmentation regions
    - Uneven skin-tone regions
    - Soft oval dark-circle regions

    This is a visual computer-vision estimate and is not
    a medical diagnosis.
    """

    original, face, face_box = extract_face(
        image_bytes
    )

    if (
        original is None
        or face is None
        or face.size == 0
        or face_box is None
    ):
        return None

    result = original.copy()

    # ========================================================
    # FACE CROP
    # ========================================================

    x1, y1, x2, y2 = face_box

    face_width = max(
        x2 - x1,
        1
    )

    face_height = max(
        y2 - y1,
        1
    )

    face_resized = cv2.resize(
        face,
        (512, 512),
        interpolation=cv2.INTER_AREA
    )

    # ========================================================
    # COLOR SPACES
    # ========================================================

    lab = cv2.cvtColor(
        face_resized,
        cv2.COLOR_BGR2LAB
    )

    L, A, B = cv2.split(lab)

    ycrcb = cv2.cvtColor(
        face_resized,
        cv2.COLOR_BGR2YCrCb
    )

    Y, Cr, Cb = cv2.split(ycrcb)

    # ========================================================
    # SKIN MASK
    # ========================================================

    skin_mask = (
        (Cr > 125) &
        (Cr < 175) &
        (Cb > 70) &
        (Cb < 140) &
        (L > 35)
    ).astype(np.uint8) * 255

    skin_mask = cv2.morphologyEx(
        skin_mask,
        cv2.MORPH_OPEN,
        np.ones((5, 5), np.uint8)
    )

    skin_mask = cv2.morphologyEx(
        skin_mask,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), np.uint8)
    )

    skin_mask = cv2.erode(
        skin_mask,
        np.ones((11, 11), np.uint8),
        iterations=1
    )

    # ========================================================
    # HELPER
    # ========================================================

    def clean_regions(
        mask,
        min_area,
        max_area
    ):

        binary = (
            mask > 0
        ).astype(np.uint8)

        count, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8
            )
        )

        cleaned = np.zeros_like(
            mask
        )

        for i in range(1, count):

            area = stats[
                i,
                cv2.CC_STAT_AREA
            ]

            if (
                area >= min_area
                and area <= max_area
            ):

                cleaned[
                    labels == i
                ] = 255

        return cleaned

    # ========================================================
    # PIGMENTATION
    # ========================================================

    skin_pixels = L[
        skin_mask > 0
    ]

    pigmentation_mask = np.zeros(
        (512, 512),
        dtype=np.uint8
    )

    if len(skin_pixels) >= 500:

        local_average = cv2.GaussianBlur(
            L.astype(np.float32),
            (41, 41),
            0
        )

        local_darkness = (
            local_average -
            L.astype(np.float32)
        )

        raw_pigmentation = (
            (skin_mask > 0) &
            (local_darkness > 15)
        ).astype(np.uint8) * 255

        raw_pigmentation = cv2.morphologyEx(
            raw_pigmentation,
            cv2.MORPH_OPEN,
            np.ones((5, 5), np.uint8)
        )

        raw_pigmentation = cv2.morphologyEx(
            raw_pigmentation,
            cv2.MORPH_CLOSE,
            np.ones((7, 7), np.uint8)
        )

        pigmentation_mask = clean_regions(
            raw_pigmentation,
            min_area=80,
            max_area=2500
        )

        pigmentation_mask = cv2.GaussianBlur(
            pigmentation_mask,
            (11, 11),
            0
        )

    # ========================================================
    # UNEVEN SKIN TONE
    # ========================================================

    L_float = L.astype(
        np.float32
    )

    local_average = cv2.GaussianBlur(
        L_float,
        (41, 41),
        0
    )

    local_difference = cv2.absdiff(
        L_float,
        local_average
    )

    raw_uneven = (
        (skin_mask > 0) &
        (local_difference > 12)
    ).astype(np.uint8) * 255

    raw_uneven = cv2.morphologyEx(
        raw_uneven,
        cv2.MORPH_OPEN,
        np.ones((5, 5), np.uint8)
    )

    raw_uneven = cv2.morphologyEx(
        raw_uneven,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), np.uint8)
    )

    uneven_mask = clean_regions(
        raw_uneven,
        min_area=100,
        max_area=2200
    )

    uneven_mask = cv2.GaussianBlur(
        uneven_mask,
        (13, 13),
        0
    )

    # ========================================================
    # MAP FACE MASKS TO ORIGINAL IMAGE
    # ========================================================

    pigment_original = cv2.resize(
        pigmentation_mask,
        (
            face_width,
            face_height
        ),
        interpolation=cv2.INTER_LINEAR
    )

    uneven_original = cv2.resize(
        uneven_mask,
        (
            face_width,
            face_height
        ),
        interpolation=cv2.INTER_LINEAR
    )

    full_pigment = np.zeros(
        original.shape[:2],
        dtype=np.uint8
    )

    full_uneven = np.zeros(
        original.shape[:2],
        dtype=np.uint8
    )

    full_pigment[
        y1:y2,
        x1:x2
    ] = pigment_original

    full_uneven[
        y1:y2,
        x1:x2
    ] = uneven_original

    # ========================================================
    # OVERLAY HELPER
    # ========================================================

    def apply_soft_overlay(
        base,
        mask,
        color,
        alpha_strength
    ):

        alpha = (
            mask.astype(np.float32)
            / 255.0
            * alpha_strength
        )

        alpha = alpha[:, :, None]

        color_layer = np.zeros_like(
            base
        )

        color_layer[:, :] = color

        output = (
            base.astype(np.float32)
            * (1.0 - alpha)
            +
            color_layer.astype(np.float32)
            * alpha
        )

        return np.clip(
            output,
            0,
            255
        ).astype(np.uint8)

    # ========================================================
    # PIGMENTATION
    # ========================================================

    result = apply_soft_overlay(
        result,
        full_pigment,
        (175, 85, 210),
        0.16
    )

    # ========================================================
    # UNEVEN TONE
    # ========================================================

    result = apply_soft_overlay(
        result,
        full_uneven,
        (40, 190, 230),
        0.10
    )
    # ========================================================
    # DARK CIRCLES — CURVED UNDER-EYE SHAPES
    # ========================================================

    _, left_eye_region, right_eye_region = (
        get_under_eye_regions(
            image_bytes
        )
    )

    dark_mask = np.zeros(
        original.shape[:2],
        dtype=np.uint8
    )


    def add_under_eye_shape(
        mask,
        region
    ):
        """
        Create a soft, slightly rotated under-eye oval.

        The oval is positioned from the center of the
        under-eye region and kept close to the lower eyelid.
        """

        if region is None:
            return

        points = region.reshape(
            -1,
            2
        ).astype(np.float32)

        if len(points) < 4:
            return

        min_x = float(
            np.min(points[:, 0])
        )

        max_x = float(
            np.max(points[:, 0])
        )

        min_y = float(
            np.min(points[:, 1])
        )

        max_y = float(
            np.max(points[:, 1])
        )

        eye_width = max(
            max_x - min_x,
            1
        )

        eye_height = max(
            max_y - min_y,
            1
        )

        # --------------------------------------------------------
        # Horizontal center
        # --------------------------------------------------------

        center_x = (
            min_x + max_x
        ) / 2

        # --------------------------------------------------------
        # IMPORTANT:
        # Put the center around the lower eyelid,
        # not at the bottom of the whole region.
        # --------------------------------------------------------

        center_y = (
            min_y
            + eye_height * 0.48
        )

        # --------------------------------------------------------
        # Wide, shallow shape similar to the reference image.
        # --------------------------------------------------------

        oval_width = max(
            eye_width * 0.78,
            35
        )

        oval_height = max(
            eye_width * 0.17,
            12
        )

        # --------------------------------------------------------
        # Estimate slight eye angle.
        # --------------------------------------------------------

        # Use left-most and right-most points to estimate
        # the direction of the eye.
        sorted_points = points[
            np.argsort(points[:, 0])
        ]

        left_point = sorted_points[0]
        right_point = sorted_points[-1]

        angle = np.degrees(
            np.arctan2(
                right_point[1] - left_point[1],
                right_point[0] - left_point[0]
            )
        )

        # Limit rotation so the indicator remains elegant.
        angle = float(
            np.clip(
                angle,
                -12,
                12
            )
        )

        # --------------------------------------------------------
        # Draw rotated ellipse.
        # --------------------------------------------------------

        cv2.ellipse(
            mask,
            (
                int(center_x),
                int(center_y)
            ),
            (
                int(oval_width / 2),
                int(oval_height / 2)
            ),
            angle,
            0,
            360,
            255,
            -1
        )


    # Left under-eye
    add_under_eye_shape(
        dark_mask,
        left_eye_region
    )

    # Right under-eye
    add_under_eye_shape(
        dark_mask,
        right_eye_region
    )

    # --------------------------------------------------------
    # Soft feathered edges.
    # --------------------------------------------------------

    dark_mask = cv2.GaussianBlur(
        dark_mask,
        (31, 31),
        0
    )

    # --------------------------------------------------------
    # Apply subtle blue/orange visualization.
    # --------------------------------------------------------

    result = apply_soft_overlay(
        result,
        dark_mask,
        (220, 100, 40),
        0.14
    )

   # ========================================================
    # ACNE / YOLO CIRCLES + LABELS
    # ========================================================

    for detection in detections:

        bbox = detection.get("bbox")

        if not bbox:
            continue

        box_x1 = int(bbox["x"])
        box_y1 = int(bbox["y"])

        box_x2 = (
            box_x1 +
            int(bbox["width"])
        )

        box_y2 = (
            box_y1 +
            int(bbox["height"])
        )

        issue = str(
            detection.get("issue", "")
        ).strip()

        issue_lower = issue.lower()

        # ----------------------------------------------------
        # Find center and radius
        # ----------------------------------------------------

        center_x = int(
            (box_x1 + box_x2) / 2
        )

        center_y = int(
            (box_y1 + box_y2) / 2
        )

        width = box_x2 - box_x1
        height = box_y2 - box_y1

        radius = int(
            max(width, height) / 2
        )

        radius = max(radius, 8)

        # ----------------------------------------------------
        # Colors
        # OpenCV uses BGR
        # ----------------------------------------------------

        if issue_lower in [
            "papules",
            "pustules",
            "nodules"
        ]:
            # RED — pimples / acne
            color = (0, 0, 255)

        elif issue_lower == "dark spot":
            # BLUE — pigmentation / dark spot
            color = (255, 0, 0)

        elif issue_lower == "blackheads":
            # ORANGE
            color = (0, 165, 255)

        elif issue_lower == "whiteheads":
            # PURPLE
            color = (255, 0, 255)

        else:
            # Default
            color = (0, 255, 0)

        # ----------------------------------------------------
        # Draw circle
        # ----------------------------------------------------

        cv2.circle(
            result,
            (
                center_x,
                center_y
            ),
            radius,
            color,
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Draw label
        # ----------------------------------------------------

        if issue:

            label = issue

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.50
            thickness = 1

            (
                text_width,
                text_height
            ), baseline = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness
            )

            label_x = center_x - int(
                text_width / 2
            )

            label_y = center_y - radius - 8

            # Keep label inside image
            if label_y - text_height < 0:
                label_y = center_y + radius + text_height + 8

            # ------------------------------------------------
            # Label background
            # ------------------------------------------------

            cv2.rectangle(
                result,
                (
                    label_x - 4,
                    label_y - text_height - baseline - 3
                ),
                (
                    label_x + text_width + 4,
                    label_y + 3
                ),
                color,
                -1
            )

            # ------------------------------------------------
            # Label text
            # ------------------------------------------------

            cv2.putText(
                result,
                label,
                (
                    label_x,
                    label_y
                ),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

    # ========================================================
    # SAVE
    # ========================================================

    output_path = (
        "app/static/combined_skin_result.jpg"
    )

    cv2.imwrite(
        output_path,
        result,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    return (
        "/static/combined_skin_result.jpg"
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
    combined_overlay = _build_combined_skin_overlay(
    image_bytes,
    detections
)


    # --------------------------------------------------------
    # Visual skin analysis
    # --------------------------------------------------------

    visual = _analyze_skin_color(
        image_bytes
    )
    # --------------------------------------------------------
    # Dark-circle analysis
    # --------------------------------------------------------

    dark_circles = _analyze_dark_circles(
    image_bytes
)
    pigmentation = _analyze_pigmentation(
    image_bytes
)
    uneven_tone = _analyze_uneven_tone(
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

    if uneven_tone["score"] >= 55:
        visual_issues.append(
        "Uneven Skin Tone"
    )

    if pigmentation["score"] >= 55:
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

    if dark_circles["overall"] >= 55:
        visual_issues.append(
        "Dark Circles"
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
    full_recommendation = build_full_recommendation(
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
        "full_recommendation": full_recommendation,

        "detections": detections,

        "skin_metrics": {

        "uneven_tone": uneven_tone[
        "score"
    ],

        "pigmentation": pigmentation[
        "score"
    ],

        "dark_spots": visual[
        "dark_spots"
    ],

        "redness": visual[
        "redness"
    ]
},
        "dark_circles": dark_circles,
        "pigmentation": pigmentation,
        "uneven_tone": uneven_tone,

        "processed_image": combined_overlay,

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