import cv2
import numpy as np


def _clamp(value, minimum=0, maximum=100):
    return max(minimum, min(maximum, value))


def _get_skin_mask(image):
    """
    Creates an approximate facial skin mask using HSV + YCrCb.
    This is a visual-analysis heuristic, not a medical diagnosis.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # HSV skin range
    lower_hsv = np.array([0, 20, 40], dtype=np.uint8)
    upper_hsv = np.array([25, 255, 255], dtype=np.uint8)

    hsv_mask = cv2.inRange(
        hsv,
        lower_hsv,
        upper_hsv
    )

    # YCrCb skin range
    lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
    upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)

    ycrcb_mask = cv2.inRange(
        ycrcb,
        lower_ycrcb,
        upper_ycrcb
    )

    # Combine both masks
    mask = cv2.bitwise_and(
        hsv_mask,
        ycrcb_mask
    )

    # Remove small noise
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


def _find_face_region(image):
    """
    Finds the largest frontal face using OpenCV Haar Cascade.
    Returns x, y, w, h or None.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    cascade_path = cv2.data.haarcascades + \
        "haarcascade_frontalface_default.xml"

    detector = cv2.CascadeClassifier(
        cascade_path
    )

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(120, 120)
    )

    if len(faces) == 0:
        return None

    # Select largest face
    face = max(
        faces,
        key=lambda item: item[2] * item[3]
    )

    return face


def _get_face_crop(image, face):
    x, y, w, h = face

    # Slightly shrink region to avoid hair/background.
    margin_x = int(w * 0.08)
    margin_y = int(h * 0.10)

    x1 = max(0, x + margin_x)
    y1 = max(0, y + margin_y)

    x2 = min(
        image.shape[1],
        x + w - margin_x
    )

    y2 = min(
        image.shape[0],
        y + h - margin_y
    )

    return image[y1:y2, x1:x2]


def _calculate_tone_variation(lab, mask):
    """
    Measures variation in facial skin lightness.
    Higher variation = more uneven tone.
    """

    pixels = lab[:, :, 0][mask > 0]

    if len(pixels) < 100:
        return 0

    std = float(np.std(pixels))

    # Approximate normalization.
    score = (std - 5) * 7

    return _clamp(score)


def _calculate_dark_spots(lab, mask):
    """
    Estimates darker-than-average regions inside the skin mask.
    """

    lightness = lab[:, :, 0].astype(np.float32)

    pixels = lightness[mask > 0]

    if len(pixels) < 100:
        return 0

    median = float(np.median(pixels))

    # Pixels significantly darker than the local median.
    threshold = median - 18

    dark_region = (
        (lightness < threshold) &
        (mask > 0)
    )

    dark_pixels = np.sum(dark_region)
    skin_pixels = np.sum(mask > 0)

    if skin_pixels == 0:
        return 0

    ratio = dark_pixels / skin_pixels

    score = ratio * 450

    return _clamp(score)


def _calculate_pigmentation(lab, mask):
    """
    Estimates color variation associated with pigmentation.
    Uses Lab a/b channel variation.
    """

    a_channel = lab[:, :, 1].astype(np.float32)
    b_channel = lab[:, :, 2].astype(np.float32)

    a_pixels = a_channel[mask > 0]
    b_pixels = b_channel[mask > 0]

    if len(a_pixels) < 100:
        return 0

    variation = (
        float(np.std(a_pixels)) +
        float(np.std(b_pixels))
    ) / 2

    score = (variation - 4) * 8

    return _clamp(score)


def _calculate_redness(image, mask):
    """
    Estimates redness from the facial skin region.
    """

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    ).astype(np.float32)

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]

    skin_r = r[mask > 0]
    skin_g = g[mask > 0]

    if len(skin_r) < 100:
        return 0

    redness_ratio = skin_r - skin_g

    average_redness = float(
        np.mean(redness_ratio)
    )

    score = (average_redness - 8) * 5

    return _clamp(score)


def _calculate_oiliness(image, mask):
    """
    Estimates visible surface shine using brightness.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    value = hsv[:, :, 2].astype(np.float32)

    skin_value = value[mask > 0]

    if len(skin_value) < 100:
        return 50

    high_brightness = np.mean(
        skin_value > 190
    )

    score = 35 + high_brightness * 130

    return _clamp(score)


def _calculate_hydration(image, mask):
    """
    Rough visual estimate based on skin brightness and color consistency.
    Not a clinical hydration measurement.
    """

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    lightness = lab[:, :, 0].astype(np.float32)

    pixels = lightness[mask > 0]

    if len(pixels) < 100:
        return 50

    mean_lightness = float(
        np.mean(pixels)
    )

    variation = float(
        np.std(pixels)
    )

    score = (
        50
        + (mean_lightness - 125) * 0.35
        - variation * 1.2
    )

    return _clamp(score)


def analyze_skin_color(image_bytes):
    """
    Performs visual skin-region analysis.

    Returns:
        uneven_skin_tone
        pigmentation
        dark_spots
        redness
        oiliness
        hydration
    """

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            "Unable to decode image."
        )

    face = _find_face_region(image)

    if face is None:
        return {
            "face_found": False,
            "uneven_skin_tone": 0,
            "pigmentation": 0,
            "dark_spots": 0,
            "redness": 0,
            "oiliness": 50,
            "hydration": 50,
        }

    face_crop = _get_face_crop(
        image,
        face
    )

    if face_crop.size == 0:
        return {
            "face_found": False,
            "uneven_skin_tone": 0,
            "pigmentation": 0,
            "dark_spots": 0,
            "redness": 0,
            "oiliness": 50,
            "hydration": 50,
        }

    skin_mask = _get_skin_mask(
        face_crop
    )

    lab = cv2.cvtColor(
        face_crop,
        cv2.COLOR_BGR2LAB
    )

    uneven_tone = _calculate_tone_variation(
        lab,
        skin_mask
    )

    pigmentation = _calculate_pigmentation(
        lab,
        skin_mask
    )

    dark_spots = _calculate_dark_spots(
        lab,
        skin_mask
    )

    redness = _calculate_redness(
        face_crop,
        skin_mask
    )

    oiliness = _calculate_oiliness(
        face_crop,
        skin_mask
    )

    hydration = _calculate_hydration(
        face_crop,
        skin_mask
    )

    return {
        "face_found": True,
        "uneven_skin_tone": round(uneven_tone, 1),
        "pigmentation": round(pigmentation, 1),
        "dark_spots": round(dark_spots, 1),
        "redness": round(redness, 1),
        "oiliness": round(oiliness, 1),
        "hydration": round(hydration, 1),
    }