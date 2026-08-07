"""
AI analysis service layer.

This module currently returns realistic MOCK predictions so the rest of the
application (routers, frontend, database writes) is fully functional without
a trained model.

TO CONNECT A REAL MODEL LATER:
  1. Keep the function signatures (`analyze_skin`, `analyze_hair`,
     `detect_face`) and their return shapes exactly as-is.
  2. Replace the body of each function with a call to your real inference
     code, e.g.:
        - Load a TensorFlow/PyTorch model once at module import time.
        - Run YOLO/MTCNN/BlazeFace for `detect_face`.
        - Run your classifier/regressor for skin type, issues and scores.
  3. No other file needs to change: routers/analysis.py only calls these
     three functions.
"""
from app.ml.yolo_detector import detect_skin
import hashlib
import random
from typing import Optional, Dict, Any


SKIN_TYPES = ["Oily", "Dry", "Combination", "Normal"]
SKIN_ISSUES = ["Mild Acne", "Dark Spots", "Large Pores", "Uneven Skin Tone", "Fine Lines", "Dryness Patches", "Redness"]
HAIR_TYPES = ["Straight", "Wavy", "Curly", "Coily"]
HAIR_ISSUES = ["Hair Fall", "Dry Hair", "Dandruff", "Frizzy Hair", "Split Ends"]


def _seed_from_bytes(data: bytes) -> int:
    return int(hashlib.sha256(data).hexdigest(), 16) % (2 ** 32)


def detect_face(image_bytes: bytes) -> bool:
    """
    Mock face/skin-region presence check.
    Real version: run a face detector (BlazeFace/MTCNN/YOLO-face) and return
    True if at least one face bounding box is found above a confidence threshold.
    """
    rng = random.Random(_seed_from_bytes(image_bytes))
    return rng.random() > 0.08  # ~92% of realistic uploads contain a detectable face


def analyze_skin(image_bytes: bytes):

    detections, output_path = detect_skin(image_bytes)

    issues = [d["issue"] for d in detections]

    if not detections:
        return {
            "mode": "skin",
            "detected_type": "Unknown",
            "detected_issues": [],
            "detections": [],
            "processed_image": "/static/scan_result.jpg",
            "scores": {
                "oiliness": 0,
                "hydration": 0,
                "health": 100,
                "confidence": 0
            },
            "face_detected": True
        }

    confidence = sum(d["confidence"] for d in detections) / len(detections)

    issue_count = len(detections)

    health = max(100 - issue_count * 12, 40)

    if "blackheads" in issues:
        oiliness = 85
    elif "whiteheads" in issues:
        oiliness = 75
    else:
        oiliness = 60

    hydration = max(100 - issue_count * 8, 45)

    if oiliness > 80:
        skin_type = "Oily"
    elif hydration < 50:
        skin_type = "Dry"
    elif oiliness > 65:
        skin_type = "Combination"
    else:
        skin_type = "Normal"

    return {
        "mode": "skin",
        "detected_type": skin_type,
        "detected_issues": issues,
        "detections": detections,
        "processed_image": "/static/scan_result.jpg",
        "scores": {
            "oiliness": oiliness,
            "hydration": hydration,
            "health": health,
            "confidence": round(confidence, 2)
        },
        "face_detected": True
    }
def analyze_hair(image_bytes: bytes) -> Dict[str, Any]:
    rng = random.Random(_seed_from_bytes(image_bytes) + 999)
    issues = [i for i in HAIR_ISSUES if rng.random() > 0.5][:3] or [HAIR_ISSUES[0]]
    return {
        "mode": "hair",
        "detected_type": rng.choice(HAIR_TYPES),
        "detected_issues": issues,
        "scores": {
            "health": round(50 + rng.random() * 45, 1),
            "confidence": round(80 + rng.random() * 17, 1),
        },
        "face_detected": False,
    }
