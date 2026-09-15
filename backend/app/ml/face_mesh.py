import cv2
import mediapipe as mp
import numpy as np


mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)


def _get_landmark_point(landmarks, index, w, h):
    lm = landmarks.landmark[index]

    x = int(lm.x * w)
    y = int(lm.y * h)

    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))

    return x, y


def _build_under_eye_region(
    landmarks,
    w,
    h,
    outer_index,
    inner_index,
    lower1_index,
    lower2_index
):
    """
    Build a polygon around the under-eye area.

    The polygon is based on MediaPipe eye landmarks and
    extended slightly downward to capture the dark-circle area.
    """

    outer = _get_landmark_point(
        landmarks,
        outer_index,
        w,
        h
    )

    inner = _get_landmark_point(
        landmarks,
        inner_index,
        w,
        h
    )

    lower1 = _get_landmark_point(
        landmarks,
        lower1_index,
        w,
        h
    )

    lower2 = _get_landmark_point(
        landmarks,
        lower2_index,
        w,
        h
    )

    eye_width = max(
        abs(outer[0] - inner[0]),
        1
    )

    # Extend below the lower eyelid.
    extension = int(
        eye_width * 0.45
    )

    lower_y = max(
        lower1[1],
        lower2[1]
    )

    center_x = int(
        (outer[0] + inner[0]) / 2
    )

    left_x = max(
        min(outer[0], inner[0]) - int(eye_width * 0.10),
        0
    )

    right_x = min(
        max(outer[0], inner[0]) + int(eye_width * 0.10),
        w - 1
    )

    top_y = min(
        lower1[1],
        lower2[1]
    )

    bottom_y = min(
        lower_y + extension,
        h - 1
    )

    # Slightly wider lower region for dark-circle analysis.
    region = np.array(
        [
            [left_x, top_y],
            [right_x, top_y],
            [right_x, bottom_y],
            [center_x, bottom_y + int(extension * 0.15)],
            [left_x, bottom_y],
        ],
        dtype=np.int32
    )

    return region


def extract_face(image_bytes):

    image = cv2.imdecode(
        np.frombuffer(
            image_bytes,
            np.uint8
        ),
        cv2.IMREAD_COLOR
    )

    if image is None:
        return None, None, None

    original = image.copy()

    print(
        "Image shape:",
        image.shape
    )

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None, None, None

    h, w, _ = image.shape

    face_landmarks = (
        results.multi_face_landmarks[0]
    )

    xs = [
        int(lm.x * w)
        for lm in face_landmarks.landmark
    ]

    ys = [
        int(lm.y * h)
        for lm in face_landmarks.landmark
    ]

    x1 = max(
        min(xs) - 20,
        0
    )

    y1 = max(
        min(ys) - 20,
        0
    )

    x2 = min(
        max(xs) + 20,
        w
    )

    y2 = min(
        max(ys) + 20,
        h
    )

    face = original[
        y1:y2,
        x1:x2
    ]

    cv2.imwrite(
        "app/static/face_crop.jpg",
        face
    )

    return (
        original,
        face,
        (x1, y1, x2, y2)
    )


def get_under_eye_regions(image_bytes):
    """
    Detect the left and right under-eye regions
    using MediaPipe Face Mesh.

    Returns:
        original image,
        left under-eye polygon,
        right under-eye polygon
    """

    image = cv2.imdecode(
        np.frombuffer(
            image_bytes,
            np.uint8
        ),
        cv2.IMREAD_COLOR
    )

    if image is None:
        return None, None, None

    original = image.copy()

    h, w, _ = image.shape

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return (
            original,
            None,
            None
        )

    landmarks = (
        results.multi_face_landmarks[0]
    )

    # MediaPipe eye landmark groups.
    #
    # One eye:
    # outer 33
    # inner 133
    # lower contour 145 / 159
    #
    # Other eye:
    # outer 362
    # inner 263
    # lower contour 374 / 386

    left_region = _build_under_eye_region(
        landmarks,
        w,
        h,
        outer_index=33,
        inner_index=133,
        lower1_index=145,
        lower2_index=159
    )

    right_region = _build_under_eye_region(
        landmarks,
        w,
        h,
        outer_index=263,
        inner_index=362,
        lower1_index=374,
        lower2_index=386
    )

    return (
        original,
        left_region,
        right_region
    )