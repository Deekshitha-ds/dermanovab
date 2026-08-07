import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

def extract_face(image_bytes):

    image = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    original = image.copy()

    print("Image shape:", image.shape)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None, None, None

    h, w, _ = image.shape

    face_landmarks = results.multi_face_landmarks[0]

    xs = [int(lm.x * w) for lm in face_landmarks.landmark]
    ys = [int(lm.y * h) for lm in face_landmarks.landmark]

    x1 = max(min(xs) - 20, 0)
    y1 = max(min(ys) - 20, 0)

    x2 = min(max(xs) + 20, w)
    y2 = min(max(ys) + 20, h)

    face = original[y1:y2, x1:x2]

    cv2.imwrite("app/static/face_crop.jpg", face)

    return original, face, (x1, y1, x2, y2)