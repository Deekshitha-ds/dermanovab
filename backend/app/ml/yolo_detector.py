from ultralytics import YOLO
import cv2
from app.ml.face_mesh import extract_face

# ---------------- LOAD MODEL ---------------- #

model = YOLO("app/ml/models/best.pt")

print("===================================")
print("Model:", model.ckpt_path)
print("Classes:", model.names)
print("===================================")


# ---------------- DETECT SKIN ---------------- #

def detect_skin(image_bytes: bytes):

    # Extract face
    original, face, face_box = extract_face(image_bytes)

    if original is None or face is None or face_box is None:
        return [], None

    offset_x, offset_y, _, _ = face_box

    # Original face size
    face_h, face_w = face.shape[:2]

    # Resize ONLY for YOLO
    yolo_input = cv2.resize(face, (640, 640))

    cv2.imwrite("app/static/yolo_input.jpg", yolo_input)

    results = model(
    yolo_input,
    imgsz=640,
    conf=0.10,
    verbose=True
)

    print(results[0])
    print("Boxes:", len(results[0].boxes))

    detections = []

    # Scale factors
    scale_x = face_w / 640
    scale_y = face_h / 640

    # Draw on original image
    for box in results[0].boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        # YOLO coordinates (640x640)
        x1, y1, x2, y2 = box.xyxy[0]

        # Convert to original face coordinates
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        # Convert to original image coordinates
        x1 += offset_x
        y1 += offset_y
        x2 += offset_x
        y2 += offset_y

        width = x2 - x1
        height = y2 - y1

        center_x = x1 + width // 2
        center_y = y1 + height // 2

        issue = model.names[cls]

        # ---------- Color ---------- #

        color = (0,255,0)

        if issue.lower() == "acne":
            color = (0,0,255)

        elif issue.lower() == "blackheads":
            color = (0,165,255)

        elif issue.lower() == "dark spots":
            color = (255,0,255)

        elif issue.lower() == "pigmentation":
            color = (255,0,255)

        elif issue.lower() == "wrinkles":
            color = (255,255,0)

        elif issue.lower() == "redness":
            color = (50,50,255)

        # ---------- Draw ---------- #

        cv2.rectangle(
            original,
            (x1,y1),
            (x2,y2),
            color,
            2
        )

        cv2.circle(
            original,
            (center_x,center_y),
            4,
            color,
            -1
        )

        label = f"{issue} {conf*100:.1f}%"

        cv2.putText(
            original,
            label,
            (x1,max(y1-10,20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        detections.append({

            "issue": issue,

            "confidence": round(conf*100,2),

            "bbox": {

                "x": x1,

                "y": y1,

                "width": width,

                "height": height

            },

            "center": {

                "x": center_x,

                "y": center_y

            }

        })

    # Save processed image
    output_path = "app/static/scan_result.jpg"

    cv2.imwrite(output_path, original)

    return detections, output_path