import cv2
from ultralytics import YOLO
import cvzone
import math
import numpy as np

cap = cv2.VideoCapture("videos/cars.mp4")
model = YOLO("yolo_weights/yolov8n.pt")
mask = cv2.imread("project-1_Car_Counter/car_mask.png")

if mask is None:
    raise FileNotFoundError("car_mask.png not found!")

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    # Resize video
    height, width = img.shape[:2]
    scale = min(1280 / width, 720 / height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    img = cv2.resize(img, (new_width, new_height))

    # Resize mask
    resized_mask = cv2.resize(mask, (new_width, new_height))

    # Convert mask to grayscale
    mask_gray = cv2.cvtColor(resized_mask, cv2.COLOR_BGR2GRAY)

    # Convert to binary mask
    _, mask_binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

    # Apply mask to original image
    image_region = cv2.bitwise_and(img, img, mask=mask_binary)

    # YOLO detection
    results = model(image_region, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            w = x2 - x1
            h = y2 - y1

            conf = float(box.conf[0])

            cls = int(box.cls[0])

            current_class = model.names[cls]

            if current_class in ["car", "truck"] and conf > 0.3:
                cvzone.putTextRect(
                    img,
                    f"{current_class} {conf:.2f}",
                    (max(0, x1), max(35, y1)),
                    scale=0.7,
                    thickness=1,
                    offset=3,
                )

                cvzone.cornerRect(
                    img,
                    (x1, y1, w, h),
                    l=8,
                    rt=1,
                    colorR=(255, 255, 0),
                    colorC=(0, 255, 255),
                )

    cv2.imshow("Car Detection", img)

    cv2.imshow("Image Region", image_region)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
