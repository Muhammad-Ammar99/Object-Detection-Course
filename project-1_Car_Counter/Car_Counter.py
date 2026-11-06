import cv2
from ultralytics import YOLO
import cvzone
import math
import numpy as np
from sort import *

cap = cv2.VideoCapture("videos/cars.mp4")
model = YOLO("yolo_weights/yolov8n.pt")
mask = cv2.imread("project-1_Car_Counter/car_mask.png")

classNames = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# Tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [120, 400, 1275, 400]

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

    # Apply mask to original image
    # Resize mask
    resized_mask = cv2.resize(mask, (new_width, new_height))
    # Convert mask to grayscale
    mask_gray = cv2.cvtColor(resized_mask, cv2.COLOR_BGR2GRAY)

    # Convert to binary mask
    _, mask_binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    image_region = cv2.bitwise_and(img, img, mask=mask_binary)

    # YOLO detection
    results = model(image_region, stream=True)

    detections = np.empty((0, 5))  # Initialize an empty array for detections

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            w = x2 - x1
            h = y2 - y1

            conf = float(box.conf[0])

            cls = int(box.cls[0])

            current_class = classNames[cls]

            if current_class in ["car", "truck"] and conf > 0.3:
                # cvzone.putTextRect(img,f"{current_class} {conf:.2f}",(max(0, x1), max(35, y1)),scale=0.7,thickness=1,offset=3,)

                cvzone.cornerRect(
                    img,
                    (x1, y1, w, h),
                    l=8,
                    rt=5,
                    colorR=(255, 255, 0),
                    colorC=(0, 255, 255),
                )

                CurrentArray = np.array([[x1, y1, x2, y2, conf]])
                detections = np.vstack(
                    (detections, CurrentArray)
                )  # Append the current detection to the detections array

    results_tracker = tracker.update(
        detections
    )  # Update the tracker with the current detections

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in results_tracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        print(result)
        w = x2 - x1
        h = y2 - y1
        cvzone.cornerRect(
            img,
            (x1, y1, w, h),
            l=8,
            rt=2,
            colorR=(255, 0, 0),
            colorC=(0, 255, 255),
        )
        cvzone.putTextRect(
            img,
            f"{int(id)}",
            (max(0, x1), max(35, y1)),
            scale=1,
            thickness=2,
            offset=5,
        )

    cv2.imshow("Car Detection", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
