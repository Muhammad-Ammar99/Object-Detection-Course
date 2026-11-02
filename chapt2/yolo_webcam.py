import cv2
from ultralytics import YOLO
import cvzone
import math

# cap = cv2.VideoCapture(0) for webcam
# cap.set(3, 1280)
# cap.set(4, 720)

cap = cv2.VideoCapture("videos/traffic.mp4")  # for videos

window_name = "Image"
cv2.namedWindow(window_name)

model = YOLO("yolo_weights/yolov8n.pt")

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

while cap.isOpened():
    success, img = cap.read()

    if not success:
        break

    height, width = img.shape[:2]  # For video resizing
    scale = min(1280 / width, 720 / height)  # For video resizing
    new_width = int(width * scale)  # For video resizing
    new_height = int(height * scale)  # For video resizing
    img = cv2.resize(img, (new_width, new_height))  # For video resizing

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # normal Box in xyxy format

            # print(x1, y1, x2, y2)
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # normal Box in xyxy format
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) # normal Box in xyxy format

            # x1, y1, w, h = box.xywh[0]  # bounding Box in xywh format
            w, h = int(x2 - x1), int(y2 - y1)  # bounding Box in xywh format
            bbox = int(x1), int(y1), int(w), int(h)  # bounding Box in xywh format

            cvzone.cornerRect(img, bbox)  # bounding Box format

            conf = math.ceil((box.conf[0] * 100)) / 100
            print(conf)

            cls = int(box.cls[0])
            cvzone.putTextRect(
                img,
                f"{conf} {classNames[cls]}",
                (max(0, bbox[0]), max(35, bbox[1])),
                scale=0.7,
                thickness=1,
            )

    cv2.imshow(window_name, img)

    cv2.waitKey(1)

    # Stop if the X button is clicked
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
