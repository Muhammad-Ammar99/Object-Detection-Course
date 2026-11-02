from ultralytics import YOLO
import cv2

model = YOLO("yolo_weights/yolov8l.pt")
results = model("images/3.jpeg")

cv2.imshow("YOLO Object Detection", results[0].plot())

cv2.waitKey(0)
cv2.destroyAllWindows()
