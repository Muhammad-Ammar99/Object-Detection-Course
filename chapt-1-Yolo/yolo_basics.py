from ultralytics import YOLO
import cv2

model = YOLO("yolo_weights/yolov8l.pt")

results = model("images/3.jpeg")


annotated_image = results[0].plot()

cv2.imshow("YOLO Object Detection", annotated_image)


cv2.waitKey(0)
cv2.destroyAllWindows()
