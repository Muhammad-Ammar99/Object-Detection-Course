import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

window_name = "Image"
cv2.namedWindow(window_name)

while cap.isOpened():
    success, img = cap.read()

    if not success:
        break

    cv2.imshow(window_name, img)

    cv2.waitKey(1)

    # Stop if the X button is clicked
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
