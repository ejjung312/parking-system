import cv2

filename = "easy1"
# Video capture
cap = cv2.VideoCapture(f"data/{filename}.mp4")
assert cap.isOpened(), "Error reading video file"

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.resize(image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)

    cv2.imshow('image', image)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        print("종료")
        break
    elif key == ord('s'):
        print("저장")
        cv2.imwrite(f"data/{filename}.jpg", image)

cap.release()
cv2.destroyAllWindows()