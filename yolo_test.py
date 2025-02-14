import cv2

from ultralytics import YOLO

video_path = 'data/parking1.mp4'

model = YOLO('yolo11n.pt')

cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

while True:
    success, frame = cap.read()

    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # detected area
    # frame = trackzone.trackzone(frame)

    # 이건 속도가 빠른데...
    result = model.predict(frame)
    plots = result[0].plot()

    cv2.imshow('yolo test', plots)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()