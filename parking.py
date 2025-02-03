import cv2

from ultralytics.solutions import ParkingManagement

# Video capture
# 상공뷰는 학습시켜야 함
filename = "easy1"
# cap = cv2.VideoCapture("data/parking.mp4")
# cap = cv2.VideoCapture("data/parking2.mp4")
cap = cv2.VideoCapture(f"data/{filename}.mp4")
assert cap.isOpened(), "Error reading video file"
# w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Video writer
# video_writer = cv2.VideoWriter("data/parking management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize parking management object
parking_manager = ParkingManagement(
    model="yolo11n.pt",
    json_file=f"boxes/{filename}_bounding_boxes.json",  # path to parking annotations file
)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.resize(image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)

    # imgae = cv2.resize(image, (1020, 500))
    image = parking_manager.process_data(image)

    cv2.imshow('image', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # video_writer.write(image)

cap.release()
# video_writer.release()
cv2.destroyAllWindows()