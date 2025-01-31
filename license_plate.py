import cv2

from ultralytics import solutions

# video_path = 'data/parking1.mp4' # 해외
video_path = 'data/parking1.mp4' # 국내

cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

# detected area
region_points = [(150, 210), (1195, 210), (1195, 535), (150, 535)]

trackzone = solutions.TrackZone(
    show=True,
    region=region_points,
    model='yolo11n.pt',
    line_width=2,
)

while True:
    success, image = cap.read()

    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    image = trackzone.trackzone(image)
    # cv2.imshow('parking system', image)

    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()