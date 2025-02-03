import cv2

from ultralytics import solutions, YOLO
from sort.sort import *
from util import *

video_path = 'data/parking1.mp4'

mot_tracker = Sort()

coco_model = YOLO('yolo11n.pt')
license_plate_detector = YOLO('./model/best.pt')

vehicles = [2,3,5,7] # car, motorcycle, bus, truck

cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))

# track zone init
# region_points = [(150, 210), (1195, 210), (1195, 535), (150, 535)]
# trackzone = solutions.TrackZone(
#     show=True,
#     region=region_points,
#     model='yolo11n.pt',
#     line_width=2,
# )

while True:
    success, frame = cap.read()

    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # detected area
    # frame = trackzone.trackzone(frame)

    detections = coco_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1,y1,x2,y2,score,class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1,y1,x2,y2,score])

    # track vehicles
    track_ids = mot_tracker.update(np.asarray(detections_))

    # detect license plates
    license_plates = license_plate_detector(frame)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # assign license plate to car
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

        if car_id != -1:
            # crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

            H, W, _ = license_plate_crop.shape

            # process license plate
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            # read license plate number
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
            # print(license_plate_text, license_plate_text_score)

            if license_plate_text is not None:
                cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), color=(255,0,0), thickness=3)

            # cv2.imshow('license_plate_crop', license_plate_crop)

    cv2.imshow('parking system', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()