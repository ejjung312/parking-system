1. 자동 번호판 인식 시스템 (ANPR)
설명: YOLO로 차량 번호판을 감지한 후, OpenCV와 Tesseract OCR을 이용해 번호판을 인식하는 프로젝트.
활용: 주차 관리 시스템, 교통 단속.
추가 기능: 차량 모델 및 색상 분류, 블러 처리된 번호판 복원.


아이디어
1) 진입하는 차량 번호판 인식
2) 주차영역 확인
3) 남은 주차공간 확인


1) 차량에서 번호판 감지 (YOLO)
2) 번호판 영역 이미지 전처리 (OpenCV)
3) 번호판의 문자 인식 (OCR)
4) 데이터 저장 및 활용 (DB, 로그 기록, API 연동)

<br/>
다운로드 영상

국내
- https://www.youtube.com/watch?v=GEWAmxiXNUk
- https://www.youtube.com/watch?v=uuPnjlUTmO0

해외
- https://www.youtube.com/watch?v=fyJB1t0o0ms
- https://www.youtube.com/watch?v=HL5s2XOmTY8
- 

데이터셋 링크
- https://www.kaggle.com/datasets/fareselmenshawii/license-plate-dataset
- https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4

github
- https://github.com/computervisioneng/automatic-number-plate-recognition-python-yolov8

--------------------------------------------------------



2. 상품 및 물체 자동 인식 (AI 카메라 스캐너)
설명: YOLO로 여러 상품을 감지하고, OpenCV를 활용해 이미지 전처리 및 OCR 인식을 적용하는 프로젝트.
활용: 무인 편의점, 창고 재고 관리.
추가 기능: 가격 비교 시스템, 바코드 인식 연동.