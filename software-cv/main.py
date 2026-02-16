import cv2
import csv
import datetime
from ultralytics import YOLO

model = YOLO('yolov8s.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Initialize CSV file
csv_file = './software-cv/data/detections.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Timestamp', 'Class', 'Confidence', 'Box_X1', 'Box_Y1', 'Box_X2', 'Box_Y2'])

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    for r in results:
        annotated_frame = r.plot()
        cv2.imshow('YOLOv8 Detection', annotated_frame)

        # Extract data for CSV
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Iterate through every detection in the current frame
        for box in r.boxes:
            # Maps class ID to name
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            confidence = float(box.conf[0])

            # Bounding box location
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # Write to file
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([current_time, class_name, confidence, x1, y1, x2, y2])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()