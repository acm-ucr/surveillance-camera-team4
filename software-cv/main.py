import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# 0 for the default webcam.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

print("Press 'q' to quit.")

while True:
    # Take frame from video
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    for r in results:
        # plot() draws the bounding boxes and labels on the image
        annotated_frame = r.plot()
        
        # Display the resulting frame
        cv2.imshow('YOLOv8 Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()