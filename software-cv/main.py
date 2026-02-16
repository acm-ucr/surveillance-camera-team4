import cv2
import csv
import datetime
import json
import paho.mqtt.client as mqtt
from ultralytics import YOLO

MQTT_BROKER = "test.mosquitto.org"  # Temp IP
MQTT_PORT = 1883
MQTT_TOPIC = "yolo/detections"

client = mqtt.Client()

try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start() # Start the network loop in a background thread
    print(f"Connected to MQTT Broker at {MQTT_BROKER}")
except Exception as e:
    print(f"Failed to connect to MQTT: {e}")
    exit()

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

        # Extract data for CSV and MQTT
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Iterate through every detection in the current frame
        for box in r.boxes:
            # Maps class ID to name
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            confidence = float(box.conf[0])

            # Bounding box location
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            payload = {
                "timestamp": current_time,
                "class": class_name,
                "confidence": round(confidence, 2),
                "box": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2)
                }
            }

            client.publish(MQTT_TOPIC, json.dumps(payload))

            # Write to file
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([current_time, class_name, confidence, x1, y1, x2, y2])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

client.loop_stop() # Stop the MQTT background thread
client.disconnect()
cap.release()
cv2.destroyAllWindows()