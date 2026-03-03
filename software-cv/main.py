import cv2
import csv
import datetime
import json
import time
import threading
import os
import paho.mqtt.client as mqtt
from ultralytics import YOLO
from flask import Flask, Response

MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "yolo/detections"

app = Flask(__name__)

latest_frame = None
frame_lock = threading.Lock()
# vid_src = "http://172.20.10.6:81/stream"
vid_src = 0

try:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
except AttributeError:
    client = mqtt.Client()

try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
    print(f"Connected to MQTT Broker at {MQTT_BROKER}")
except Exception as e:
    print(f"Failed to connect to MQTT: {e}")
    exit()

# Initialize CSV file
csv_file = './software-cv/data/detections.csv'
os.makedirs(os.path.dirname(csv_file), exist_ok=True)

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Timestamp', 'Class', 'Confidence', 'Box_X1', 'Box_Y1', 'Box_X2', 'Box_Y2'])

def process_video():
    global latest_frame

    model = YOLO('yolov8s.pt')
    cap = cv2.VideoCapture(vid_src)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting video thread.")
            break

        results = model(frame, stream=True)

        for r in results:
            annotated_frame = r.plot()
            
            with frame_lock:
                latest_frame = annotated_frame.copy()

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

                # Publish to MQTT
                client.publish(MQTT_TOPIC, json.dumps(payload))

                # Write to CSV
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([current_time, class_name, confidence, x1, y1, x2, y2])

    cap.release()

def generate_frames():
    global latest_frame

    while True:
        with frame_lock:
            if latest_frame is None:
                # Sleep briefly to wait for the first frame
                time.sleep(0.1)
                continue
            
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.05)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Starting video processing thread...")
    t = threading.Thread(target=process_video, daemon=True)
    t.start()

    print("Starting Flask server... Access stream at http://localhost:5000/video_feed")
    app.run(host='0.0.0.0', port=5000, threaded=True)