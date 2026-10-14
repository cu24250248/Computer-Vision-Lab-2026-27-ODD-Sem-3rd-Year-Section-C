"""
TASK 4 (Part B): Face Recognition Attendance System
- Haar Cascade se face detect
- LBPH recognizer se pehchan
- CSV me attendance mark (ek din me ek baar)
- Outputs me attendance log + verified photo + video record
"""
import cv2
import os
import json
import csv
from datetime import datetime
from camera import Camera, wait_for_frame
from config import CAMERA_SOURCE
from recorder import OutputRecorder

recorder = OutputRecorder("04_face_attendance")

MODEL_FILE = "trainer.yml"
LABELS_FILE = "labels.json"
ATTENDANCE_FILE = os.path.join(recorder.dir, "attendance.csv")
CONFIDENCE_LIMIT = 75  # LBPH: kam = better match

if not os.path.exists(MODEL_FILE):
    raise SystemExit("Pehle 'python 04_train_faces.py' chalayein!")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_FILE)

with open(LABELS_FILE) as f:
    name_to_id = json.load(f)
id_to_name = {v: k for k, v in name_to_id.items()}

if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["Name", "Date", "Time", "Status"])

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

marked_today = set()

def load_today_marks():
    today = datetime.now().strftime("%Y-%m-%d")
    if not os.path.exists(ATTENDANCE_FILE):
        return
    with open(ATTENDANCE_FILE, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[1] == today:
                marked_today.add(row[0])

load_today_marks()

print("Connecting to camera...")
cam = Camera(CAMERA_SOURCE).start()
ok, _ = wait_for_frame(cam)
if not ok:
    raise SystemExit("Frame nahi mila. config.py check karein.")

print("Attendance System chalu... 's' = snapshot, 'q' = quit")

while True:
    ret, frame = cam.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        roi = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
        user_id, confidence = recognizer.predict(roi)

        if confidence < CONFIDENCE_LIMIT:
            name = id_to_name.get(user_id, "Unknown")
            color = (0, 255, 0)
            label = f"{name} ({int(100 - confidence)}%)"

            if name not in marked_today and name != "Unknown":
                now = datetime.now()
                with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([
                        name,
                        now.strftime("%Y-%m-%d"),
                        now.strftime("%H:%M:%S"),
                        "Present"
                    ])
                marked_today.add(name)
                # Save attendance verification snapshot
                recorder.save_snapshot(frame, prefix=f"present_{name}")
                print(f"[ATTENDANCE MARKED] {name} at {now.strftime('%H:%M:%S')}")
        else:
            name = "Unknown"
            color = (0, 0, 255)
            label = "Unknown"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        cv2.rectangle(frame, (x, y - 35), (x + w, y), color, -1)
        cv2.putText(frame, label, (x + 8, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.putText(frame, f"Present today: {len(marked_today)}", (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, "'s'=Save  'q'=Quit", (frame.shape[1] - 220, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    recorder.record_frame(frame)

    cv2.imshow("Task 4 - Attendance System (q to quit)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        recorder.save_snapshot(frame, prefix="manual_shot")
    elif key == ord('q'):
        break

cam.stop()
recorder.close()
cv2.destroyAllWindows()
