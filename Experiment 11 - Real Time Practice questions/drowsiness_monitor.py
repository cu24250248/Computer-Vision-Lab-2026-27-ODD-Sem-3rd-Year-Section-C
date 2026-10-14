"""
TASK 9: Mobile Driver Drowsiness / Blink Monitor
- Mediapipe Face Mesh -> EAR (Eye Aspect Ratio)
- Aankh band rahi to PC speaker pe alarm
- Auto video recording + drowsiness event logging & snapshots in outputs/
"""
import cv2
import csv
import os
import time
import threading
import math
from datetime import datetime
import mediapipe as mp
from camera import Camera, wait_for_frame
from config import CAMERA_SOURCE
from recorder import OutputRecorder

recorder = OutputRecorder("09_drowsiness_monitor")
LOG_FILE = os.path.join(recorder.dir, "drowsiness_log.csv")

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["timestamp", "event", "ear_val", "blink_count"])

# ---- Alarm (Windows) ----
try:
    import winsound
    def play_alarm():
        try:
            for _ in range(3):
                winsound.Beep(2500, 300)
                time.sleep(0.08)
        except Exception:
            print("\a[ALARM] DROWSINESS DETECTED!")
except ImportError:
    def play_alarm():
        print("\a[ALARM] DROWSINESS DETECTED!")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# EAR landmark indices (Mediapipe FaceMesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.22   # isse neeche = aankh band
CLOSED_SECONDS = 2.0   # itni der band rahi to alarm

print("Connecting to camera...")
cam = Camera(CAMERA_SOURCE).start()
ok, _ = wait_for_frame(cam)
if not ok:
    raise SystemExit("Frame nahi mila. config.py check karein.")

eye_closed_start = None
blink_count = 0
was_closed = False
alarm_playing = False
last_alert_snap_time = 0
status = "AWAKE"
status_color = (0, 255, 0)

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    v1 = euclidean(pts[1], pts[5])
    v2 = euclidean(pts[2], pts[4])
    hz = euclidean(pts[0], pts[3])
    if hz == 0:
        return 0.0
    return (v1 + v2) / (2.0 * hz)

def _sound_worker():
    global alarm_playing
    play_alarm()
    alarm_playing = False

print("Drowsiness Monitor chalu... 's' = snapshot, 'q' = quit")

while True:
    ret, frame = cam.read()
    if not ret:
        continue

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        left_ear = eye_aspect_ratio(lm, LEFT_EYE, w, h)
        right_ear = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
        ear = (left_ear + right_ear) / 2.0

        for eye in (LEFT_EYE, RIGHT_EYE):
            pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in eye]
            for p in pts:
                cv2.circle(frame, p, 2, (0, 255, 255), -1)

        if ear < EAR_THRESHOLD:
            if eye_closed_start is None:
                eye_closed_start = time.time()
            closed_dur = time.time() - eye_closed_start

            if closed_dur >= CLOSED_SECONDS:
                status = "DROWSY! WAKE UP!"
                status_color = (0, 0, 255)
                if not alarm_playing:
                    alarm_playing = True
                    threading.Thread(target=_sound_worker, daemon=True).start()

                now = time.time()
                if now - last_alert_snap_time > 3.0:
                    last_alert_snap_time = now
                    recorder.save_snapshot(frame, prefix="drowsy_alert")
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([ts, "DROWSY_ALERT", f"{ear:.3f}", blink_count])

                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)
            else:
                status = f"EYES CLOSED ({closed_dur:.1f}s)"
                status_color = (0, 165, 255)
            was_closed = True
        else:
            if was_closed:
                blink_count += 1
                was_closed = False
            eye_closed_start = None
            status = "AWAKE"
            status_color = (0, 255, 0)

        cv2.putText(frame, f"EAR: {ear:.3f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.putText(frame, f"Blinks: {blink_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    else:
        cv2.putText(frame, "Face detect karein...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.putText(frame, status, (20, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, status_color, 3)
    cv2.putText(frame, "'s'=Save  'q'=Quit", (frame.shape[1] - 220, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    recorder.record_frame(frame)

    cv2.imshow("Task 9 - Drowsiness Monitor (q to quit)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        recorder.save_snapshot(frame, prefix="manual_shot")
    elif key == ord('q'):
        break

cam.stop()
recorder.close()
cv2.destroyAllWindows()
