"""
TASK 1: Live Motion & Security Alarm Detector (Zero-Lag & Adaptive Background)
- Running weighted background update (eliminates false lighting alarms)
- Bounding box tracking & Red security border flash
- Auto video recording + emergency snapshots in outputs/
"""
import cv2
import os
import time
from datetime import datetime
from camera import Camera, wait_for_frame
from config import CAMERA_SOURCE
from recorder import OutputRecorder

recorder = OutputRecorder("01_motion_alarm")

MIN_AREA = 1200        # Pixel sensitivity
DIFF_THRESHOLD = 25    # Delta threshold
ALERT_COOLDOWN = 2.5   # Cooldown between snapshots
ACCUM_WEIGHT = 0.05    # Background adaptation rate

print("Connecting to camera...")
cam = Camera(CAMERA_SOURCE).start()
ok, first = wait_for_frame(cam)
if not ok:
    raise SystemExit("Frame nahi mila. config.py check karein.")

avg_bg = None
last_alert_time = 0
motion_count = 0

print("\nMotion Detector chalu... 's' = Save snapshot, 'q' = Quit")

while True:
    ret, frame = cam.read()
    if not ret or frame is None:
        continue

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    if avg_bg is None:
        avg_bg = blurred.astype("float")
        continue

    # Update running background
    cv2.accumulateWeighted(blurred, avg_bg, ACCUM_WEIGHT)
    frame_delta = cv2.absdiff(blurred, cv2.convertScaleAbs(avg_bg))

    _, thresh = cv2.threshold(frame_delta, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    for c in contours:
        if cv2.contourArea(c) < MIN_AREA:
            continue
        motion_detected = True
        x, y, cw, ch = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + cw, y + ch), (0, 255, 0), 2)

    if motion_detected:
        motion_count += 1
        cv2.putText(frame, "!! MOTION DETECTED !!", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        # Red warning border
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)

        now = time.time()
        if now - last_alert_time > ALERT_COOLDOWN:
            last_alert_time = now
            recorder.save_snapshot(frame, prefix="motion_alert")

    cv2.putText(frame, f"Motion Events: {motion_count}", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
    cv2.putText(frame, "'s'=Save  'q'=Quit", (w - 220, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    recorder.record_frame(frame)
    cv2.imshow("Task 1 - Motion Detector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        recorder.save_snapshot(frame, prefix="manual_shot")
    elif key == ord('q'):
        break

cam.stop()
recorder.close()
cv2.destroyAllWindows()
