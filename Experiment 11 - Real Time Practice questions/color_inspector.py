"""
TASK 6: Portable Color & Defect Inspector
- RGB -> HSV
- cv2.inRange se color threshold
- Masking + object detection
- Auto video recording + defect snapshots in outputs/
Trackbars se live tuning karein.
"""
import cv2
import csv
import os
import time
from datetime import datetime
import numpy as np
from camera import Camera, wait_for_frame
from config import CAMERA_SOURCE
from recorder import OutputRecorder

recorder = OutputRecorder("06_color_inspector")
LOG_FILE = os.path.join(recorder.dir, "color_log.csv")

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["timestamp", "color_name", "area", "status"])

def nothing(x):
    pass

WIN = "Task 6 - Color Inspector"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.createTrackbar("H Min", WIN, 100, 179, nothing)
cv2.createTrackbar("H Max", WIN, 130, 179, nothing)
cv2.createTrackbar("S Min", WIN, 100, 255, nothing)
cv2.createTrackbar("S Max", WIN, 255, 255, nothing)
cv2.createTrackbar("V Min", WIN, 80, 255, nothing)
cv2.createTrackbar("V Max", WIN, 255, 255, nothing)
cv2.createTrackbar("Min Area", WIN, 1500, 20000, nothing)

def get_color_name(h, s, v):
    if s < 60:
        return "White/Gray/Black"
    if h < 8 or h > 170:
        return "RED"
    elif h < 22:
        return "ORANGE"
    elif h < 33:
        return "YELLOW"
    elif h < 78:
        return "GREEN"
    elif h < 100:
        return "CYAN"
    elif h < 130:
        return "BLUE"
    elif h < 160:
        return "PURPLE/MAGENTA"
    return "RED"

print("Connecting to camera...")
cam = Camera(CAMERA_SOURCE).start()
ok, _ = wait_for_frame(cam)
if not ok:
    raise SystemExit("Frame nahi mila. config.py check karein.")

print("Color Inspector chalu... Trackbars se HSV tune karein. 's' = snapshot, 'q' = quit")
last_defect_time = 0

while True:
    ret, frame = cam.read()
    if not ret:
        continue

    h_min = cv2.getTrackbarPos("H Min", WIN)
    h_max = cv2.getTrackbarPos("H Max", WIN)
    s_min = cv2.getTrackbarPos("S Min", WIN)
    s_max = cv2.getTrackbarPos("S Max", WIN)
    v_min = cv2.getTrackbarPos("V Min", WIN)
    v_max = cv2.getTrackbarPos("V Max", WIN)
    min_area = cv2.getTrackbarPos("Min Area", WIN)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w // 2, y + h // 2
        h_val, s_val, v_val = hsv[cy, cx]
        cname = get_color_name(int(h_val), int(s_val), int(v_val))

        is_defect = area < min_area * 1.5
        status_txt = "DEFECT" if is_defect else "OK"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(frame, f"{cname} | Area:{int(area)}", (x, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if is_defect:
            cv2.putText(frame, "DEFECT?", (x, y + h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            now = time.time()
            if now - last_defect_time > 4.0:
                last_defect_time = now
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([ts, cname, int(area), status_txt])
                recorder.save_snapshot(frame, prefix=f"defect_{cname}")

    cv2.putText(frame, "'s'=Save Snapshot  'q'=Quit", (frame.shape[1] - 340, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    recorder.record_frame(frame)

    cv2.imshow(WIN, frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        recorder.save_snapshot(frame, prefix="manual_shot")
    elif key == ord('q'):
        break

cam.stop()
recorder.close()
cv2.destroyAllWindows()
