"""
TASK 3: Hand Gesture Desktop Controller (Ultra-Responsive & Zero-Lag)
- Mediapipe Hands (Rotation-Invariant finger detection)
- Visual countdown debounce progress bar
- PC Actions: Volume / Slides / Play-Pause / Mute
- Auto video recording + action logging in outputs/
"""
import cv2
import csv
import os
import time
import math
from datetime import datetime
import mediapipe as mp
import pyautogui
from camera import Camera, wait_for_frame
from config import CAMERA_SOURCE
from recorder import OutputRecorder

recorder = OutputRecorder("03_hand_gesture")
LOG_FILE = os.path.join(recorder.dir, "gestures_log.csv")

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["timestamp", "fingers", "action"])

pyautogui.FAILSAFE = False

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

print("Connecting to camera...")
cam = Camera(CAMERA_SOURCE).start()
ok, _ = wait_for_frame(cam)
if not ok:
    raise SystemExit("Frame nahi mila. config.py check karein.")

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def count_fingers(lm):
    """Accurate rotation-invariant finger counter."""
    if not lm or len(lm) < 21:
        return -1

    fingers = []

    # 1. Thumb: Check distance of thumb tip (4) from pinky MCP (17) vs thumb IP (3) from pinky MCP (17)
    d_tip_pinky = euclidean(lm[4], lm[17])
    d_ip_pinky = euclidean(lm[3], lm[17])
    fingers.append(1 if d_tip_pinky > d_ip_pinky * 1.15 else 0)

    # 2. Index, Middle, Ring, Pinky: Check tip distance from wrist (0) vs PIP (tip-2) from wrist (0)
    for tip in [8, 12, 16, 20]:
        d_tip_wrist = euclidean(lm[tip], lm[0])
        d_pip_wrist = euclidean(lm[tip - 2], lm[0])
        fingers.append(1 if d_tip_wrist > d_pip_wrist else 0)

    return sum(fingers)

GESTURE_ACTIONS = {
    0: ("MUTE", lambda: pyautogui.press("volumemute")),
    1: ("VOLUME DOWN", lambda: [pyautogui.press("volumedown") for _ in range(2)]),
    2: ("VOLUME UP", lambda: [pyautogui.press("volumeup") for _ in range(2)]),
    3: ("NEXT SLIDE", lambda: pyautogui.press("right")),
    4: ("PREV SLIDE", lambda: pyautogui.press("left")),
    5: ("PLAY / PAUSE", lambda: pyautogui.press("playpause")),
}

last_gesture = -1
stable_count = 0
STABLE_TARGET = 5
last_action_time = 0
ACTION_COOLDOWN = 1.2
last_action_name = ""
action_display_time = 0

print("\nGesture Controller chalu...")
print("0=Mute | 1=Vol- | 2=Vol+ | 3=Next | 4=Prev | 5=Play/Pause | 's'=Snap | 'q'=Quit\n")

while True:
    ret, frame = cam.read()
    if not ret or frame is None:
        continue

    frame = cv2.flip(frame, 1)  # mirror view
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture = -1

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        lm = [[int(pt.x * w), int(pt.y * h)] for pt in hand_landmarks.landmark]
        gesture = count_fingers(lm)

        # Draw finger count badge
        cv2.rectangle(frame, (20, 20), (220, 75), (0, 0, 0), -1)
        cv2.rectangle(frame, (20, 20), (220, 75), (0, 255, 255), 2)
        cv2.putText(frame, f"Fingers: {gesture}", (35, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    # ---------- DEBOUNCE WITH VISUAL PROGRESS BAR ----------
    now = time.time()
    if gesture != -1 and gesture == last_gesture:
        stable_count += 1
    else:
        stable_count = 0

    last_gesture = gesture

    # Progress bar for gesture trigger
    if gesture in GESTURE_ACTIONS and stable_count > 0:
        pct = min(1.0, stable_count / float(STABLE_TARGET))
        bar_w = int(200 * pct)
        cv2.rectangle(frame, (20, 85), (20 + bar_w, 100), (0, 255, 0), -1)
        cv2.rectangle(frame, (20, 85), (220, 100), (255, 255, 255), 1)

    if stable_count == STABLE_TARGET and gesture in GESTURE_ACTIONS and (now - last_action_time) > ACTION_COOLDOWN:
        label, action = GESTURE_ACTIONS[gesture]
        try:
            action()
            last_action_name = label
            action_display_time = now
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([ts, gesture, label])
            recorder.save_snapshot(frame, prefix=f"gesture_{label.replace(' ', '_')}")
            print(f"[ACTION TRIGGERED] -> {label}")
        except Exception as e:
            print("Action error:", e)

        last_action_time = now
        stable_count = 0

    # Display active action banner
    if now - action_display_time < 1.5:
        cv2.rectangle(frame, (20, 115), (350, 165), (0, 200, 0), -1)
        cv2.putText(frame, f"TRIGGERED: {last_action_name}", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    cv2.putText(frame, "'s'=Save Snapshot  'q'=Quit", (w - 320, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    recorder.record_frame(frame)
    cv2.imshow("Task 3 - Gesture Controller", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        recorder.save_snapshot(frame, prefix="manual_shot")
    elif key == ord('q'):
        break

cam.stop()
recorder.close()
cv2.destroyAllWindows()
