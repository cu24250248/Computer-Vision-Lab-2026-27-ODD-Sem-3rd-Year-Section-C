"""
TASK 7: Virtual Whiteboard / Air Painting (AI Finger & Color Tracking)
- Index finger tip se hawa me draw karein (Mediapipe AI Hand Tracking)
- Colors: Blue, Green, Red, Yellow, Eraser (Top palette bar)
- Two fingers = Selection Mode | One index finger = Drawing Mode
- Keys: 'c' = Clear, 's' = Save, 'q' = Quit
"""
import cv2
import numpy as np
import os
import time
from datetime import datetime
import mediapipe as mp
from camera import Camera, wait_for_frame
from config import CAMERA_SOURCE
from recorder import OutputRecorder

recorder = OutputRecorder("07_air_painting")

# Mediapipe hands setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Color Palette definition (BGR)
COLORS = [
    ("BLUE",   (255, 0, 0)),
    ("GREEN",  (0, 255, 0)),
    ("RED",    (0, 0, 255)),
    ("YELLOW", (0, 255, 255)),
    ("ERASER", (0, 0, 0))
]
current_color = (255, 0, 0)
brush_thickness = 8
eraser_thickness = 45

print("Connecting to camera...")
cam = Camera(CAMERA_SOURCE).start()
ok, first = wait_for_frame(cam)
if not ok:
    raise SystemExit("Frame nahi mila. config.py check karein.")

canvas = None
prev_x, prev_y = 0, 0

def draw_palette(img):
    """Draws top interactive color selection buttons."""
    h_box = 60
    w_box = img.shape[1] // len(COLORS)
    for i, (name, col) in enumerate(COLORS):
        x1 = i * w_box
        x2 = (i + 1) * w_box
        cv2.rectangle(img, (x1, 0), (x2, h_box), col, -1)
        cv2.rectangle(img, (x1, 0), (x2, h_box), (255, 255, 255), 2)
        text_col = (255, 255, 255) if col == (0, 0, 0) else (0, 0, 0)
        cv2.putText(img, name, (x1 + 15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_col, 2)
    return h_box

print("\nAir Painting chalu... Index finger se hawa me likhein!")
print("2 Ungliyan = Color Chunein | 1 Ungli (Index) = Draw Karein | 'c' = Clear, 's' = Save\n")

while True:
    ret, frame = cam.read()
    if not ret or frame is None:
        continue

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    if canvas is None or canvas.shape[:2] != (h, w):
        canvas = np.zeros_like(frame)

    palette_h = draw_palette(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_face_landmarks if hasattr(results, 'multi_face_landmarks') else None:
        pass

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            lm = hand_lms.landmark
            # Coordinates of index tip (8) and middle tip (12)
            ix, iy = int(lm[8].x * w), int(lm[8].y * h)
            mx, my = int(lm[12].x * w), int(lm[12].y * h)

            # Check if index and middle fingers are up
            index_up = lm[8].y < lm[6].y
            middle_up = lm[12].y < lm[10].y

            # 1) Selection Mode: Both Index & Middle are UP
            if index_up and middle_up:
                prev_x, prev_y = 0, 0
                cv2.circle(frame, (ix, iy), 15, (0, 255, 255), 3)
                cv2.putText(frame, "SELECTION MODE", (ix + 20, iy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Check if touching palette bar at the top
                if iy < palette_h:
                    box_idx = ix // (w // len(COLORS))
                    if 0 <= box_idx < len(COLORS):
                        current_color = COLORS[box_idx][1]

            # 2) Drawing Mode: Only Index is UP
            elif index_up and not middle_up:
                is_eraser = (current_color == (0, 0, 0))
                th = eraser_thickness if is_eraser else brush_thickness
                circle_col = (255, 255, 255) if is_eraser else current_color

                cv2.circle(frame, (ix, iy), th // 2, circle_col, -1)
                cv2.circle(frame, (ix, iy), th // 2 + 2, (0, 0, 0), 2)

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = ix, iy

                cv2.line(canvas, (prev_x, prev_y), (ix, iy), current_color, th)
                prev_x, prev_y = ix, iy
            else:
                prev_x, prev_y = 0, 0
    else:
        prev_x, prev_y = 0, 0

    # Merge Canvas with live camera feed
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    composite = cv2.bitwise_and(frame, inv)
    composite = cv2.bitwise_or(composite, canvas)

    cv2.putText(composite, "'c'=Clear  's'=Save  'q'=Quit", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    recorder.record_frame(composite)
    cv2.imshow("Task 7 - Air Painting", composite)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        canvas = np.zeros_like(frame)
        print("[CLEARED] Canvas clear ho gaya.")
    elif key == ord('s'):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        c_path = os.path.join(recorder.dir, f"artwork_{ts}.png")
        cv2.imwrite(c_path, composite)
        print(f"[ARTWORK SAVED] -> {c_path}")
    elif key == ord('q'):
        if canvas is not None and np.any(canvas > 0):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            c_path = os.path.join(recorder.dir, f"artwork_{ts}.png")
            cv2.imwrite(c_path, composite)
        break

cam.stop()
recorder.close()
cv2.destroyAllWindows()
