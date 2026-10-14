"""
TASK 10: Real-Time Augmented Reality (AR) Filter Overlay
- Mediapipe Face Mesh se landmarks
- Virtual glasses + hat render (alpha blending ke saath)
- Auto video recording + AR snapshot capture in outputs/
"""
import cv2
import numpy as np
import math
import mediapipe as mp
from camera import Camera, wait_for_frame
from config import CAMERA_SOURCE
from recorder import OutputRecorder

recorder = OutputRecorder("10_ar_filter")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark IDs
L_EYE_OUT, L_EYE_IN = 33, 133
R_EYE_IN, R_EYE_OUT = 362, 263
FOREHEAD, CHIN = 10, 152

print("Connecting to camera...")
cam = Camera(CAMERA_SOURCE).start()
ok, _ = wait_for_frame(cam)
if not ok:
    raise SystemExit("Frame nahi mila. config.py check karein.")

def pt(lm, idx, w, h):
    return (int(lm[idx].x * w), int(lm[idx].y * h))

def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def draw_glasses(frame, lm, w, h):
    l_out = pt(lm, L_EYE_OUT, w, h)
    l_in = pt(lm, L_EYE_IN, w, h)
    r_in = pt(lm, R_EYE_IN, w, h)
    r_out = pt(lm, R_EYE_OUT, w, h)

    l_center = ((l_out[0] + l_in[0]) // 2, (l_out[1] + l_in[1]) // 2)
    r_center = ((r_in[0] + r_out[0]) // 2, (r_in[1] + r_out[1]) // 2)

    radius = int(max(dist(l_out, l_in), dist(r_in, r_out)) * 0.85)
    radius = max(radius, 12)

    lens_color = (255, 200, 0)
    frame_color = (0, 0, 0)

    # Semi-transparent lenses
    overlay = frame.copy()
    cv2.circle(overlay, l_center, radius, lens_color, -1)
    cv2.circle(overlay, r_center, radius, lens_color, -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    # Frames
    cv2.circle(frame, l_center, radius, frame_color, 4)
    cv2.circle(frame, r_center, radius, frame_color, 4)
    cv2.line(frame, l_center, r_center, frame_color, 4)

    # Temple arms
    cv2.line(frame, (l_center[0] - radius, l_center[1]),
             (l_center[0] - int(radius * 1.6), l_center[1] - int(radius * 0.3)), frame_color, 4)
    cv2.line(frame, (r_center[0] + radius, r_center[1]),
             (r_center[0] + int(radius * 1.6), r_center[1] - int(radius * 0.3)), frame_color, 4)

def draw_hat(frame, lm, w, h):
    forehead = pt(lm, FOREHEAD, w, h)
    chin = pt(lm, CHIN, w, h)
    face_h = dist(forehead, chin)

    cx, cy = forehead
    r = int(face_h * 0.42)
    base_y = cy - int(face_h * 0.05)

    hat_color = (0, 0, 180)     # dark blue hat
    band_color = (0, 215, 255)  # yellow band

    # Crown
    pts = np.array([
        [cx - r, base_y],
        [cx - int(r * 0.75), base_y - int(r * 0.9)],
        [cx, base_y - int(r * 1.3)],
        [cx + int(r * 0.75), base_y - int(r * 0.9)],
        [cx + r, base_y]
    ], np.int32)
    cv2.fillPoly(frame, [pts], hat_color)

    # Band
    cv2.rectangle(frame, (cx - r, base_y - int(r * 0.22)),
                  (cx + r, base_y + int(r * 0.08)), band_color, -1)

    # Brim
    cv2.ellipse(frame, (cx, base_y + int(r * 0.05)),
                (int(r * 1.35), int(r * 0.32)), 0, 0, 180, hat_color, -1)

print("AR Filter chalu... 'g' = glasses ON/OFF, 'h' = hat ON/OFF, 's' = Save snapshot, 'q' = quit")

show_glasses = True
show_hat = True

while True:
    ret, frame = cam.read()
    if not ret:
        continue

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            lm = face_landmarks.landmark
            if show_glasses:
                draw_glasses(frame, lm, w, h)
            if show_hat:
                draw_hat(frame, lm, w, h)
        cv2.putText(frame, "FACE DETECTED - AR ON", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Chehra camera ke saamne laayein...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.putText(frame, "'g'=Glasses 'h'=Hat 's'=Save 'q'=Quit", (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    recorder.record_frame(frame)

    cv2.imshow("Task 10 - AR Filter Overlay", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('g'):
        show_glasses = not show_glasses
    elif key == ord('h'):
        show_hat = not show_hat
    elif key == ord('s'):
        recorder.save_snapshot(frame, prefix="ar_filter")
    elif key == ord('q'):
        break

cam.stop()
recorder.close()
cv2.destroyAllWindows()
