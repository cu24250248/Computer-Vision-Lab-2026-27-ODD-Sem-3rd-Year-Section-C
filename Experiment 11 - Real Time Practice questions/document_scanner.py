"""
TASK 5: Smart Document Scanner with Perspective Transform (CamScanner Style)
- Ultra-fast, zero-lag document boundary detection
- Fallback 4-corner extreme finder (failsafe: paper wrinkled ya shadows ho tab bhi detect hoga)
- Multi-mode scanning: [1] Crisp B/W  [2] Color Enhanced  [3] Grayscale
- Keys: 's' = Save Scan, 'm' = Toggle Mode, 'q' = Quit
"""
import cv2
import numpy as np
import os
from datetime import datetime
from camera import Camera, wait_for_frame
from config import CAMERA_SOURCE
from recorder import OutputRecorder

recorder = OutputRecorder("05_document_scanner")
OUT_DIR = os.path.join(recorder.dir, "scans")
os.makedirs(OUT_DIR, exist_ok=True)

def order_points(pts):
    """Sorts 4 coordinates in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]       # top-left (min sum x+y)
    rect[2] = pts[np.argmax(s)]       # bottom-right (max sum x+y)
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]       # top-right (min diff y-x)
    rect[3] = pts[np.argmax(d)]       # bottom-left (max diff y-x)
    return rect

def get_extreme_corners(contour):
    """Fallback method: finds the 4 corner points of any quadrilateral contour."""
    pts = contour.reshape(-1, 2)
    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 1] - pts[:, 0]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype="float32")

def four_point_transform(image, pts):
    """Performs perspective transform to un-warp the document."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    if maxWidth < 50 or maxHeight < 50:
        return None

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def apply_scan_filter(warped, mode):
    """Applies document enhancement filter."""
    if warped is None:
        return None

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    if mode == 1:
        # Crisp B/W (Adaptive Gaussian Threshold)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        scanned = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 6
        )
        return cv2.cvtColor(scanned, cv2.COLOR_GRAY2BGR)

    elif mode == 2:
        # CamScanner Color Enhance (Detail + Contrast boost)
        lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    else:
        # Crisp Grayscale
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

print("Connecting to camera...")
cam = Camera(CAMERA_SOURCE).start()
ok, _ = wait_for_frame(cam)
if not ok:
    raise SystemExit("Frame nahi mila. config.py check karein.")

scan_mode = 1
MODE_NAMES = {1: "Crisp B/W", 2: "Color Enhanced", 3: "Grayscale"}
save_msg_time = 0
last_save_path = ""

print("\nDocument Scanner chalu... Document camera ke saamne rakhein.")
print("Keys: 's' = Save Scan | 'm' = Switch Filter Mode | 'q' = Quit\n")

while True:
    ret, frame = cam.read()
    if not ret or frame is None:
        continue

    h_orig, w_orig = frame.shape[:2]
    display = frame.copy()

    # Preprocessing at fixed 480p height for high-speed edge detection
    proc_h = 480
    ratio = h_orig / float(proc_h)
    proc_w = int(w_orig / ratio)
    resized = cv2.resize(frame, (proc_w, proc_h))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 60, 180)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    doc_pts = None

    for c in contours:
        area = cv2.contourArea(c)
        if area < (proc_w * proc_h * 0.08):  # Must cover at least 8% of the frame
            continue

        peri = cv2.arcLength(c, True)

        # Multi-epsilon attempt (0.015 se 0.05 tak)
        found_4 = False
        for eps_factor in [0.02, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(c, eps_factor * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                doc_pts = approx.reshape(4, 2) * ratio
                found_4 = True
                break

        if found_4:
            break

        # Fallback: if area is large enough, get extreme 4 corners
        if area > (proc_w * proc_h * 0.15):
            doc_pts = get_extreme_corners(c) * ratio
            break

    warped_filtered = None

    if doc_pts is not None:
        pts_int = doc_pts.astype(int)
        # Draw glowing green boundary
        cv2.polylines(display, [pts_int], True, (0, 255, 0), 3)
        for i, pt in enumerate(pts_int):
            cv2.circle(display, tuple(pt), 7, (0, 0, 255), -1)
            cv2.circle(display, tuple(pt), 9, (255, 255, 255), 2)

        warped = four_point_transform(frame, doc_pts)
        if warped is not None:
            warped_filtered = apply_scan_filter(warped, scan_mode)

            # Picture-in-picture preview
            prev_w = 260
            prev_h = int(prev_w * 1.35)
            if prev_h < display.shape[0] - 80 and prev_w < display.shape[1] - 40:
                preview_img = cv2.resize(warped_filtered, (prev_w, prev_h))
                # Border box
                x1 = display.shape[1] - prev_w - 20
                y1 = 20
                cv2.rectangle(display, (x1 - 3, y1 - 3), (x1 + prev_w + 3, y1 + prev_h + 3), (0, 255, 255), 3)
                display[y1:y1 + prev_h, x1:x1 + prev_w] = preview_img
                cv2.putText(display, f"SCAN PREVIEW ({MODE_NAMES[scan_mode]})",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    else:
        cv2.putText(display, "Document camera ke saamne rakhein...", (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    # UI Banner
    header = f"Mode: {MODE_NAMES[scan_mode]} ('m' to switch) | 's' = Save | 'q' = Quit"
    cv2.putText(display, header, (20, display.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

    # Temporary Save notification
    if datetime.now().timestamp() - save_msg_time < 3.0:
        cv2.rectangle(display, (20, 60), (display.shape[1] - 20, 110), (0, 200, 0), -1)
        cv2.putText(display, f"DOCUMENT SAVED: {os.path.basename(last_save_path)}",
                    (35, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    recorder.record_frame(display)
    cv2.imshow("Task 5 - Smart Document Scanner", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('m'):
        scan_mode = 1 if scan_mode >= 3 else scan_mode + 1
        print(f"[MODE CHANGED] Now using: {MODE_NAMES[scan_mode]}")
    elif key == ord('s') and warped_filtered is not None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        last_save_path = os.path.join(OUT_DIR, f"scan_{ts}.png")
        cv2.imwrite(last_save_path, warped_filtered)
        save_msg_time = datetime.now().timestamp()
        print(f"[DOCUMENT SAVED] -> {last_save_path}")
    elif key == ord('q'):
        break

cam.stop()
recorder.close()
cv2.destroyAllWindows()
