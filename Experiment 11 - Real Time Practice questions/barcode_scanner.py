"""
TASK 2: Mobile QR Code & Barcode Scanner for Inventory
- cv2.QRCodeDetector for QR
- pyzbar for barcodes (optional)
- CSV logging with timestamp in outputs/
- Auto video recording + snapshot saves
"""
import cv2
import csv
import os
import time
from datetime import datetime
from camera import Camera, wait_for_frame
from config import CAMERA_SOURCE
from recorder import OutputRecorder

recorder = OutputRecorder("02_qr_barcode")

try:
    from pyzbar import pyzbar
    HAS_PYZBAR = True
except Exception:
    HAS_PYZBAR = False
    print("[INFO] pyzbar nahi mila -> sirf QR scanning chalegi.")

CSV_FILE = os.path.join(recorder.dir, "inventory_log.csv")

# CSV header create karein agar file nahi hai
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["timestamp", "code_type", "decoded_data"])

qr_detector = cv2.QRCodeDetector()

print("Connecting to camera...")
cam = Camera(CAMERA_SOURCE).start()
ok, _ = wait_for_frame(cam)
if not ok:
    raise SystemExit("Frame nahi mila. config.py check karein.")

last_data = ""
last_time = 0
DUPLICATE_COOLDOWN = 2.5  # same code dobara log na ho

def log_item(code_type, data):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([ts, code_type, data])
    print(f"[LOGGED] {ts} | {code_type} | {data}")

print("QR/Barcode Scanner chalu... 's' = snapshot, 'q' = quit")

while True:
    ret, frame = cam.read()
    if not ret:
        continue

    found_this_frame = []

    # ---------- QR CODE ----------
    data, points, _ = qr_detector.detectAndDecode(frame)
    if points is not None and data:
        pts = points.astype(int)
        for i in range(len(pts[0])):
            p1 = tuple(pts[0][i])
            p2 = tuple(pts[0][(i + 1) % len(pts[0])])
            cv2.line(frame, p1, p2, (0, 255, 0), 3)
        cv2.putText(frame, f"QR: {data}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        found_this_frame.append(("QRCODE", data))

    # ---------- BARCODE ----------
    if HAS_PYZBAR:
        for bc in pyzbar.decode(frame):
            (x, y, w, h) = bc.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
            bdata = bc.data.decode("utf-8", errors="ignore")
            cv2.putText(frame, f"{bc.type}: {bdata}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            found_this_frame.append((bc.type, bdata))

    # ---------- LOGGING (with duplicate filter) ----------
    now = time.time()
    for ctype, cdata in found_this_frame:
        if cdata != last_data or (now - last_time) > DUPLICATE_COOLDOWN:
            log_item(ctype, cdata)
            recorder.save_snapshot(frame, prefix=f"scan_{ctype}")
            last_data = cdata
            last_time = now

    cv2.putText(frame, f"Logged in: {CSV_FILE}", (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, "'s'=Save  'q'=Quit", (frame.shape[1] - 220, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    recorder.record_frame(frame)

    cv2.imshow("Task 2 - QR/Barcode Scanner (q to quit)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        recorder.save_snapshot(frame, prefix="manual_shot")
    elif key == ord('q'):
        break

cam.stop()
recorder.close()
cv2.destroyAllWindows()
