"""
TASK 8: Real-Time OCR Text Reader (Zero-Freeze Threaded OCR)
- Non-blocking background worker se text extraction (camera 100% lag-free)
- Overlay recognised text & word bounding boxes
- Text log file save in outputs/
"""
import cv2
import pytesseract
import os
import time
import threading
from datetime import datetime
from camera import Camera, wait_for_frame
from config import CAMERA_SOURCE
from recorder import OutputRecorder

recorder = OutputRecorder("08_ocr_reader")
TEXT_LOG_FILE = os.path.join(recorder.dir, "ocr_extracted_text.txt")

# Check Tesseract candidates
TESSERACT_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
]

tesseract_found = False
for path in TESSERACT_CANDIDATES:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        tesseract_found = True
        break

if not tesseract_found:
    print("\n" + "=" * 60)
    print("[WARN] TESSERACT OCR NOT FOUND AT DEFAULT PATH:")
    print("   C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
    print("Download and install from:")
    print("   https://github.com/UB-Mannheim/tesseract/wiki")
    print("=" * 60 + "\n")

print("Connecting to camera...")
cam = Camera(CAMERA_SOURCE).start()
ok, _ = wait_for_frame(cam)
if not ok:
    raise SystemExit("Frame nahi mila. config.py check karein.")

ocr_boxes = []
ocr_text = ""
ocr_running = False
logged_texts = set()

def ocr_worker(img_crop):
    """Background thread to process OCR without blocking camera stream."""
    global ocr_boxes, ocr_text, ocr_running
    try:
        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 25, 12
        )
        data = pytesseract.image_to_data(
            thresh, output_type=pytesseract.Output.DICT,
            config="--psm 6"
        )
        new_boxes = []
        words = []
        n = len(data["text"])
        for i in range(n):
            txt = data["text"][i].strip()
            conf = int(float(data["conf"][i])) if data["conf"][i] != "-1" else -1
            if txt and conf > 50:
                bx, by, bw, bh = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
                new_boxes.append((bx, by, bw, bh, txt))
                words.append(txt)

        ocr_boxes = new_boxes
        ocr_text = " ".join(words)

        if ocr_text and ocr_text not in logged_texts:
            logged_texts.add(ocr_text)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(TEXT_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {ocr_text}\n")
            print(f"[OCR EXTRACTED] {ocr_text}")
    except Exception as e:
        pass
    finally:
        ocr_running = False

print("\nOCR Reader chalu... Text par camera point karein.")
print("Keys: 's' = Save Snapshot | 'q' = Quit\n")

frame_idx = 0

while True:
    ret, frame = cam.read()
    if not ret or frame is None:
        continue

    frame_idx += 1
    h, w = frame.shape[:2]

    # Trigger OCR asynchronously every 15 frames if not already running
    if tesseract_found and not ocr_running and (frame_idx % 12 == 0):
        ocr_running = True
        threading.Thread(target=ocr_worker, args=(frame.copy(),), daemon=True).start()

    # Draw word boxes
    for (bx, by, bw, bh, txt) in ocr_boxes:
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

    # Text overlay bar
    if not tesseract_found:
        cv2.putText(frame, "Install Tesseract OCR to read text",
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif ocr_text:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
        cv2.putText(frame, "TEXT: " + ocr_text[:75], (15, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        if len(ocr_text) > 75:
            cv2.putText(frame, ocr_text[75:150], (15, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    cv2.putText(frame, "'s'=Save  'q'=Quit", (w - 220, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    recorder.record_frame(frame)
    cv2.imshow("Task 8 - Real-Time OCR Reader", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        recorder.save_snapshot(frame, prefix="ocr_snapshot")
    elif key == ord('q'):
        break

cam.stop()
recorder.close()
cv2.destroyAllWindows()
