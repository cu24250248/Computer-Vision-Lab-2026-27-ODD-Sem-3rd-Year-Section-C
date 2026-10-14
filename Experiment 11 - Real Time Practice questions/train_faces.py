"""
TASK 4 (Part A): Known faces ka LBPH model train karna
Folder: known_faces/<PersonName>/*.jpg
Output: trainer.yml + labels.json
"""
import cv2
import os
import json
import numpy as np

KNOWN_DIR = "known_faces"
MODEL_FILE = "trainer.yml"
LABELS_FILE = "labels.json"

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if not os.path.isdir(KNOWN_DIR):
    os.makedirs(KNOWN_DIR, exist_ok=True)
    raise SystemExit("'{KNOWN_DIR}' folder nahi mila ya khali hai. Har person ka alag folder banayein (e.g., known_faces/Rahul/1.jpg).")

faces = []
labels = []
name_to_id = {}
current_id = 0

for person in sorted(os.listdir(KNOWN_DIR)):
    person_dir = os.path.join(KNOWN_DIR, person)
    if not os.path.isdir(person_dir):
        continue

    name_to_id[person] = current_id
    count = 0

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in rects:
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (200, 200))
            faces.append(roi)
            labels.append(current_id)
            count += 1

    print(f"{person}: {count} face samples mile")
    current_id += 1

if len(faces) == 0:
    raise SystemExit("Koi face sample nahi mila! 'known_faces/<PersonName>/' folder me images (.jpg/.png) dalein.")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.save(MODEL_FILE)

with open(LABELS_FILE, "w") as f:
    json.dump(name_to_id, f, indent=2)

print(f"\n[DONE] Training complete! {len(faces)} samples, {len(name_to_id)} persons")
print(f"   Model  -> {MODEL_FILE}")
print(f"   Labels -> {LABELS_FILE}")
