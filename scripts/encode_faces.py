import face_recognition
import cv2
import os
import pickle

DATASET_DIR = "dataset"
OUTPUT = "encodings/encodings.pickle"

known_encodings = []
known_names = []

for person_name in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue
    print(f"[INFO] Processing {person_name}...")
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)
        for enc in encodings:
            known_encodings.append(enc)
            known_names.append(person_name)

os.makedirs("encodings", exist_ok=True)
with open(OUTPUT, "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)

print(f"✅ All encodings saved to {OUTPUT}")
