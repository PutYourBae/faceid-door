import cv2
import face_recognition
import pickle
import time

ENCODINGS_PATH = "encodings/encodings.pickle"
TOLERANCE = 0.45  # Semakin kecil → deteksi lebih ketat

# Load data wajah
with open(ENCODINGS_PATH, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

video = cv2.VideoCapture(0)
print("🎥 Kamera aktif — tekan Q untuk keluar")

while True:
    ret, frame = video.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encs = face_recognition.face_encodings(rgb, boxes)

    for box, enc in zip(boxes, encs):
        matches = face_recognition.compare_faces(known_encodings, enc, tolerance=TOLERANCE)
        name = "Unknown"
        if True in matches:
            idx = matches.index(True)
            name = known_names[idx]

        top, right, bottom, left = box
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if name != "Unknown":
            print(f"✅ {name} dikenali → pintu terbuka (simulasi)")
        else:
            print("❌ Wajah tidak dikenal")

    cv2.imshow("Face ID Door Lock", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
