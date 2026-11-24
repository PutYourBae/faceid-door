# 🔐 Face ID Door Lock

A simple face recognition system that simulates a smart door lock — only recognized faces trigger an "unlock" event.

![Python](https://img.shields.io/badge/python-3.x-blue?style=flat-square)
![face-recognition](https://img.shields.io/badge/face--recognition-dlib-orange?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-camera-green?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square)

---

## ✨ Features

- Enroll up to 4 persons (Person 1–4)
- Real-time face recognition via webcam
- Simulated door unlock when a known face is detected
- Unknown faces trigger an "ACCESS DENIED" response
- Lightweight — no cloud, fully local

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| face_recognition | Face encoding & matching (dlib) |
| OpenCV | Camera capture & display |
| Python 3.x | Core language |

---

## 📂 Project Structure

```
faceid-door/
├── dataset/
│   ├── person_1/     # 20–50 photos of Person 1
│   ├── person_2/
│   ├── person_3/
│   └── person_4/
├── scripts/
│   ├── encode_faces.py           # Build face encodings from dataset
│   └── recognize_and_unlock.py   # Real-time recognition script
├── encodings/
│   └── encodings.pkl             # Saved face encodings (auto-generated)
└── README.md
```

---

## ⚙️ Setup & Run

### 1. Install Dependencies

```bash
pip install face_recognition opencv-python
```

> **Note:** `face_recognition` requires `dlib`. On Windows, install via:
> ```bash
> pip install cmake dlib face_recognition
> ```

### 2. Prepare Dataset

Create the folder structure and add **20–50 photos per person**:

```
dataset/
├── person_1/   ← add photos here (jpg/png)
├── person_2/
├── person_3/
└── person_4/
```

### 3. Encode Faces

```bash
python scripts/encode_faces.py
```

Output: `encodings/encodings.pkl`

### 4. Run Recognition

```bash
python scripts/recognize_and_unlock.py
```

The camera opens and starts recognizing faces in real time.  
Press **Q** to quit.

---

## 📸 System Flow

```
Camera Frame
     ↓
Detect face locations
     ↓
Compute face encodings
     ↓
Compare with known encodings
     ↓
Match found?  → "UNLOCKED ✅ — Welcome, [Name]!"
No match?     → "ACCESS DENIED ❌"
```

---

## 📊 Performance

| Condition | Accuracy |
|-----------|---------|
| Good lighting, frontal face | ~95% |
| Side angle or low light | ~75–85% |

> Performance depends on dataset quality and number of training photos.

---

## 🔧 Tips for Better Accuracy

- Use **varied angles** (front, slight left/right, tilted) in your dataset photos
- Ensure **good, consistent lighting** in your photos
- Add at least **30 photos per person** for reliable recognition

---

## 📄 License

MIT License — free to use and modify.
