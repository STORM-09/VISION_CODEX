import cv2
import pickle
import os
import face_recognition

# === Setup ===
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

name = input("Enter your name: ").strip()
face_embeddings = []
count = 0

print(f"Collecting data for: {name}...")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_crop = frame[y:y + h, x:x + w]
        rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        rgb_face = cv2.resize(rgb_face, (150, 150))

        # Get encoding
        encodings = face_recognition.face_encodings(rgb_face)
        if encodings:
            face_embeddings.append(encodings[0])
            count += 1

        # Draw progress
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.putText(frame, f"Samples: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 50), 2)

    cv2.imshow("Registering Face", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 100:
        break

video.release()
cv2.destroyAllWindows()

# === Save Data ===
if not os.path.exists("data"):
    os.makedirs("data")

# Load existing data if available
if os.path.exists("data/embeddings.pkl"):
    with open("data/embeddings.pkl", "rb") as f:
        all_embeddings = pickle.load(f)
    with open("data/names.pkl", "rb") as f:
        all_names = pickle.load(f)
else:
    all_embeddings = []
    all_names = []

# Add new data
all_embeddings.extend(face_embeddings)
all_names.extend([name] * len(face_embeddings))

# Save back to disk
with open("data/embeddings.pkl", "wb") as f:
    pickle.dump(all_embeddings, f)
with open("data/names.pkl", "wb") as f:
    pickle.dump(all_names, f)

print(f"✅ Successfully registered {name} with {len(face_embeddings)} face embeddings.")
