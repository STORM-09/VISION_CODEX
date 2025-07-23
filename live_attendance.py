import cv2
import os
import csv
import time
import pickle
import face_recognition
from datetime import datetime

# Load trained model
with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

video = cv2.VideoCapture(0)

attendance_dir = "Attendance"
os.makedirs(attendance_dir, exist_ok=True)

known_names = set()
threshold = 0.6  # Adjust based on accuracy requirements

print("Face recognition started... Press SPACE to mark attendance, Q to quit.")

while True:
    ret, frame = video.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, faces)

    for (top, right, bottom, left), face_enc in zip(faces, encodings):
        probs = model.predict_proba([face_enc])[0]
        max_prob = max(probs)
        predicted_name = model.classes_[probs.argmax()] if max_prob > threshold else "Unknown"

        # Draw box and name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, top - 30), (right, top), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, predicted_name, (left + 6, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        timestamp = datetime.now().strftime("%H:%M:%S")
        date = datetime.now().strftime("%d-%m-%Y")
        csv_path = os.path.join(attendance_dir, f"Attendance_{date}.csv")

        if predicted_name != "Unknown" and cv2.waitKey(1) == 32:  # Spacebar
            if predicted_name not in known_names:
                known_names.add(predicted_name)
                file_exists = os.path.exists(csv_path)
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["Name", "Time"])
                    writer.writerow([predicted_name, timestamp])
                print(f"{predicted_name} marked present at {timestamp}")

    cv2.imshow("Face Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
