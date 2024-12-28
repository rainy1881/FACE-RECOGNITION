import cv2
import numpy as np
import pickle
import os
from datetime import datetime

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.database_file = "face_database.pkl"
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.setThreshold(65)
        self.load_database()

    def load_database(self):
        if os.path.exists(self.database_file):
            with open(self.database_file, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']
            if self.known_face_encodings:
                self.recognizer.read('trainer.yml')

    def save_database(self):
        with open(self.database_file, 'wb') as f:
            pickle.dump({
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }, f)

    def preprocess_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=7,
            minSize=(60, 60)
        )
        return gray, faces

    def register_new_face(self, name):
        cap = cv2.VideoCapture(0)
        face_samples = []
        count = 0
        
        while count < 50:
            ret, frame = cap.read()
            if not ret:
                continue

            gray, faces = self.preprocess_face(frame)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                if len(faces) == 1:
                    face_roi = gray[y:y+h, x:x+w]
                    face_roi = cv2.equalizeHist(face_roi)
                    face_samples.append(face_roi)
                    count += 1
                    
            cv2.putText(frame, f"Capturing: {count}/50", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Register New Face - Keep still', frame)
            
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if face_samples:
            face_id = len(self.known_face_names)
            self.known_face_names.append(name)
            
            faces = []
            labels = []
            for face in face_samples:
                face = cv2.resize(face, (200, 200))
                face = cv2.equalizeHist(face)
                faces.append(face)
                labels.append(face_id)

            self.known_face_encodings.append(face_id)
            self.recognizer.train(faces, np.array(labels))
            self.recognizer.save('trainer.yml')
            self.save_database()
            print(f"Training complete for {name}")
        else:
            print("No face samples collected")

    def start_recognition(self):
        if not self.known_face_names:
            print("No faces registered yet. Please register at least one face.")
            return

        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            gray, faces = self.preprocess_face(frame)

            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.equalizeHist(face_roi)
                face_roi = cv2.resize(face_roi, (200, 200))
                
                try:
                    label, confidence = self.recognizer.predict(face_roi)
                    
                    if confidence < 65:
                        name = self.known_face_names[label]
                        confidence_text = f"Confidence: {100-confidence:.1f}%"
                        color = (0, 255, 0)
                    else:
                        name = "Unknown"
                        confidence_text = "Low confidence"
                        color = (0, 0, 255)
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    cv2.putText(frame, name, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    cv2.putText(frame, confidence_text, (x, y+h+30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                except:
                    pass

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows() 