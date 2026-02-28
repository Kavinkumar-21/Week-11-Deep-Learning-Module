import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("mask_detector_model.h5")

# Initialize MediaPipe Face Detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)

# Start camera
cap = cv2.VideoCapture(0)

IMG_SIZE = 128

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape

            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            box_w = int(bboxC.width * w)
            box_h = int(bboxC.height * h)

            # Prevent negative coordinates
            x = max(0, x)
            y = max(0, y)

            face = frame[y:y + box_h, x:x + box_w]

            if face.size == 0:
                continue

            # Preprocess face for model
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face = face / 255.0
            face = np.reshape(face, (1, IMG_SIZE, IMG_SIZE, 3))

            # Predict
            prediction = model.predict(face)[0][0]

            if prediction < 0.5:
                label = f"Mask ({round((1-prediction)*100,2)}%)"
                color = (0, 255, 0)
            else:
                label = f"No Mask ({round(prediction*100,2)}%)"
                color = (0, 0, 255)

            # Draw rectangle & text
            cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2)

    cv2.imshow("Face Mask Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()