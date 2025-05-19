import cv2
import numpy as np
import tensorflow as tf
import os

# Load trained model
model = tf.keras.models.load_model('saved_model/asl_alphabet_model.h5')

# Label map: assumes folder names in sorted order (A-Z)
label_map = sorted(os.listdir('asl_alphabet_train'))

# Webcam settings
cap = cv2.VideoCapture(0)
print("\n[INFO] Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define region of interest (ROI)
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    # Preprocess ROI to 64x64 (like training data)
    roi_resized = cv2.resize(roi, (64, 64))
    roi_normalized = roi_resized / 255.0
    roi_reshaped = roi_normalized.reshape(1, 64, 64, 3)

    # Prediction
    preds = model.predict(roi_reshaped, verbose=0)
    class_id = np.argmax(preds)
    confidence = np.max(preds)
    letter = label_map[class_id]

    # Display ROI and Prediction
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{letter} ({confidence*100:.1f}%)", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("ASL Alphabet Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()