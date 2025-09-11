import cv2
import numpy as np
import tensorflow as tf
import pickle
import os
from collections import deque

# --- Configuration ---
MODEL_PATH = os.path.join('dynamic_model', 'dynamic_sign_model.keras')
LABEL_ENCODER_PATH = os.path.join('processed_sequences', 'label_encoder.pkl')
SEQUENCE_LENGTH = 50
IMG_SIZE = 64
ROI_SIZE = 300

def load_dependencies():
    """Loads the trained model and label encoder."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        return model, label_encoder
    except Exception as e:
        print(f"Error loading dependencies: {e}")
        return None, None

def preprocess_frame(frame):
    """Preprocesses a single frame for the model."""
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    return img_normalized

def run_live_translation():
    """Runs the real-time dynamic sign language translation."""
    model, label_encoder = load_dependencies()
    if model is None:
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # A deque to hold the sequence of frames
    frame_sequence = deque(maxlen=SEQUENCE_LENGTH)
    current_prediction = ""
    confidence = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        x1 = int((width - ROI_SIZE) / 2)
        y1 = int((height - ROI_SIZE) / 2)
        x2 = x1 + ROI_SIZE
        y2 = y1 + ROI_SIZE

        # Draw ROI
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Extract and preprocess ROI
        roi = frame[y1:y2, x1:x2]
        processed = preprocess_frame(roi)
        frame_sequence.append(processed)

        # Predict only when we have a full sequence
        if len(frame_sequence) == SEQUENCE_LENGTH:
            # Reshape the sequence for the model
            sequence_array = np.array(frame_sequence)
            sequence_reshaped = np.expand_dims(sequence_array, axis=0) # Add batch dimension
            sequence_reshaped = np.expand_dims(sequence_reshaped, axis=-1) # Add channel dimension

            # Make prediction
            prediction = model.predict(sequence_reshaped)[0]
            predicted_index = np.argmax(prediction)
            confidence = prediction[predicted_index]
            
            # Get the label
            if confidence > 0.6: # Confidence threshold
                current_prediction = label_encoder.inverse_transform([predicted_index])[0]
            else:
                current_prediction = "..."

        # Display the prediction of the dataset
        text = f"{current_prediction} ({confidence*100:.1f}%)"
        cv2.rectangle(frame, (x1, y1 - 40), (x2, y1), (0, 255, 0), -1)
        cv2.putText(frame, text, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Dynamic Sign Translation', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_translation()

