import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

# --- Configuration ---
VIDEOS_DIR = 'videos'
PROCESSED_DIR = 'processed_sequences'
IMG_SIZE = 64
SEQUENCE_LENGTH = 50 # Number of frames per sequence (must match VIDEO_LENGTH_FRAMES in collection)

def create_directory(path):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def preprocess_videos():
    """
    Reads video files, extracts frames, processes them, and saves them as numpy arrays.
    """
    print("Starting video preprocessing...")
    sequences = []
    labels = []

    sign_folders = [f for f in os.listdir(VIDEOS_DIR) if os.path.isdir(os.path.join(VIDEOS_DIR, f))]
    if not sign_folders:
        print(f"Error: No data found in '{VIDEOS_DIR}'. Please run collect_videos.py first.")
        return

    for sign in sign_folders:
        sign_path = os.path.join(VIDEOS_DIR, sign)
        print(f"Processing videos for sign: {sign}")

        for video_file in os.listdir(sign_path):
            video_path = os.path.join(sign_path, video_file)
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
                img_normalized = img_resized / 255.0
                frames.append(img_normalized)
            
            cap.release()

            # Ensure we have the correct number of frames
            if len(frames) == SEQUENCE_LENGTH:
                sequences.append(frames)
                labels.append(sign)
            else:
                print(f"Warning: Video {video_file} has {len(frames)} frames, expected {SEQUENCE_LENGTH}. Skipping.")

    if not sequences:
        print("Error: No sequences were successfully created.")
        return

    X = np.array(sequences)
    y_array = np.array(labels)

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_array)

    # Save the processed data and label encoder
    create_directory(PROCESSED_DIR)
    np.save(os.path.join(PROCESSED_DIR, 'sequences.npy'), X)
    np.save(os.path.join(PROCESSED_DIR, 'labels.npy'), y)
    with open(os.path.join(PROCESSED_DIR, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)

    print("\nVideo preprocessing complete!")
    print(f"Total sequences created: {len(sequences)}")
    print(f"Shape of sequences array (X): {X.shape}")
    print(f"Shape of labels array (y): {y.shape}")
    print(f"Classes: {label_encoder.classes_}")
    print(f"Processed data saved to '{PROCESSED_DIR}'.")

if __name__ == "__main__":
    preprocess_videos()
