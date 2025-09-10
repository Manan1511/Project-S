import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# --- Configuration ---
PROCESSED_DIR = 'processed_sequences'
MODEL_DIR = 'dynamic_model'
SEQUENCE_LENGTH = 50
IMG_SIZE = 64
EPOCHS = 50
BATCH_SIZE = 16

def create_directory(path):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def load_data():
    """Loads the preprocessed sequence data."""
    try:
        X = np.load(os.path.join(PROCESSED_DIR, 'sequences.npy'))
        y = np.load(os.path.join(PROCESSED_DIR, 'labels.npy'))
        return X, y
    except FileNotFoundError:
        print(f"Error: Processed data not found in '{PROCESSED_DIR}'. Please run preprocess_videos.py first.")
        return None, None

def build_cnn_lstm_model(num_classes):
    """Builds the CNN-LSTM model."""
    model = Sequential([
        # Input shape: (SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 1)
        Input(shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 1)),

        # CNN part to extract features from each frame
        TimeDistributed(Conv2D(32, (3, 3), activation='relu')),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Flatten()),

        # LSTM part to understand the sequence of features
        LSTM(64, return_sequences=False), # Only return the final output
        Dropout(0.5),

        # Classifier part
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_dynamic_model():
    """Loads data, builds, trains, and saves the CNN-LSTM model."""
    print("Loading data...")
    X, y = load_data()
    if X is None:
        return

    # Reshape X to include the channel dimension
    X = np.expand_dims(X, axis=-1)
    
    num_classes = len(np.unique(y))
    print(f"Found {num_classes} classes.")
    
    # One-hot encode labels
    y_categorical = to_categorical(y, num_classes=num_classes)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical)

    print("\nData shapes:")
    print(f"Training sequences:   {X_train.shape}")
    print(f"Validation sequences: {X_val.shape}")
    
    print("\nBuilding model...")
    model = build_cnn_lstm_model(num_classes)
    model.summary()
    
    print(f"\nStarting training for {EPOCHS} epochs...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val)
    )
    
    print("\nTraining complete.")
    
    # Save the model
    create_directory(MODEL_DIR)
    model_path = os.path.join(MODEL_DIR, 'dynamic_sign_model.keras')
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_dynamic_model()
