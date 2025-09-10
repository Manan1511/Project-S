import cv2
import os
import numpy as np
import time

# --- Configuration ---
VIDEOS_DIR = 'videos'
ROI_SIZE = 300      # Size of the square region of interest
NUM_VIDEOS_PER_SIGN = 50 # Number of videos to capture per sign
VIDEO_LENGTH_FRAMES = 30 # Number of frames to record per video

def create_directory(path):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def collect_videos():
    """
    Uses the webcam to capture and save short video clips for sign language actions.
    """
    sign_name = input("Enter the name of the sign/action you are collecting videos for: ")
    sign_path = os.path.join(VIDEOS_DIR, sign_name)
    create_directory(sign_path)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    video_count = 0
    while video_count < NUM_VIDEOS_PER_SIGN:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        x1 = int((width - ROI_SIZE) / 2)
        y1 = int((height - ROI_SIZE) / 2)
        x2 = x1 + ROI_SIZE
        y2 = y1 + ROI_SIZE

        # Display instructions
        cv2.putText(frame, f"Collecting videos for: {sign_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Video Count: {video_count}/{NUM_VIDEOS_PER_SIGN}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Press 's' to start recording.", (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit.", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('Video Collection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print(f"Starting recording for video {video_count}...")
            frames = []
            
            # Record for a fixed number of frames
            for i in range(VIDEO_LENGTH_FRAMES):
                ret, rec_frame = cap.read()
                if not ret:
                    break
                
                rec_frame = cv2.flip(rec_frame, 1)
                
                # Display "RECORDING" text
                cv2.putText(rec_frame, "RECORDING", (width // 2 - 60, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.rectangle(rec_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.imshow('Video Collection', rec_frame)
                cv2.waitKey(1) # Necessary to update the window
                
                # Extract and store the ROI
                roi = rec_frame[y1:y2, x1:x2]
                frames.append(roi)
            
            if len(frames) == VIDEO_LENGTH_FRAMES:
                # Save the collected frames as a video file
                video_path = os.path.join(sign_path, f'{sign_name}_{video_count}.avi')
                
                # Get the frame size from the first frame
                frame_height, frame_width, _ = frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(video_path, fourcc, 15.0, (frame_width, frame_height))

                for f in frames:
                    out.write(f)

                out.release()
                print(f"Saved {video_path}")
                video_count += 1
            else:
                print("Recording failed. Not enough frames captured.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    create_directory(VIDEOS_DIR)
    collect_videos()
