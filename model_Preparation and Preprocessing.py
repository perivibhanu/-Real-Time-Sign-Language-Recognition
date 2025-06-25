import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

DATA_DIR = "F:\\project\\RealTimeSignLanguageRecognition\\asl_alphabet_train" # Replace with the actual path to your dataset
OUTPUT_DIR = 'hand_landmarks_dataset/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

labels = sorted(os.listdir(DATA_DIR))
data = []
processed_labels = []

for label_idx, label in enumerate(labels):
    print(f"Processing label: {label}")
    label_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_path):
        continue

    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        
        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue
        
        # Convert the image to RGB (MediaPipe requires RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image with MediaPipe
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmarks (x, y, z coordinates for each of 21 keypoints)
                # Normalize coordinates to be relative to the image size
                landmarks_flat = []
                for lm in hand_landmarks.landmark:
                    landmarks_flat.extend([lm.x, lm.y, lm.z])
                
                # You might want to normalize these further, e.g., relative to the wrist
                # to make the model invariant to hand position/size.
                # Example: Subtract wrist (landmark 0) coordinates from all other landmarks.
                wrist_x, wrist_y, wrist_z = landmarks_flat[0:3]
                normalized_landmarks = []
                for i in range(0, len(landmarks_flat), 3):
                    normalized_landmarks.extend([
                        landmarks_flat[i] - wrist_x,
                        landmarks_flat[i+1] - wrist_y,
                        landmarks_flat[i+2] - wrist_z
                    ])
                
                data.append(normalized_landmarks)
                processed_labels.append(label_idx) # Store numerical label

# Convert to NumPy arrays
X = np.array(data)
y = np.array(processed_labels)

# Save the processed data
np.save(os.path.join(OUTPUT_DIR, 'X.npy'), X)
np.save(os.path.join(OUTPUT_DIR, 'y.npy'), y)

# It's good practice to also save the mapping from numerical labels back to text
label_map = {idx: label for idx, label in enumerate(labels)}
with open(os.path.join(OUTPUT_DIR, 'label_map.txt'), 'w') as f:
    for idx, label_name in label_map.items():
        f.write(f"{idx}:{label_name}\n")

print(f"Data extraction complete. X shape: {X.shape}, y shape: {y.shape}")
print(f"Labels processed: {labels}")