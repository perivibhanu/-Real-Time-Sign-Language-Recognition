import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import time
import os
from collections import Counter
import pyttsx3

# --- Configuration ---
OUTPUT_DIR = 'hand_landmarks_dataset/'

# --- Initialize Text-to-Speech Engine ---
engine = pyttsx3.init()
# You can adjust properties like speed and volume if needed
# engine.setProperty('rate', 200) # Increased speed for faster output (e.g., 200 WPM)
# engine.setProperty('volume', 0.9) # Volume (0.0 to 1.0)

# Function to speak the text
def speak_text(text):
    if text: # Only speak if there's text
        engine.say(text)
        engine.runAndWait()

# Load the trained model
model = load_model('sign_language_model.h5')

# Load the label map
label_map = {}
with open(os.path.join(OUTPUT_DIR, 'label_map.txt'), 'r') as f:
    for line in f:
        idx, label_name = line.strip().split(':')
        label_map[int(idx)] = label_name

# Initialize MediaPipe Hands
# Adjust confidence if needed, but be mindful of accuracy
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

predicted_text = "" # This will display the current sign being recognized
detection_threshold = 0.5 # Confidence threshold for displaying prediction
prediction_history = [] # To store recent predictions for smoothing

# --- ADJUSTED FOR FASTER OUTPUT ---
history_length = 5 # Reduced from 10: Fewer frames to consider for smoothing, faster response
stable_prediction_duration = 0.2 # Reduced from 0.4: Sign needs to be stable for less time
word_break_duration = 0.2 # Reduced from 0.3: Shorter pause to trigger word completion

# Variables for word building
current_word = ""
last_predicted_char = ""
last_prediction_time = time.time()


print("Starting real-time sign language recognition...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Initialize predicted_text here so it's always set
    predicted_text = "No Hand Detected"

    # --- MAIN LOGIC BLOCK: IF HANDS ARE DETECTED ---
    if results.multi_hand_landmarks:
        # Only process the first detected hand for simplicity
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmarks_flat = []
        for lm in hand_landmarks.landmark:
            landmarks_flat.extend([lm.x, lm.y, lm.z])
        wrist_x, wrist_y, wrist_z = landmarks_flat[0:3]
        normalized_landmarks = []
        for i in range(0, len(landmarks_flat), 3):
            normalized_landmarks.extend([
                landmarks_flat[i] - wrist_x,
                landmarks_flat[i+1] - wrist_y,
                landmarks_flat[i+2] - wrist_z
            ])

        input_data = np.array(normalized_landmarks).reshape(1, 21, 3)

        # Removed verbose=0 as it doesn't affect speed
        prediction = model.predict(input_data)
        predicted_class_idx = np.argmax(prediction)
        confidence = prediction[0][predicted_class_idx]

        # --- SUB-LOGIC BLOCK: IF CONFIDENCE IS HIGH ---
        if confidence > detection_threshold:
            current_sign_prediction = label_map[predicted_class_idx]
            prediction_history.append(current_sign_prediction)

            if len(prediction_history) > history_length:
                prediction_history.pop(0)

            most_common = Counter(prediction_history).most_common(1)
            smoothed_prediction = most_common[0][0]

            predicted_text = smoothed_prediction # Update predicted_text for display

            # Word building logic
            if smoothed_prediction != last_predicted_char and \
               (time.time() - last_prediction_time) > stable_prediction_duration:

                if smoothed_prediction == "space": # Define a "space" sign in your dataset
                    current_word += " "
                elif smoothed_prediction == "backspace": # Define a "backspace" sign
                    current_word = current_word[:-1]
                else:
                    current_word += smoothed_prediction

                last_predicted_char = smoothed_prediction
                last_prediction_time = time.time()

        # --- ELSE (CONFIDENCE IS LOW, BUT HAND IS STILL DETECTED) ---
        else: # confidence <= detection_threshold
            predicted_text = "Detecting..." # Sign detected but too low confidence
            prediction_history = [] # Clear history if confidence drops
            last_predicted_char = "" # Reset char to allow new sign
            last_prediction_time = time.time() # Reset timer for stability

    # --- ELSE (NO HAND DETECTED) ---
    else: # No hand detected at all
        predicted_text = "No Hand Detected"
        prediction_history = [] # Clear history if no hand

        # Word break logic if no hand is detected for a prolonged period
        if (time.time() - last_prediction_time) > word_break_duration and current_word.strip():
            print(f"Word completed: {current_word.strip()}")
            speak_text(current_word.strip())
            current_word = "" # Reset for next word

        last_predicted_char = "" # Reset for next gesture
        last_prediction_time = time.time() # Reset timer for stability


    # Display the current word being formed
    cv2.putText(frame, f"Word: {current_word}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    # Display the instantaneous predicted sign
    cv2.putText(frame, f"Sign: {predicted_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Sign Language Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()