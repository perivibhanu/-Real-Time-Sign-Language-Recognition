import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# --- Configuration ---
# Set the path to the directory where your X.npy, y.npy, and label_map.txt are saved.
# IMPORTANT: Ensure this path is correct for your system!
OUTPUT_DIR = 'hand_landmarks_dataset/' 

# --- Load Data ---
print(f"Loading data from: {os.path.abspath(OUTPUT_DIR)}")
try:
    X = np.load(os.path.join(OUTPUT_DIR, 'X.npy'))
    y = np.load(os.path.join(OUTPUT_DIR, 'y.npy'))
except FileNotFoundError as e:
    print(f"Error: Required data file not found. Make sure '{OUTPUT_DIR}' contains X.npy and y.npy.")
    print("Did you run the 'Data Preparation and Preprocessing' script successfully?")
    raise e # Re-raise the error to stop execution

# Load the label map (for mapping back to names, not directly for num_classes)
label_map = {}
try:
    with open(os.path.join(OUTPUT_DIR, 'label_map.txt'), 'r') as f:
        for line in f:
            idx, label_name = line.strip().split(':')
            label_map[int(idx)] = label_name
except FileNotFoundError as e:
    print(f"Error: label_map.txt not found in '{OUTPUT_DIR}'.")
    print("Did you run the 'Data Preparation and Preprocessing' script successfully?")
    raise e # Re-raise the error to stop execution


# --- Data Preprocessing for Training ---
# Reshape X for Conv1D: (samples, timesteps, features)
# MediaPipe outputs 21 landmarks, each with (x, y, z) coordinates = 63 features.
# Reshape to (num_samples, 21, 3)
if X.shape[1] != 63:
    print(f"Warning: Expected X.shape[1] to be 63 (21 landmarks * 3 coords), but got {X.shape[1]}.")
    print("This indicates a potential issue in your data preprocessing. Attempting to reshape anyway.")
    X = X.reshape(X.shape[0], 21, 3) 
else:
    X = X.reshape(X.shape[0], 21, 3) 


print(f"Original data shape (X): {X.shape}, Labels shape (y): {y.shape}")
print(f"Unique labels found in y.npy: {np.unique(y)}")


# Split data into training and testing sets
# stratify=y ensures that the proportion of classes in the training and testing sets is the same as in the full dataset.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert labels to one-hot encoding
lb = LabelBinarizer()
y_train_one_hot = lb.fit_transform(y_train)
y_test_one_hot = lb.transform(y_test)

# --- THE CRITICAL FIX: DETERMINE NUMBER OF CLASSES DIRECTLY FROM ONE-HOT ENCODED LABELS ---
# This ensures that 'num_classes' precisely matches the actual number of classes in your training data.
num_classes = y_train_one_hot.shape[1] 

print(f"\n--- Model Configuration Details ---")
print(f"Number of classes derived from one-hot encoded labels (num_classes): {num_classes}")
print(f"Shape of X_train: {X_train.shape}, y_train_one_hot: {y_train_one_hot.shape}")
print(f"Shape of X_test: {X_test.shape}, y_test_one_hot: {y_test_one_hot.shape}")
print(f"Number of classes from original label_map: {len(label_map)}")
print("-------------------------------------")


# --- Define the CNN Model ---
print("Building CNN model...")
model = Sequential([
    # Conv1D is suitable for sequence data like landmarks (21 steps, 3 features per step)
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(21, 3)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(), # Flatten the output of convolutional layers for dense layers
    Dense(128, activation='relu'),
    Dropout(0.5), # Helps prevent overfitting
    Dense(num_classes, activation='softmax') # Output layer: number of neurons = number of classes
])

# Compile the model
# Using Adam optimizer and categorical_crossentropy for multi-class classification
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary() # Print a summary of the model architecture

# --- Callbacks for Better Training ---
# Early Stopping: Stop training if validation loss doesn't improve for 'patience' epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Reduce Learning Rate on Plateau: Reduce learning rate if validation loss plateaus
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# --- Train the Model ---
print("\nStarting model training...")
history = model.fit(
    X_train, y_train_one_hot,
    epochs=50, # Maximum number of epochs
    batch_size=32,
    validation_data=(X_test, y_test_one_hot),
    callbacks=[early_stopping, reduce_lr]
)

# --- Evaluate the Model ---
print("\nEvaluating model on test data...")
loss, accuracy = model.evaluate(X_test, y_test_one_hot, verbose=0)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# --- Save the Trained Model ---
model_save_path = 'sign_language_model.h5'
model.save(model_save_path)
print(f"Model saved as {model_save_path}")

print("\nTraining process completed.")