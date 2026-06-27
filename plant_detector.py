import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# ----------------------------
# Step 1: Load your trained model
# ----------------------------
try:
    model_tl = load_model("plant_disease_model.h5")
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
    exit()

# ----------------------------
# Step 2: Load classes from classes.npy
# ----------------------------
classes_file = "classes.npy"  # should contain your classes in order

if os.path.exists(classes_file):
    classes = np.load(classes_file, allow_pickle=True).tolist()
    print(f"✅ Loaded {len(classes)} classes from {classes_file}")
else:
    # Fallback if no classes.npy exists
    num_classes = model_tl.output_shape[-1]
    classes = [f"Class_{i}" for i in range(num_classes)]
    print(f"⚠️ No classes file found. Using {num_classes} placeholder class names.")

# ----------------------------
# Step 3: Prediction function
# ----------------------------
def predict_frame(frame):
    img_resized = cv2.resize(frame, (128, 128)) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    pred = model_tl.predict(img_resized)
    
    class_idx = np.argmax(pred)
    confidence = np.max(pred) * 100
    label = classes[class_idx]  # safe since classes are loaded
    return label, confidence

# ----------------------------
# Step 4: Open webcam
# ----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("✅ Webcam started. Press 'c' to capture image and predict disease.")

# Create folder to save captured images
save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)

img_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)  # mirror view

    # Show live camera
    cv2.imshow("Plant Disease Detector", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):  # capture and predict
        img_count += 1
        label, confidence = predict_frame(frame)
        
        # Save the captured image
        filename = os.path.join(save_dir, f"{label}_{img_count}.png")
        cv2.imwrite(filename, frame)
        
        print(f"\n📸 Image captured and saved: {filename}")
        print(f"🌱 Predicted Disease: {label}")
        print(f"✅ Confidence: {confidence:.2f}%")
        break  # exit after capturing

# ----------------------------
# Step 5: Release resources
# ----------------------------
cap.release()
cv2.destroyAllWindows()
print("✅ Webcam closed. Program exited.")
