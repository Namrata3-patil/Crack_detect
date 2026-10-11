# run_automated_tests.py
import os
import cv2
import csv
import numpy as np
import tensorflow as tf
from metrics_extractor import AdvancedMetricsExtractor # Ensure this is your latest, dynamically calibrated class

# --- CONFIGURATION ---
MODEL_PATH = 'models/unet_finetuned.keras'
TEST_IMAGES_DIR = 'test_images' # Folder containing your test images
METADATA_CSV = 'test_dataset_metadata.csv' # The file defined in Step 1
OUTPUT_REPORT_CSV = 'automated_test_report.csv'

# Load AI model and metrics tool
extractor = AdvancedMetricsExtractor()
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("AI Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Main Test Loop
test_report = []
print(f"Starting automated batch testing using metadata from '{METADATA_CSV}'...")

# 1. Read the metadata file
with open(METADATA_CSV, mode='r') as file:
    reader = csv.DictReader(file)
    test_cases = list(reader)

for idx, case in enumerate(test_cases):
    filename = case['filename']
    true_distance = float(case['true_distance_mm'])
    
    try:
        # 2. Load the corresponding image
        img_path = os.path.join(TEST_IMAGES_DIR, filename)
        if not os.path.exists(img_path):
            print(f"Skipping {filename}: File not found.")
            continue

        raw_img = cv2.imread(img_path)
        if raw_img is None: continue

        # AI Model preprocessing (mirroring the backend logic)
        rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(rgb_img, (256, 256)) # U-Net input size
        normalized_img = resized_img.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(normalized_img, axis=0)

        # 3. Run AI Inference (Phase 3 Segmentation)
        pred = model.predict(input_tensor, verbose=0)
        pred_2d = np.squeeze(pred)

        # 4. Call the metrics pipeline with the specific TRUE distance (Phase 4 Metrology)
        # We pass the unique distance for THIS image into the metadata dictionary.
        processed_results = extractor.compute_all_metrics(
            pred_2d, 
            metadata={'distance_mm': true_distance}
        )

        measured_length = processed_results["total_length_mm"]
        measured_width = processed_results["max_width_mm"]

        print(f"[{idx+1}/{len(test_cases)}] Tested {filename} at {true_distance}mm: {measured_length:.2f}mm Len, {measured_width:.2f}mm Wid")

        # 5. Store the results for the final report
        # If expected ground truth was in the CSV, you can add pass/fail logic here too.
        test_report.append([filename, true_distance, measured_length, measured_width])

    except Exception as err:
        print(f"Error processing {filename}: {err}")
        test_report.append([filename, true_distance, "ERROR", str(err)])

# Save results to CSV report
with open(OUTPUT_REPORT_CSV, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Filename', 'True_Distance_mm', 'Measured_Length_mm', 'Measured_Width_mm'])
    writer.writerows(test_report)

print(f"\nAutomated Batch Testing Complete. Detailed report saved to '{OUTPUT_REPORT_CSV}'.")