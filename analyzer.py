import cv2
import numpy as np
import os

def process_crack_image(image_path, output_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert to Grayscale & Blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive Thresholding to isolate crack fissures
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Find contours (fissure structures)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    crack_width_pixels = 0
    annotated_img = img.copy()
    
    if contours:
        # Filter out minor noise contours
        valid_contours = [c for c in contours if cv2.contourArea(c) > 30]
        if valid_contours:
            # Highlight all detected crack contours in Red (BGR: 0, 0, 255)
            cv2.drawContours(annotated_img, valid_contours, -1, (0, 0, 255), 2)
            
            # Find largest crack contour for measurement
            largest = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            crack_width_pixels = max(w, h)
            
            # Draw bounding box around main crack
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Save processed image showing highlighted crack
    cv2.imwrite(output_path, annotated_img)
    
    # Calculate Risk Score (0 - 100%)
    img_dimension = max(img.shape[0], img.shape[1])
    risk_score = min(100, int((crack_width_pixels / (img_dimension * 0.3)) * 100))
    if risk_score < 5 and len(contours) > 0:
        risk_score = 15  # Baseline score for minor cracks
        
    return {
        "crack_detected": len(contours) > 0,
        "crack_pixel_span": int(crack_width_pixels),
        "risk_score": risk_score
    }