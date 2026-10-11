# app.py (Member A/Integrator)
import os
import joblib
import cv2
import time
import numpy as np
import tensorflow as tf
import xgboost as xgb
from google import genai
from google.genai import types
from flask import Flask, request, jsonify, render_template

from metrics_extractor import AdvancedMetricsExtractor
from structural_health_analyzer import HealthAnalyzer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

MODEL_PATH = 'models/unet_finetuned.keras'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Model file error ({e}). Using fallback.")
    model = None

metrics_tool = AdvancedMetricsExtractor()

# Load Tabular ML Model globally
risk_model = joblib.load('risk_engine.pkl')

# Your API Key Rotation Pool
API_KEYS = ["YOUR_API_KEY"]

# --- UPDATE 1: New Agentic Recommendations Function ---
def get_agentic_recommendations(fast_image_bytes, width, risk_score):
    # Strict prompt to forbid markdown stars and require a timeline estimate
    prompt = f"""You are a structural engineer assessing a physical surface.
The crack is {width}mm wide with a {risk_score}% structural risk score. 
Analyze the image context and provide the following:

RECOMMENDATIONS:

- Provide exactly 3 concise bullet points of actionable maintenance advice. 
- Structure each bullet point with an UPPERCASE TITLE followed by a colon and a brief explanation. 
- Strict limit: Maximum 2 sentences per bullet point.

SUSTAINABILITY TIMELINE:
- Provide exactly one single, definitive sentence estimating how long the structure can sustain its current condition based on the {risk_score}% risk score. 
- Format strictly as: 'SUSTAINABILITY TIMELINE: [Your 1-sentence estimate here]'. Do not explain or justify the timeline.

CRITICAL FORMATTING RULE: You must use plain text only. Do NOT use markdown. Do NOT use asterisks (*) or bold formatting anywhere in your response."""
    
    for key in API_KEYS:
        if key.startswith('YOUR_'): continue
        try:
            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    prompt,
                    # Hardcoded to jpeg because OpenCV converts everything to a fast jpg buffer
                    types.Part.from_bytes(data=fast_image_bytes, mime_type='image/jpeg')
                ]
            )
            return response.text
        except Exception as e:
            print(f"API Error with key: {e}")
            continue
            
    return "- Monitor crack expansion bi-weekly.\n- Restrict heavy load near the affected surface.\n- Consult an engineer for physical assessment.\n\nSUSTAINABILITY TIMELINE:\n- Unable to calculate timeline. Defaulting to 30-day reassessment window."


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/rules')
def rules():
    return render_template('rules.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename).replace("\\", "/")
        
        # Save file for OpenCV processing
        file.save(filepath)

        raw_img = cv2.imread(filepath)
        if raw_img is None:
            return jsonify({'error': 'Invalid image file'}), 400

        orig_h, orig_w = raw_img.shape[:2]

        # --- UPDATE 2: Instantly compress image for ultra-fast API processing ---
        api_img = cv2.resize(raw_img, (512, 512)) # Shrink to small square
        _, api_buffer = cv2.imencode('.jpg', api_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        fast_image_bytes = api_buffer.tobytes()
        # ------------------------------------------------------------------------

        # 1. Preprocess image to 256x256
        rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(rgb_img, (256, 256))
        normalized_img = resized_img.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(normalized_img, axis=0)

        # 2. Run inference
        if model is not None:
            pred = model.predict(input_tensor, verbose=0)
            pred_2d = np.squeeze(pred)
        else:
            return jsonify({'error': 'Model not available'}), 500

        # 3. Safely retrieve and convert distance from the frontend
        raw_distance = request.form.get('distance_mm')
        try:
            distance_val = float(raw_distance)
        except (TypeError, ValueError):
            print(f"Warning: Invalid distance '{raw_distance}'. Defaulting to 300.0 mm.")
            distance_val = 300.0

        # 4. Extract Metrics using the safe dynamic distance
        processed_results = metrics_tool.compute_all_metrics(
            pred_2d,
            metadata={'distance_mm': distance_val}
        )
        
        width_val = processed_results["max_width_mm"]
        length_val = processed_results["total_length_mm"]
        area_val = processed_results["area_mm2"]

        # 5 & 6. Dynamic Risk Prediction & AI Recommendations
        # 5 & 6. Dynamic Risk Prediction & AI Recommendations
        # 5 & 6. Dynamic Risk Prediction & AI Recommendations
        # 5 & 6. Dynamic Risk Prediction & AI Recommendations
       # 5 & 6. Dynamic Risk Prediction & AI Recommendations
        if processed_results["crack_detected"]:
            # 1. Use ML Model to determine the primary category (0, 1, or 2)
            risk_probabilities = risk_model.predict_proba([[width_val, length_val, area_val]])[0]
            ml_class = np.argmax(risk_probabilities) 
            
            # --- STRICT ENGINEERING GUARDRAIL ---
            # Prevents ML approximation from rounding up borderline cases like 0.59mm
            if width_val < 0.6 and not (width_val >= 0.4 and length_val > 100):
                ml_class = 0  # Force to Hairline
            elif width_val >= 1.8 or (width_val >= 1.5 and length_val > 150):
                ml_class = 2  # Force to Severe
            # ------------------------------------
            
            # 2. Apply a continuous mathematical gradient inside the ML category
            if ml_class == 0:
                # Class 0 (Hairline): Maps widths under 0.6mm smoothly to 5% - 25%
                status = "STABLE (HAIRLINE)"
                badge_color = "green"
                risk_score = int(5 + (width_val / 0.6) * 20)
                
            elif ml_class == 1:
                # Class 1 (Moderate): Maps widths 0.6mm - 1.5mm smoothly to 35% - 66%
                status = "MODERATE FISSURE"
                badge_color = "amber"
                ramp_factor = min(max(width_val - 0.6, 0), 0.9) / 0.9
                risk_score = int(35 + (ramp_factor * 31))
                
            else:
                # Class 2 (Severe): Maps widths >1.5mm smoothly to 75% - 100%
                status = "SEVERE STRUCTURAL RISK"
                badge_color = "red"
                ramp_factor = min(max(width_val - 1.5, 0), 1.0) / 1.0
                risk_score = int(75 + (ramp_factor * 25))
                
            # Ensure score doesn't exceed 100 in extreme edge cases
            risk_score = min(risk_score, 100)
            
            # Ask Gemini for advice using the compressed, high-speed image
            recommendations = get_agentic_recommendations(fast_image_bytes, width_val, risk_score)
        # 7. Upscale binary mask to full image for visualization
        binary_mask_256 = processed_results["analysis_mask"]
        upscaled_mask = cv2.resize(binary_mask_256, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        blended_visual = raw_img.copy()
        crack_pixels = upscaled_mask > 0

        if np.any(crack_pixels):
            overlay_color = np.array([0, 0, 255], dtype=np.uint8) # Red (BGR)
            blended_visual[crack_pixels] = cv2.addWeighted(
                raw_img[crack_pixels], 0.25,
                np.full_like(raw_img[crack_pixels], overlay_color), 0.75, 0
            )

        processed_filename = f"mask_{filename}"
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename).replace("\\", "/")
        cv2.imwrite(processed_path, blended_visual)

        timestamp = int(time.time())

        return jsonify({
            "detected": processed_results["crack_detected"],
            "pixel_span": width_val,
            "length_mm": length_val,
            "depth_mm": processed_results.get("estimated_depth_mm", 0.0),
            "area_mm2": area_val,
            "risk_score": risk_score, 
            "status": status,
            "badge_color": badge_color,
            "recommendations": recommendations,
            "original_image": f"/{filepath}?t={timestamp}",
            "processed_image": f"/{processed_path}?t={timestamp}"
        })

    except Exception as err:
        return jsonify({"error": f"Internal processing error: {str(err)}"}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('models', exist_ok=True)
    app.run(debug=True)