import pandas as pd
import numpy as np

# Generate 1000 synthetic crack records
np.random.seed(42)
num_samples = 1000

# Feature generation
widths = np.random.uniform(0.1, 8.0, num_samples) # Width in mm
lengths = np.random.uniform(10.0, 500.0, num_samples) # Length in mm
areas = widths * lengths * np.random.uniform(0.7, 1.0, num_samples) # Area in mm2

# Updated Stricter Risk Logic
# Updated Graduated Risk Logic for smoother, realistic scoring
# Final Calibrated Risk Logic
labels = []
for w, l in zip(widths, lengths):
    # Severe: Wider than 1.8mm OR (Wider than 1.5mm and long)
    if w >= 1.8 or (w >= 1.5 and l > 150):
        labels.append(2) 
        
    # Moderate: Wider than 0.6mm OR (Wider than 0.4mm and long)
    elif w >= 0.6 or (w >= 0.4 and l > 100):
        labels.append(1) 
        
    # Low (Hairline): Everything else
    else:
        labels.append(0)

# Save to CSV
df = pd.DataFrame({'width_mm': widths, 'length_mm': lengths, 'area_mm2': areas, 'risk_label': labels})
df.to_csv('structural_risk_data.csv', index=False)
print("Dataset generated successfully: structural_risk_data.csv")