# metrics_extractor.py (Importable Version)
import numpy as np
import cv2

class AdvancedMetricsExtractor:
    def __init__(self):
        # 1. Focal Length (Calibrated for dynamic pinhole scaling)
        self.focal_length_px = 500.0  
        
        # 2. Aggressively reduce width bloat caused by U-Net's thick segmentation masks
        self.width_adjustment = 0.15  

    def compute_all_metrics(self, prediction_map, metadata=None):
        if metadata is None:
            metadata = {}
        
        # 1. Enforce strict 256x256 resolution
        if prediction_map.shape[:2] != (256, 256):
            prediction_map = cv2.resize(prediction_map, (256, 256), interpolation=cv2.INTER_NEAREST)

        # 2. Retrieve dynamic standoff distance (default 300 mm)
        distance_mm = float(metadata.get('distance_mm', 300.0))

        # 3. Dynamic Optical Scale: Pinhole Model
        mm_per_pixel = distance_mm / self.focal_length_px

        if prediction_map.dtype != np.uint8:
            pred_norm = (prediction_map * 255).astype(np.uint8)
        else:
            pred_norm = prediction_map

        _, mask = cv2.threshold(pred_norm, 127, 255, cv2.THRESH_BINARY)

        # Remove isolated background noise component pixels (less than 20 px)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        clean_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 20:
                clean_mask[labels == i] = 255

        if np.sum(clean_mask) < 50:
            return {
                "crack_detected": False,
                "max_width_mm": 0.0,
                "total_length_mm": 0.0,
                "estimated_depth_mm": 0.0,
                "area_mm2": 0.0,
                "analysis_mask": clean_mask
            }

        # 4. Extract Contours (FIXED LOOPHOLE)
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        total_area_px = 0.0
        all_contour_points = []
        
        for cnt in contours:
            total_area_px += cv2.contourArea(cnt)
            # Pool all points together instead of measuring fragments individually
            all_contour_points.extend(cnt)

        # Draw ONE global bounding rectangle around the entire crack system
        total_length_px = 0.0
        if len(all_contour_points) > 0:
            all_contour_points = np.array(all_contour_points)
            rect = cv2.minAreaRect(all_contour_points)
            (_, _), (w, h), _ = rect
            total_length_px = max(w, h)

        # 5. Apply Dynamic Scale to Length
        length_mm = round(total_length_px * mm_per_pixel, 2)

        # 6. Apply Dynamic Scale to Width (FIXED LOOPHOLE)
        if total_length_px > 0:
            raw_width_px = total_area_px / total_length_px
        else:
            raw_width_px = 0.0

        # Now correctly using the class variable instead of a hardcoded 0.45
        width_mm = round((raw_width_px * mm_per_pixel) * self.width_adjustment, 2)
        
        # Safeguard to prevent 0.0 mm width on detected cracks
        if width_mm < 0.1 and length_mm > 0:
            width_mm = 0.1

        # 7. Empirical Penetration Depth
        estimated_depth_mm = self._predict_depth(width_mm, length_mm)
        area_mm2 = round(total_area_px * (mm_per_pixel ** 2), 2)
        return {
            "crack_detected": True,
            "max_width_mm": width_mm,
            "total_length_mm": length_mm,
            "estimated_depth_mm": estimated_depth_mm,
            "area_mm2": area_mm2,
            "analysis_mask": clean_mask
        }

    def _predict_depth(self, width_mm, length_mm):
        if width_mm == 0 or length_mm == 0:
            return 0.0
        depth = (3.2 * width_mm) + (0.04 * np.sqrt(length_mm))
        return round(float(depth), 2)