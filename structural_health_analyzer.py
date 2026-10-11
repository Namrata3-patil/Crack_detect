# structural_health_analyzer.py

class HealthAnalyzer:
    @staticmethod
    def calculate_condition(width_mm):
        if width_mm == 0:
            return {
                "status": "Optimal",
                "risk_score": 0,
                "color": "green",
                "action": "No action required."
            }
        elif width_mm < 1.0:
            return {
                "status": "Hairline Fissure",
                "risk_score": 25,
                "color": "yellow",
                "action": "Monitor for changes."
            }
        elif width_mm < 3.0:
            return {
                "status": "Moderate Damage",
                "risk_score": 60,
                "color": "orange",
                "action": "Surface repair recommended."
            }
        else:
            return {
                "status": "Structural Fissure",
                "risk_score": 90,
                "color": "red",
                "action": "High Risk - Immediate engineering inspection required."
            }