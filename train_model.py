import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load Data
df = pd.DataFrame(pd.read_csv('structural_risk_data.csv'))
X = df[['width_mm', 'length_mm', 'area_mm2']]
y = df['risk_label']

# 2. Split into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train XGBoost Model
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)
model.fit(X_train, y_train)

# 4. Verify Accuracy
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(f"XGBoost Model Trained. Accuracy: {acc * 100:.2f}%")

# 5. Export Model for Flask
joblib.dump(model, 'risk_engine.pkl')
print("Model saved to risk_engine.pkl")