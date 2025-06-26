import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

# ==============================
# TRAINING: DIABETES MODEL
# ==============================

print("🔵 Training Diabetes Model with XGBoost...")

# Load dataset
diabetes_df = pd.read_csv("datasets/diabetes.csv")

# Convert target column ('diabetes') from 0/1 if needed (already 0/1 in your dataset)
# If it were 'Yes'/'No', you'd map it like this:
# diabetes_df['diabetes'] = diabetes_df['diabetes'].map({'Yes': 1, 'No': 0})

# One-hot encode categorical features (e.g., 'gender', 'smoking_history')
diabetes_df_encoded = pd.get_dummies(diabetes_df, drop_first=True)

# Split features and labels
X_d = diabetes_df_encoded.drop('diabetes', axis=1)
y_d = diabetes_df_encoded['diabetes']

# Split into train/test sets
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.25, random_state=42)

# Train model using XGBoost
diabetes_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
diabetes_model.fit(X_train_d, y_train_d)

# Evaluate on test set
y_pred_d = diabetes_model.predict(X_test_d)
accuracy_d = accuracy_score(y_test_d, y_pred_d)
print(f"📊 Diabetes Model Accuracy: {accuracy_d:.2f}")
print("🧾 Classification Report (Diabetes):\n", classification_report(y_test_d, y_pred_d))

# Save model
joblib.dump(diabetes_model, 'saved_models/diabetes_model.pkl')
print("✅ Saved: diabetes_model.pkl")

# ==============================
# TRAINING: HEART DISEASE MODEL
# ==============================
print("\n🔴 Training Heart Disease Model...")

# Load dataset
heart_df = pd.read_csv("datasets/heart.csv")

# Map 'HeartDisease' column Yes/No to 1/0
heart_df['HeartDisease'] = heart_df['HeartDisease'].map({'Yes': 1, 'No': 0})

# One-hot encode categorical columns except the label
heart_df_encoded = pd.get_dummies(heart_df.drop(columns=['HeartDisease']), drop_first=True)

# Features and target
X_h = heart_df_encoded
y_h = heart_df['HeartDisease']

# Split into train/test sets
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_h, y_h, test_size=0.2, random_state=42)

# Train model
heart_model = RandomForestClassifier()
heart_model.fit(X_train_h, y_train_h)

# Evaluate on test set
y_pred_h = heart_model.predict(X_test_h)
accuracy_h = accuracy_score(y_test_h, y_pred_h)
print(f"📊 Heart Disease Model Accuracy: {accuracy_h:.2f}")
print("🧾 Classification Report (Heart Disease):\n", classification_report(y_test_h, y_pred_h))

# Save model
joblib.dump(heart_model, 'saved_models/heart_model.pkl')
print("✅ Saved: heart_model.pkl")