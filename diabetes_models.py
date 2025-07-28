import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib
from sklearn.preprocessing import LabelEncoder
import pickle
# ==============================
# TRAINING: DIABETES MODEL
# ==============================

print("🔵 Training Diabetes Model with XGBoost...")

# Load dataset
diabetes_df = pd.read_csv("datasets/diabetes.csv")
print(diabetes_df.diabetes.value_counts())

# Initialize encoders dictionary
encoders = {}
categorical_columns = ['gender', 'smoking_history']  # adjust based on your categorical columns

# Create copy of dataframe for encoding
diabetes_df_encoded = diabetes_df.copy()

# Apply LabelEncoder to categorical columns
for col in categorical_columns:
    if col in diabetes_df_encoded.columns:
        le = LabelEncoder()
        diabetes_df_encoded[col] = le.fit_transform(diabetes_df_encoded[col])
        encoders[col] = le

# Save encoders for later use in prediction
with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

# Split features and labels
X_d = diabetes_df_encoded.drop('diabetes', axis=1)
y_d = diabetes_df_encoded['diabetes']

# Split into train/test sets
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.25, random_state=42)

# Train model using XGBoost
diabetes_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
diabetes_model.fit(X_train_d, y_train_d)

print("Encoders saved successfully!")
print("Model trained successfully!")
print("Encoded columns:", X_d.columns.tolist())
print (X_train_d.shape)

# Evaluate on test set
y_pred_d = diabetes_model.predict(X_test_d)
accuracy_d = accuracy_score(y_test_d, y_pred_d)
print(f"📊 Diabetes Model Accuracy: {accuracy_d:.2f}")
print("🧾 Classification Report (Diabetes):\n", classification_report(y_test_d, y_pred_d))

# Save model
joblib.dump(diabetes_model, 'saved_models/diabetes_model.pkl')
print("✅ Saved: diabetes_model.pkl")

