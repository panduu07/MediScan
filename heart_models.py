import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.preprocessing import LabelEncoder
import pickle

# ==============================
# TRAINING: HEART DISEASE MODEL
# ==============================

print("\n🔴 Training Heart Disease Model...")

heart_df = pd.read_csv("datasets/heart.csv")

print("Unique values in 'HeartDisease' column:", heart_df['HeartDisease'].unique())

print(f"📄 Original heart data rows: {len(heart_df)}")

# Check type of HeartDisease column
print("HeartDisease column dtype:", heart_df['HeartDisease'].dtype)

# Since values are already 0 and 1 (int), no need to map or filter strings
# Just drop rows with NaN in HeartDisease or any feature columns
heart_df = heart_df.dropna(subset=['HeartDisease'])

print(f"✅ Heart data rows after dropna on 'HeartDisease': {heart_df.shape[0]}")

# If any NaNs in features, drop those rows too
heart_df = heart_df.dropna()

print(f"✅ Heart data rows after dropna on all columns: {heart_df.shape[0]}")

# Split features and target
label_encoders = {}

# Assuming heart_df is your DataFrame
# Create a copy to avoid modifying original data
heart_df_encoded = heart_df.copy()

# List of categorical columns that need encoding
categorical_columns = [
    'smoking', 'alcohol_drinking', 'stroke', 'diff_walking', 
    'sex', 'age_category', 'race', 'diabetic', 'physical_activity',
    'gen_health', 'asthma', 'kidney_disease', 'skin_cancer'
]
encoders ={}
# Apply label encoding to categorical columns
for col in categorical_columns:
    if col in heart_df_encoded.columns:
        # Convert to string first to handle mixed types
        heart_df_encoded[col] = heart_df_encoded[col].astype(str)
        le = LabelEncoder()
        heart_df_encoded[col] = le.fit_transform(heart_df_encoded[col])
        encoders[col] = le

# Save encoders for later use in prediction
with open('saved_models/encoders_heart.pkl', 'wb') as f:
    pickle.dump(encoders, f)

# Prepare features and target
X_h = heart_df_encoded.drop(columns=['HeartDisease'])
y_h = heart_df_encoded['HeartDisease']

with open('saved_models/heart_model_columns.pkl', 'wb') as f:
    pickle.dump(X_h.columns.tolist(), f)

# If target is also categorical, encode it
if y_h.dtype == 'object':
    target_encoder = LabelEncoder()
    y_h = target_encoder.fit_transform(y_h)
    label_encoders['HeartDisease'] = target_encoder
    print(f"Encoded HeartDisease: {dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))}")

# Verify the conversion worked
print(f"y_h dtype after encoding: {y_h.dtype}")
print(f"y_h unique values: {pd.Series(y_h).unique()}")
print("\nEncoded features shape:", X_h.shape)
print("Encoded features head:")
print(X_h.head())

# Check for NaN values
assert not X_h.isnull().values.any(), "Features contain NaN"
assert not pd.Series(y_h).isnull().values.any(), "Target contains NaN"

# Split the data
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_h, y_h, test_size=0.2, random_state=42)

# Train the model
heart_model = RandomForestClassifier(random_state=42)
heart_model.fit(X_train_h, y_train_h)

print (X_train_h.shape)
y_pred_h = heart_model.predict(X_test_h)
accuracy_h = accuracy_score(y_test_h, y_pred_h)
print(f"📊 Heart Disease Model Accuracy: {accuracy_h:.2f}")
print("🧾 Classification Report (Heart Disease):\n", classification_report(y_test_h, y_pred_h))

joblib.dump(heart_model, 'saved_models/heart_model.pkl')
print("✅ Saved: heart_model.pkl")