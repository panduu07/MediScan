import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the heart disease dataset CSV file you downloaded
df = pd.read_csv('datasets/heart.csv')  # <-- put your actual path here

# Prepare features and target variable
X = df.drop('target', axis=1)
y = df['target']

# Split into train and test sets (optional, for evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Save the trained model as a pickle file
with open('heart_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Heart disease model trained and saved as heart_model.pkl")