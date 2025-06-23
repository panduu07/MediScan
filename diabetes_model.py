import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the diabetes dataset CSV file you downloaded
df = pd.read_csv('datasets/diabetes.csv')  # <-- put your actual path here

# Prepare features and target variable
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split into train and test sets (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Save the trained model as a pickle file
with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Diabetes model trained and saved as diabetes_model.pkl")