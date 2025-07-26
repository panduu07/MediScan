from django.shortcuts import render
import pandas as pd
import joblib
import os
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import torch
import pickle
from django.shortcuts import render
from predictor.models import CustomNeuralNetResNet
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
# Base directory for model paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load your models
heart_model = joblib.load(os.path.join(base_dir, 'saved_models', 'heart_model.pkl'))

with open('saved_models/heart_model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

with open('saved_models/encoders_heart.pkl', 'rb') as f:
    encoders = pickle.load(f)
    
diabetes_model = joblib.load(os.path.join(base_dir, 'saved_models', 'diabetes_model.pkl'))



xray_model = CustomNeuralNetResNet(outputs_number=3)
state_dict = torch.load("saved_models/best_model.pth", map_location="cpu")

# Strip incompatible final layer
del state_dict['net.fc.weight']
del state_dict['net.fc.bias']

# Load remaining weights
xray_model.load_state_dict(state_dict, strict=False)
xray_model.eval()
xray_classes = ['Normal', 'Pneumonia', 'Virus']
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Home view
def home(request):
    return render(request, 'predictor/home.html')

# Heart disease prediction view
def predict_heart(request):
    result = None
    if request.method == 'POST':
        try:
            # Collect raw input values as a dict
            raw_data = {
                'bmi': float(request.POST.get('bmi')),
                'smoking': request.POST.get('smoking'),
                'alcohol_drinking': request.POST.get('alcohol_drinking'),
                'stroke': request.POST.get('stroke'),
                'physical_health': float(request.POST.get('physical_health')),
                'mental_health': float(request.POST.get('mental_health')),
                'diff_walking': request.POST.get('diff_walking'),
                'sex': request.POST.get('sex'),
                'age_category': request.POST.get('age_category'),
                'race': request.POST.get('race'),
                'diabetic': request.POST.get('diabetic'),
                'physical_activity': request.POST.get('physical_activity'),
                'gen_health': request.POST.get('gen_health'),
                'sleep_time': float(request.POST.get('sleep_time')),
                'asthma': request.POST.get('asthma'),
                'kidney_disease': request.POST.get('kidney_disease'),
                'skin_cancer': request.POST.get('skin_cancer')
            }

            # Create single-row DataFrame
            df_input = pd.DataFrame([raw_data])

            # Label encode using stored encoders
            for col, le in encoders.items():
                df_input[col] = le.transform(df_input[col].astype(str))

            # Reindex to match training features
            df_input = df_input.reindex(columns=model_columns, fill_value=0)

            prediction = heart_model.predict(df_input)
            result = "Positive for Heart Disease" if prediction[0] == 1 else "Negative for Heart Disease"

        except Exception as e:
            result = f"Error: {str(e)}"

    return render(request, 'predictor/heart.html', {'result': result})

# Diabetes prediction view
def predict_diabetes(request):
    result = None
    if request.method == 'POST':
        try:
            # Extracting values safely
            gender = request.POST.get('gender')
            age = request.POST.get('age')
            hypertension = request.POST.get('hypertension')
            heart_disease = request.POST.get('heart_disease')
            smoking_history = request.POST.get('smoking_history')
            bmi = request.POST.get('bmi')
            hba1c_level = request.POST.get('HbA1c_level')
            blood_glucose_level = request.POST.get('blood_glucose_level')

            # Check for any missing values
            if not all([gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c_level, blood_glucose_level]):
                raise ValueError("All fields must be filled out.")

            # Convert to correct types
            gender_map = {'Female': 0, 'Male': 1}
            smoking_map = {'never': 0, 'former': 1, 'current': 2, 'No Info': 3}

            gender_val = gender_map.get(gender)
            smoking_val = smoking_map.get(smoking_history.lower(), 3)

            with open('encoders.pkl', 'rb') as f:
                encoders = pickle.load(f)

            input_df = pd.DataFrame({
                'gender': [gender_val],
                'age': [float(age)],
                'hypertension': [int(hypertension)],
                'heart_disease': [int(heart_disease)],
                'smoking_history': [smoking_val],
                'bmi': [float(bmi)],
                'HbA1c_level': [float(hba1c_level)],
                'blood_glucose_level': [float(blood_glucose_level)]
            })

            # Apply saved encodings to categorical columns
            for col in ['gender', 'smoking_history']:
                if col in input_df.columns:
                    try:
                        input_df[col] = encoders[col].transform(input_df[col])
                    except ValueError as e:
                        print(f"Warning: Unknown category in {col}. Using default encoding.")
                        # Handle unknown categories by assigning a default value or most frequent class
                        input_df[col] = 0  # or use encoders[col].transform([most_frequent_class])[0]

            print(input_df.head())
            prediction = diabetes_model.predict(input_df.values)
            result = "Positive for Diabetes" if prediction[0] == 1 else "Negative for Diabetes"

        except Exception as e:
            result = f"Error: {str(e)}"

    return render(request, 'predictor/diabetes.html', {'result': result})



def predict_pneumonia(request):
    result = None
    if request.method =='GET':
        return render(request, 'predictor/pneumonia.html')
    if request.method == 'POST' and request.FILES.get('xray_image'):
        try:
            image_file = request.FILES['xray_image']
            image = Image.open(image_file).convert('RGB')
            image = image_transforms(image).unsqueeze(0)

            with torch.no_grad():
                output = xray_model(image)
                probs = F.softmax(output, dim=1)[0]
                predicted_class = torch.argmax(probs).item()

                result = {
                    'prediction': xray_classes[predicted_class],
                    'probabilities': list(zip(xray_classes, [round(p.item() * 100, 2) for p in probs]))
                }

        except Exception as e:
            result = {'error': f"Error during prediction: {e}"}

    return render(request, 'predictor/pneumonia.html', {'result': result})