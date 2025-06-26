from django.shortcuts import render
import joblib
import os

# Base directory for model paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load your models
heart_model = joblib.load(os.path.join(base_dir, 'saved_models', 'heart_model.pkl'))
diabetes_model = joblib.load(os.path.join(base_dir, 'saved_models', 'diabetes_model.pkl'))

# Home view
def home(request):
    return render(request, 'predictor/home.html')

# Heart disease prediction view
def predict_heart(request):
    result = None
    if request.method == 'POST':
        try:
            # Extract and convert inputs
            bmi = float(request.POST.get('bmi'))
            yes_no_map = {'Yes': 1, 'No': 0}
            smoking = yes_no_map.get(request.POST.get('smoking'))
            alcohol_drinking = yes_no_map.get(request.POST.get('alcohol_drinking'))
            stroke = yes_no_map.get(request.POST.get('stroke'))
            diff_walking = yes_no_map.get(request.POST.get('diff_walking'))
            physical_activity = yes_no_map.get(request.POST.get('physical_activity'))
            asthma = yes_no_map.get(request.POST.get('asthma'))
            kidney_disease = yes_no_map.get(request.POST.get('kidney_disease'))
            skin_cancer = yes_no_map.get(request.POST.get('skin_cancer'))
            physical_health = float(request.POST.get('physical_health'))
            mental_health = float(request.POST.get('mental_health'))

            sex_map = {'Female': 0, 'Male': 1}
            sex = sex_map.get(request.POST.get('sex'))

            age_map = {
                '18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3,
                '40-44': 4, '45-49': 5, '50-54': 6, '55-59': 7,
                '60-64': 8, '65-69': 9, '70-74': 10, '75-79': 11,
                '80 or older': 12
            }
            age_category = age_map.get(request.POST.get('age_category'))

            race_map = {'White': 0, 'Black': 1, 'Asian': 2, 'Other': 3, 'Hispanic': 4}
            race = race_map.get(request.POST.get('race'))

            diabetic_map = {
                'Yes': 0,
                'No': 1,
                'No, borderline diabetes': 2,
                'Yes (during pregnancy)': 3
            }
            diabetic = diabetic_map.get(request.POST.get('diabetic'))

            gen_health_map = {
                'Excellent': 0,
                'Very good': 1,
                'Good': 2,
                'Fair': 3,
                'Poor': 4
            }
            gen_health = gen_health_map.get(request.POST.get('gen_health'))

            sleep_time = float(request.POST.get('sleep_time'))

            features = [[
                bmi, smoking, alcohol_drinking, stroke, physical_health, mental_health,
                diff_walking, sex, age_category, race, diabetic, physical_activity,
                gen_health, sleep_time, asthma, kidney_disease, skin_cancer
            ]]

            prediction = heart_model.predict(features)
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

            features = [[
                gender_val,
                float(age),
                int(hypertension),
                int(heart_disease),
                smoking_val,
                float(bmi),
                float(hba1c_level),
                float(blood_glucose_level),
                0,  # dummy value 1
                0,  # dummy value 2
                0,  # dummy value 3
                0,  # dummy value 4
                0   # dummy value 5
            ]]
            # TODO: Replace dummy values with actual form inputs if needed

            # Predict
            prediction = diabetes_model.predict(features)
            result = "Positive for Diabetes" if prediction[0] == 1 else "Negative for Diabetes"

        except Exception as e:
            result = f"Error: {str(e)}"

    return render(request, 'predictor/diabetes.html', {'result': result})

def predict_pneumonia(request):
    if request.method == 'POST':
        # Placeholder for image processing
        result = "Pneumonia prediction coming soon..."
        return render(request, 'predictor/pneumonia.html', {'result': result})
    
    return render(request, 'predictor/pneumonia.html')