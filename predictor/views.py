import numpy as np
import pickle
from django.shortcuts import render

def heart_form(request):
    return render(request, 'predictor/heart.html')
from django.shortcuts import render

# ... (your other imports and views)

def diabetes_form(request):
    return render(request, 'predictor/diabetes.html')

# Load the models once on server start
heart_model = pickle.load(open('saved_models/heart_disease_model.pkl', 'rb'))
diabetes_model = pickle.load(open('saved_models/diabetes_model.pkl', 'rb'))

def home(request):
    return render(request, 'predictor/home.html')

def predict_heart(request):
    result = None
    if request.method == 'POST':
        try:
            features = [
               # age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target

                float(request.POST.get('age', 0)),
                float(request.POST.get('sex', 0)),
                float(request.POST.get('cp', 0)),
                float(request.POST.get('trestbps', 0)),
                float(request.POST.get('chol', 0)),
                float(request.POST.get('fbs', 0)),
                float(request.POST.get('restecg', 0)),
                float(request.POST.get('thalach', 0)),
                float(request.POST.get('exang', 0)),
                float(request.POST.get('oldpeak', 0)),
                float(request.POST.get('slope', 0)),
                float(request.POST.get('ca', 0)),
                float(request.POST.get('thal', 0)),
            ]
            print (features)
            prediction = heart_model.predict([features])[0]
            result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        except Exception as e:
            result = f"Error during prediction: {e}"
    print (result)
    return render(request, 'predictor/heart.html', {'result': result})

def predict_diabetes(request):
    result = None
    if request.method == 'POST':
        try:
            features = [
                float(request.POST.get('Pregnancies', 0)),
                float(request.POST.get('Glucose', 0)),
                float(request.POST.get('BloodPressure', 0)),
                float(request.POST.get('SkinThickness', 0)),
                float(request.POST.get('Insulin', 0)),
                float(request.POST.get('BMI', 0)),
                float(request.POST.get('DiabetesPedigreeFunction', 0)),
                float(request.POST.get('Age', 0)),
            ]
            prediction = diabetes_model.predict([features])[0]
            result = "Diabetes Detected" if prediction == 1 else "No Diabetes"
        except Exception as e:
            result = f"Error during prediction: {e}"
    return render(request, 'predictor/diabetes.html', {'result': result})