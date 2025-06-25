import torch
import pickle
from django.shortcuts import render
from predictor.models import CustomNeuralNetResNet
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

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

heart_model = pickle.load(open('saved_models/heart_disease_model.pkl', 'rb'))
diabetes_model = pickle.load(open('saved_models/diabetes_model.pkl', 'rb'))

def home(request):
    return render(request, 'predictor/home.html')

def heart_form(request):
    return render(request, 'predictor/heart.html')

def diabetes_form(request):
    return render(request, 'predictor/diabetes.html')

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


def predict_xray(request):
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