from django.urls import path
from . import views

urlpatterns = [
    path('', views.login, name='login'), 
    path('home/', views.home, name='home'),  # your home page
    path('heart/', views.predict_heart, name='heart'),  # heart prediction URL
    path('diabetes/', views.predict_diabetes, name='predict_diabetes'),  # diabetes prediction URL
    path('pneumonia/', views.predict_pneumonia, name='predict_pneumonia'), 
]