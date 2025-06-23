from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('heart/', views.heart_form, name='heart'),  # ✅ this fixes your issue
    path('predict_heart/', views.predict_heart, name='predict_heart'),
    path('diabetes/', views.diabetes_form, name='diabetes'),
    path('predict_diabetes/', views.predict_diabetes, name='predict_diabetes'),
]