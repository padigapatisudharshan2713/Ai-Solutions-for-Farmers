from django.contrib import admin
from django.urls import path
from . import views

urlpatterns=[
    path("",views.index,name='index'),
    path("about/",views.about,name='about'),
    path('register/', views.register, name='register'),
    path('login/', views.login, name='login'),
    path('home/', views.home, name='home'),
    path("soilprediction/",views.soilprediction,name="soilprediction"),
    path("croppredictiopn/",views.croppredictiopn,name="croppredictiopn"),
    path("plantprediction/",views.plantprediction,name="plantprediction"),

]