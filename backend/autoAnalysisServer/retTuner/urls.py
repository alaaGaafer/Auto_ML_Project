from django.urls import path
from . import views

urlpatterns = [
    path('check', views.inputvalidation, name='inputvalidation'),
    path('signup/', views.signup, name='signup')
]


