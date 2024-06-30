from django.urls import path
from . import views

urlpatterns = [
    path('check', views.inputvalidation, name='inputvalidation'),
    path('notify', views.notify, name='notify'),
    path('signup/', views.signup, name='signup')
]


