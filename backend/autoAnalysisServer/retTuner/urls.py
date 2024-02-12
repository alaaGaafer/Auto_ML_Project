from django.urls import path
from . import views
urlpatterns = [
    path('check/data', views.inputvalidation, name='inputvalidation'),
]
