from django.urls import path
from . import views

urlpatterns = [
    path('check', views.inputvalidation, name='inputvalidation'),
    path('notify', views.notify, name='notify'),
    path('preprocessingAll', views.preprocessingAll, name='preprocessingAll'),
    path('trainCurrentdata',views.trainCurrentdata,name='trainCurrentdata'),
    path('predict', views.predict, name='predict'),
    path('register', views.signup, name='signup'),
    path('handlenulls', views.handlenulls, name='handlenulls'),
    path('varianceThreshold', views.handlelowvar, name='handlelowvar'),
    path('removeID', views.removeID, name='removeID'),
    path('handleCategorical', views.Encode, name='handleCategorical'),
]


