from django.urls import path
from . import views

urlpatterns = [
    path('check', views.inputvalidation, name='inputvalidation'),
    path('notify', views.notify, name='notify'),
    path('preprocessingAll', views.preprocessingAll, name='preprocessingAll'),
    # path('getsavedmodels', views.getsavedmodels, name='getsavedmodels'),
    path('predict', views.predict, name='predict'),
    path('register', views.signup, name='signup'),
]


