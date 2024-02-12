from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .models import usersdata
# Create your views here.



def inputvalidation(request):
    parameters = {}
    if request.method == 'POST':
        data = json.load(request.body)
        name = data.get('name')
        password = data.get('password')
        
        user=usersdata.objects.filter(name=name,password=password)
        if user.exists():
            parameters['status']='success'
            parameters['message']='User is valid'
        else:
            parameters['status']='failure'
            parameters['message']='User is invalid'

    # Add your logic here for other HTTP methods or return a response
    return JsonResponse(parameters)