from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .models import usersData

# Create your views here.



@csrf_exempt
def inputvalidation(request):
    parameters = {}
    if request.method == 'POST':
        # print('test1')
        data = json.loads(request.body.decode('utf-8'))
        name = data.get('name')
        password = data.get('password')
        # print('test2')
        user=usersData.objects.filter(email=name,password=password)
        if user.exists():
            parameters['status']='success'
            parameters['message']='User is valid'
        else:
            parameters['status']='failure'
            parameters['message']='User is invalid'

    # Add your logic here for other HTTP methods or return a response
    return JsonResponse(parameters)