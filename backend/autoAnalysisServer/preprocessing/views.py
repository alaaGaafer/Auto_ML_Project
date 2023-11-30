from django.shortcuts import render
from django.http import JsonResponse
import json
from preprocessing_Scripts.functions import *
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def my_function(request):
    parameters={}
    if request.method == 'POST':
        parameters = json.loads(request.body)
        parameters=dict(parameters)
        testobj=test_server()
        parameters=testobj.change_parameters(parameters)
        print('test_server().change_parameters(parameters)')
        parameters=JsonResponse(parameters)
        return parameters
# Create your views here.
