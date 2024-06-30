from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .models import usersData
import pandas as pd


# Create your views here.


@csrf_exempt
def signup(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        phone_number = data.get('phone_number')

        # Check if the email already exists in the database
        if usersData.objects.filter(email=email).exists():
            return JsonResponse({'status': 'failure', 'message': 'Email already exists'})

        # Create a new user
        user = usersData.objects.create(name=name, email=email, password=password, phone_number=phone_number)
        user.save()
        return JsonResponse({'status': 'success', 'message': 'User successfully added to the database'})

    return JsonResponse({'status': 'failure', 'message': 'Invalid request'})


@csrf_exempt
def inputvalidation(request):
    parameters = {}
    if request.method == 'POST':
        # print('test1')
        data = json.loads(request.body.decode('utf-8'))
        name = data.get('name')
        password = data.get('password')
        # print('test2')
        user = usersData.objects.filter(email=name, password=password)
        if user.exists():
            parameters['status'] = 'success'
            parameters['message'] = 'User is valid'
        else:
            parameters['status'] = 'failure'
            parameters['message'] = 'User is invalid'

    # Add your logic here for other HTTP methods or return a response
    return JsonResponse(parameters)
@csrf_exempt
def notify(request):
    if request.method == 'POST':

        uploaded_file = request.FILES['dataset']
        file_name = uploaded_file.name
        file=pd.read_csv(uploaded_file)
        print(file.head())

        response_variable = request.POST.get('responseVariable')
        is_time_series = request.POST.get('isTimeSeries')

        print('File name:', file_name)
        print('Response Variable:', response_variable)
        print('Is Time Series:', is_time_series)



        return JsonResponse({'status': 'success', 'fileName': file_name})
    else:
        return JsonResponse({'status': 'fail', 'message': 'Only POST method is allowed.'}, status=405)

