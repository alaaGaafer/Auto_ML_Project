from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .models import usersData
import pandas as pd
#import from main function called detection
from preprocessing_Scripts.main import Detections


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
        toprocesseddf=pd.read_csv(uploaded_file)

        response_variable = request.POST.get('responseVariable')
        is_time_series = request.POST.get('isTimeSeries')

        df_copy, nulls_dict, outlier_info, duplicates, imbalance_info, numerical_columns, low_variance_columns, low_variance_info, categorical_columns, deletion_messages, carednality_messages = Detections(toprocesseddf, response_variable)

        # serializable_nulls_dict = convert_dict_to_serializable(nulls_dict)
        #convert df_copy to json
        df_copy_json = df_copy.to_json(orient='records')
        print('Nulls dict:', df_copy_json)
        # serialized_nulls_dict = json.dumps(nulls_dict, default=convert_float64_dtype)
        response_data= {'status': 'success', 'df_copy_json': df_copy_json} 

        # response_data_serializable = json.dumps(response_data, default=convert_float64_dtype)



        return JsonResponse(response_data, safe=False)
    else:
        return JsonResponse({'status': 'fail', 'message': 'Only POST method is allowed.'}, status=405)


        # def convert_float64_dtype(obj):
        #     if isinstance(obj, pd.Float64Dtype):
        #         return float('NaN')  # Convert to NaN or appropriate value
        #     return obj

        # def convert_dict_to_serializable(d):
        #     serializable_dict = {}
        #     for key, value in d.items():
        #         serializable_value = {
        #             'type': str(value['type']),  # Convert dtype to string
        #             'number_of_nulls': value['number_of_nulls'],
        #             'locations_of_nulls': value['locations_of_nulls']
        #         }
        #         serializable_dict[key] = serializable_value
        #     return serializable_dict