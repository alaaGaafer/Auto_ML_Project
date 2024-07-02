from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .models import usersData
import pandas as pd
# import from main function called detection
from preprocessing_Scripts.main import AutoClean



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
        problemtype = request.POST.get('problemtype')
        autocleanobj=AutoClean(toprocesseddf, response_variable, problemtype)
        df_copy, nulls_dict, outlier_info, imbalance_info, low_variance_columns, categorical_columns = autocleanobj.Detections()

        # serializable_nulls_dict = convert_dict_to_serializable(nulls_dict)
        #convert df_copy to json
        df_copy_json = df_copy.to_json(orient='records')
        # print('Nulls dict:', df_copy_json)
        # serialized_nulls_dict = json.dumps(nulls_dict, default=convert_float64_dtype)
        response_data= {'status': 'success', 'df_copy_json': df_copy_json} 

        # response_data_serializable = json.dumps(response_data, default=convert_float64_dtype)



        return JsonResponse(response_data, safe=False)
    else:
        return JsonResponse({'status': 'fail', 'message': 'Only POST method is allowed.'}, status=405)

@csrf_exempt
def preprocessingAll(request):
    if request.method == 'POST':
        # print ("the request is",request)
        uploaded_file = request.POST.get('dataset')
        data_list = json.loads(uploaded_file)
        df=pd.DataFrame(data_list)
        # print("the head is: ", df.head())
        istime_series = request.POST.get('isTimeSeries')
        response_variable = request.POST.get('responseVariable')
        problemtype = request.POST.get('problemtype')
        print("the problem type is",problemtype)

        autocleanobj=AutoClean(df, response_variable, problemtype)

        df_copy, nulls_dict, outlier_info, imbalance_info, low_variance_columns, categorical_columns = autocleanobj.Detections()
        fill_na_dict = {}
        for col in nulls_dict:
            fill_na_dict[col] = 'auto'

        # Handling outliers
        outliers_method_input = ('z_score', 'auto', 3)
        if imbalance_info:
            imb_instruction = "auto"
        else:
            imb_instruction=None
        Norm_method = "auto"
        low_actions = {}
        encoding_dict = {}

        for col in categorical_columns:
            encoding_dict[col] = 'auto'
        for col in low_variance_columns:
            encoding_dict[col] = 'auto'

        reduce = 'True'
        auto_reduce = 'True'
        num_components_to_keep = 3
        processed_data = autocleanobj.Handling_calls(fill_na_dict, outliers_method_input,
                                                  imb_instruction, Norm_method,
                                                  low_actions, encoding_dict, reduce, auto_reduce,
                                                  num_components_to_keep)
        return JsonResponse({'status': 'success', ' 