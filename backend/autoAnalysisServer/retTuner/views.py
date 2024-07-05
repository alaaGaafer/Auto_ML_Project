from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .models import usersData, datasetsData
import pandas as pd
import base64
from preprocessing_Scripts.main import AutoClean
from datetime import date
from preprocessing_Scripts.trying import calculate_date_frequency,Detections_,_process_data,Cleaning,user_interaction
from preprocessing_Scripts.bestmodel import Bestmodel
from preprocessing_Scripts.cashAlgorithm.smacClass import ProblemType

# Create your views here.
def getPhoto(phone):
    imagepath = 'preprocessing_Scripts/media/' + phone + '.jpeg'
    image=open(imagepath, 'rb')
    return image

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
            user=user[0]
            phone=user.phone
            username = user.name
            userimage_file = getPhoto(phone)
            # print(userimage_file.read())
            userimage = base64.b64encode(userimage_file.read()).decode('utf-8')
            datasets = datasetsData.objects.filter(phone=phone)
            print(list(datasets.values()))
            print(datasets)
            datasets_json = json.dumps(list(datasets.values()))
            parameters['status'] = 'success'
            parameters['username'] = username
            parameters['datasets'] = datasets_json
            parameters['userimage'] = userimage
            parameters['phone'] = phone
        else:
            parameters['status'] = 'failure'
            parameters['message'] = 'User is invalid'

    # Add your logic here for other HTTP methods or return a response
    return JsonResponse(parameters)
@csrf_exempt
def notify(request):
    if request.method == 'POST':

        uploaded_file = request.FILES['dataset']
        if uploaded_file.name.endswith('.csv'):
            toprocesseddf=pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            toprocesseddf=pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.xls'):
            toprocesseddf=pd.read_excel(uploaded_file)
        # toprocesseddf=pd.read_csv(uploaded_file)
        #make variable called the datasetid which is 'ds+maxidindex+1'
        #make date vairable which is today's date from datetime library
        
        file_name = uploaded_file.name
        datasetid = 'ds' + str(datasetsData.objects.all().count() + 1)
        today = date.today()
        phone = request.POST.get('phone')
        problemtype = request.POST.get('problemtype')
        description = request.POST.get('description')
        response_variable = request.POST.get('responseVariable')
        #add theseData to datasetsData
        print(phone)
        dataset = datasetsData.objects.create(datasetID=datasetid, phone=usersData.objects.get(phone=phone), datasetName=file_name, problemType=problemtype, description=description, modelname='None',date=today, responseVariable=response_variable)
        dataset.save()
        df_copy_json = toprocesseddf.to_json(orient='records')
        response_data= {'status': 'success', 'df_copy_json': df_copy_json,'datasetid': datasetid} 
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
        datasetid = request.POST.get('datasetid')
        # print("the problem type is",problemtype)
        print(problemtype)
        problemtype = problemtype.lower()
        if problemtype=='timeseries':
            trainData, testData,frequency = user_interaction(df,problemtype,response_variable,date_col='Date')
            choosenModels=['Arima','Sarima']
            dummy='lol'
            Bestmodelobj=Bestmodel(ProblemType.TIME_SERIES,choosenModels,dummy,dummy,trainData,testData,frequency)
            Bestmodelobj.splitTestData()
            Bestmodelobj.trainModels()
            
        
def getsavedmodels(request):
    if request.method == 'POST':
        pass