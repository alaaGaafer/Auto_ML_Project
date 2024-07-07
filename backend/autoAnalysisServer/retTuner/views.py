from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

from sklearn.model_selection import train_test_split
from .models import usersData, datasetsData
import pandas as pd
import base64
from preprocessing_Scripts.main import AutoClean
from datetime import date
from preprocessing_Scripts.trying import calculate_date_frequency,Detections_,_process_data,Cleaning,user_interaction
from preprocessing_Scripts.bestmodel import Bestmodel
from preprocessing_Scripts.cashAlgorithm.smacClass import ProblemType
import PIL
from PIL import Image
from io import BytesIO
from preprocessing_Scripts.similaritySearch.functions import *

# Create your views here.
def getPhoto(phone):
    try:
        imagepath = 'preprocessing_Scripts/media/' + phone + '.jpeg'
        image=open(imagepath, 'rb')
        return image
    except:
        #jpg
        imagepath = 'preprocessing_Scripts/media/' + phone + '.jpg'
        image=open(imagepath, 'rb')
    return image
def savephoto(phone,photo):
    #check extension
    image = Image.open(photo)
    if photo.name.endswith('.jpeg'):
        image.save('preprocessing_Scripts/media/' + phone + '.jpeg')
    elif photo.name.endswith('.jpg'):
        image.save('preprocessing_Scripts/media/' + phone + '.jpg')



@csrf_exempt
def signup(request):
    if request.method == 'POST':
        # data = json.loads(request.body.decode('utf-8'))
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('cpass')
        phone_number = request.POST.get('phone')
        print(name)

        try:
            photo=request.FILES['photoupload']
            savephoto(phone_number,photo)
            
        except:
            print('no photo')

        # Check if the email already exists in the database
        if usersData.objects.filter(email=email).exists():
            return JsonResponse({'status': 'failure', 'message': 'Email already exists'})

        # Create a new user
        user = usersData.objects.create(name=name, email=email, password=password, phone=phone_number)
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
            # print(list(datasets.values()))
            # print(datasets)
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
        dataset = datasetsData.objects.create(datasetID=datasetid, phone=usersData.objects.get(phone=phone), datasetName=file_name, problemType=problemtype, description=description, modelname='None',date=today, responseVariable=response_variable, modelaccuracy=0, modelmse=0)
        dataset.save()
        df_copy_json = toprocesseddf.to_json(orient='records')
        response_data= {'status': 'success', 'df_copy_json': df_copy_json,'datasetid': datasetid} 
        return JsonResponse(response_data, safe=False)
    else:
        return JsonResponse({'status': 'fail', 'message': 'Only POST method is allowed.'}, status=405)
@csrf_exempt
def handlenulls(request):
    if request.method == 'POST':
        #formData.append("dataset", datasetjson);
    # formData.append("responseVariable", modelData.responseVariable);
    # formData.append("imputationMethod", imputationMethod);
    # formData.append("problemtype", modelData.problemtype);
        uploaded_file = request.POST.get('dataset')
        imputationmethod=request.POST.get('imputationMethod')
        responsevariable=request.POST.get('responseVariable')
        problemtype=request.POST.get('problemtype')
        print(imputationmethod)
        data_list = json.loads(uploaded_file)
        df=pd.DataFrame(data_list)
        imputationmethod=imputationmethod.lower()
        df_cleaned =MissingValues().handle_nan(df, imputationmethod)
        #remove the first column
        # df = df.drop(df.columns[0], axis=1)
        newdfjson=df_cleaned.to_json(orient='records')
        return JsonResponse({'status': 'success', 'newdf': newdfjson})
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
        # istime_series = request.POST.get('isTimeSeries')
        response_variable = request.POST.get('responseVariable')
        problemtype = request.POST.get('problemtype')
        datasetid = request.POST.get('datasetid')
        # print("the problem type is",problemtype)
        # print(problemtype)
        problemtype = problemtype.lower()
        if problemtype=='timeseries':
            trainData, testData,frequency = user_interaction(df,problemtype,response_variable,date_col='Date')
            choosenModels=['Arima','Sarima']
            dummy='lol'
            Bestmodelobj=Bestmodel(ProblemType.TIME_SERIES,choosenModels,dummy,dummy,trainData,testData,frequency)

        elif problemtype =='classification':
            choosenModels=["KNN", "LR", "RF"]
            x_train, y_train, x_test, y_test = user_interaction(df, problemtype, response_variable, date_col=None)
            Bestmodelobj = Bestmodel(ProblemType.CLASSIFICATION, choosenModels, x_train,x_test,y_train,y_test)
        elif problemtype == 'regression':
            x_train, y_train, x_test, y_test = user_interaction(df, problemtype, response_variable, date_col="Date")
            choosenModels = ['LinearRegression', "Lasso",'Ridge','RF','XGboost']
            Bestmodelobj = Bestmodel(ProblemType.REGRESSION, choosenModels, x_train, x_test, y_train, y_test)
        
        Bestmodelobj.splitTestData()
        Bestmodelobj.TrainModel()
        modelname = Bestmodelobj.modelstr
        modelmse = Bestmodelobj.mse
        modelaccuracy = Bestmodelobj.accuracy
        Bestmodelobj.saveModel(datasetid)

        # print("modelname",modelname)
        # print("modelmse",modelmse)
        # print("modelaccuracy",modelaccuracy)
        dataset = datasetsData.objects.get(datasetID=datasetid)
        print(modelmse)
        print(modelaccuracy)
        if modelmse is  None:
            modelmse = 0
        if modelaccuracy is None:
            modelaccuracy = 0
        dataset.modelname = modelname
        dataset.modelmse = modelmse
        dataset.modelaccuracy = modelaccuracy
        dataset.save()
        phone=dataset.phone
        datasets = datasetsData.objects.filter(phone=phone)
            # print(list(datasets.values()))
            # print(datasets)
        datasets_json = json.dumps(list(datasets.values()))

        
        # print("the model name is",modelname)
        return JsonResponse({'status': 'success', 'accuracy': modelaccuracy, 'mse': modelmse, 'modelname': modelname, 'datasets': datasets_json})
    else:
        return JsonResponse({'status': 'fail', 'message': 'Only POST method is allowed.'}, status=405)
@csrf_exempt
def predict(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['dataset']
        datasetid = request.POST.get('datasetid')
        dataset = datasetsData.objects.get(datasetID=datasetid)
        Problemtype = dataset.problemType
        print("the problem type is",Problemtype)
        responsevariable=dataset.responseVariable
        Bestmodelobj = Bestmodel(problemtype=Problemtype)
        print("theuploadedfile is",uploaded_file.name)
        Bestmodelobj.loadModel(datasetid)
        df=pd.read_csv(uploaded_file)
        print(df.head())
        if Problemtype == 'timeseries':
            Bestmodelobj=Bestmodel(ProblemType.TIME_SERIES)
        elif Problemtype == 'classification':
            Bestmodelobj=Bestmodel(ProblemType.CLASSIFICATION)
        elif Problemtype == 'regression':
            Bestmodelobj=Bestmodel(ProblemType.REGRESSION)
        Bestmodelobj.loadModel(datasetid)
        concateddf=Bestmodelobj.PredictModel(df)
        print(concateddf.head())
        #change every boolean true or false to 0 or 1
        concateddf=concateddf*1
        jsondf=concateddf.to_json(orient='records')

        response_data= {'status': 'success', 'df_copy_json': jsondf} 
        return JsonResponse(response_data, safe=False)
@csrf_exempt
def trainCurrentdata(request):
    if request.method == 'POST':
        uploaded_file = request.POST.get('dataset')
        data_list = json.loads(uploaded_file)
        df=pd.DataFrame(data_list)
        response_variable = request.POST.get('responseVariable')
        problemtype = request.POST.get('problemtype')
        datasetid = request.POST.get('datasetid')
        problemtype = problemtype.lower()
        if problemtype=='timeseries':
            split_ratio= 0.8
            split_index = int(len(df)*split_ratio)
            train_data = df[:split_index]
            test_data = df[split_index:]
            print("the data types are",df.dtypes)
            # Bestmodelobj=Bestmodel(ProblemType.TIME_SERIES,choosenModels,dummy,dummy,trainData,testData,frequency)
            Bestmodelobj = Bestmodel(ProblemType.TIME_SERIES, ['Arima', 'Sarima'], train_data, test_data, calculate_date_frequency(train_data))
        elif problemtype =='classification':
            #getx and y
            x=df.drop(response_variable,axis=1)
            y=df[response_variable]
            x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
            Bestmodelobj = Bestmodel(ProblemType.CLASSIFICATION, ['KNN', 'LR', 'RF'], x_train, x_test, y_train, y_test)

            print("the data types are",df.dtypes)
        elif problemtype == 'regression':
            x=df.drop(response_variable,axis=1)
            y=df[response_variable]
            x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
            Bestmodelobj = Bestmodel(ProblemType.REGRESSION, ['LinearRegression', 'Lasso', 'Ridge', 'RF', 'XGboost'], x_train, x_test, y_train, y_test)
            print("the data types are",df.dtypes)
        Bestmodelobj.splitTestData()
        Bestmodelobj.TrainModel()
        modelname = Bestmodelobj.modelstr
        modelmse = Bestmodelobj.mse
        modelaccuracy = Bestmodelobj.accuracy
        Bestmodelobj.saveModel(datasetid)
        print("datasetid",datasetid)
        dataset = datasetsData.objects.get(datasetID=datasetid)
        print(modelmse)
        print(modelaccuracy)
        if modelmse is  None:
            modelmse = 0
        if modelaccuracy is None:
            modelaccuracy = 0
        dataset.modelname = modelname
        dataset.modelmse = modelmse
        dataset.modelaccuracy = modelaccuracy
        dataset.save()
        phone=dataset.phone
        print(phone)
        datasets = datasetsData.objects.filter(phone=phone)
            # print(list(datasets.values()))
            # print(datasets)
        datasets_json = json.dumps(list(datasets.values()))
        return JsonResponse({'status': 'success', 'accuracy': modelaccuracy, 'mse': modelmse, 'modelname': modelname, 'datasets': datasets_json})
    else:
        return JsonResponse({'status': 'fail', 'message': 'Only POST method is allowed.'}, status=405)
        
        # print("the data types are",df.dtypes)


@csrf_exempt
def handlelowvar(request):
    if request.method == 'POST':
        uploaded_file = request.POST.get('dataset')
        imputationMethod = request.POST.get('imputationMethod')
        data_list = json.loads(uploaded_file)
        df=pd.DataFrame(data_list)
        low_columns, low_info=HandlingColinearity().detect_low_variance(df)
        df=HandlingColinearity().handle_low_variance(df,low_columns,imputationMethod)
        print(df.head())
        # df_cleaned =LowVariance().handle_low_variance(df, threshold)
        newdfjson=df.to_json(orient='records')
        return JsonResponse({'status': 'success', 'newdf': newdfjson})
    else:
        return JsonResponse({'status': 'fail', 'message': 'Only POST method is allowed.'}, status=405)
        
def getsavedmodels(request):
    if request.method == 'POST':
        pass