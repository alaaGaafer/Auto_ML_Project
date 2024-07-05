from django.db import models


class usersData(models.Model):
    name = models.CharField(max_length=100)
    email = models.CharField(max_length=100)
    password = models.CharField(max_length=100)
    phone = models.CharField(max_length=100, primary_key=True)
class organizationData(models.Model):
    organizationID = models.CharField(primary_key=True,max_length=100)
    name = models.CharField(max_length=100)
    description = models.TextField()
class userOrganizationData(models.Model):
    organizationID = models.ForeignKey(organizationData, on_delete=models.CASCADE)
    phone = models.ForeignKey(usersData, on_delete=models.CASCADE)
class datasetsData(models.Model):
    datasetID = models.CharField(primary_key=True,max_length=100)
    phone = models.ForeignKey(usersData, on_delete=models.CASCADE)
    datasetName = models.CharField(max_length=100)
    problemType = models.CharField(max_length=100)
    description = models.TextField()
    modelname = models.CharField(max_length=100)
    date= models.CharField(max_length=100)
    responseVariable = models.CharField(max_length=100)
class preprocessMethodData(models.Model):
    methodID = models.CharField(primary_key=True,max_length=100)
    name = models.CharField(max_length=100)
    description = models.TextField()
    avgComplexity = models.FloatField()
class preprocessedByData(models.Model):
    datasetID = models.ForeignKey(datasetsData, on_delete=models.CASCADE)
    methodID = models.ForeignKey(preprocessMethodData, on_delete=models.CASCADE)
    arguments = models.TextField()
class metaFeaturesData(models.Model):
    datasetID = models.ForeignKey(datasetsData, on_delete=models.CASCADE)
    numberOfInstances = models.IntegerField()
    logNumberOfInstances = models.FloatField()
    logNumberOfFeatures = models.FloatField()
    numberOfClasses = models.IntegerField()
    numberOfCategoricalFeatures = models.IntegerField()
    numberOfNumericalFeatures = models.IntegerField()
    ratioOfNumericFeatures = models.FloatField()
    ratioOfCategoricalFeatures = models.FloatField()
    classEntropy = models.FloatField()
    classProbability = models.FloatField()
    symbolsMean = models.FloatField()
    skewnessMean = models.FloatField()
    kurtosisMean = models.FloatField()
    datasetRatio = models.FloatField()
class evaluationScoreData(models.Model):
    datasetID = models.ForeignKey(datasetsData, on_delete=models.CASCADE)
    stratiID = models.IntegerField()
    score = models.FloatField()
class hyperParametersData(models.Model):
    datasetID = models.ForeignKey(datasetsData, on_delete=models.CASCADE)
    hyperParameterName = models.CharField(max_length=500)
    value= models.CharField(max_length=500)
class modelData(models.Model):
    modelID = models.CharField(primary_key=True,max_length=100)
    modelName = models.CharField(max_length=100)
    classPath = models.CharField(max_length=255)
    task = models.CharField(max_length=100)
    description = models.TextField()
    avgComplexity = models.FloatField()
class evaluationStrategyData(models.Model):
    stratiID = models.IntegerField(primary_key=True)
    modelID = models.ForeignKey(modelData, on_delete=models.CASCADE)
    stratName = models.CharField(max_length=100)
    description = models.TextField()
    reliableThreshold = models.FloatField()
    category = models.CharField(max_length=100)
    avgComplexity = models.FloatField()
class obtainsData(models.Model):
    datasetID = models.ForeignKey(datasetsData, on_delete=models.CASCADE)
    methodID = models.ForeignKey(preprocessMethodData, on_delete=models.CASCADE)
    hyperParameterName = models.ForeignKey(hyperParametersData, on_delete=models.CASCADE)
    value = models.IntegerField()
# Create your models here.
