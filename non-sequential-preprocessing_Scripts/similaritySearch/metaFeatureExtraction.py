import os
import pandas as pd
import numpy as np


class metafeatureExtraction:
    def __init__(self,df):
        self.df=df
        self.KBPath='knowledgeBase.csv'
    # 1- number of instances

    def getNumberOfFeaturesandTheirlog(self):
        numberofFeatres=len(self.df.columns)
        logofFeatures=np.log(numberofFeatres)
        return numberofFeatres,logofFeatures
    #2- log number of instances

    def getNumberOfInstancesandTheirlog(self):
        numberofInstances=len(self.df)
        logofInstances=np.log(numberofInstances)
        return numberofInstances,logofInstances
    #3- number of classes
    def getnumberofClasses(self):
        numberofClasses=len(self.df['Target'].unique())
        return numberofClasses
    #4- number of numerical features
    def getnumberofNumericalFeatures(self):
        numberofNumericalFeatures=len(self.df.select_dtypes(include=[np.number]).columns)
        return numberofNumericalFeatures
    #5- number of categorical features
    def getnumberofCategoricalFeatures(self):
        numberofCategoricalFeatures=len(self.df.select_dtypes(exclude=['number']).columns)
        return numberofCategoricalFeatures
    #6- ratio of numerical to categorical features
    def ratioofNumericaltoCategoricalFeatures(self):
        numberofNumericalFeatures=len(self.df.select_dtypes(include=[np.number]).columns)
        numberofCategoricalFeatures=len(self.df.select_dtypes(exclude=['number']).columns)
        ratio=numberofNumericalFeatures/numberofCategoricalFeatures
        return ratio
    #7- number class entropy
    def classeEntropy(self):
        entropy=-np.sum(self.df['Target'].value_counts(normalize=True) * np.log2(self.df['Target'].value_counts(normalize=True)))
        return entropy
    #8- class probability max
    def classProbabilityMax(self):
        return self.df['Target'].value_counts(normalize=True).max()
    #9- class probability min
    def classProbabilityMin(self):
        return self.df['Target'].value_counts(normalize=True).min()
    #10- class probability mean
    def classProbabilityMean(self):
        return self.df['Target'].value_counts(normalize=True).mean()
    #11- class probability std
    def classProbabilityStd(self):
        return self.df['Target'].value_counts(normalize=True).std()
    #12- symbols mean
    def symbolsMean(self):
        return self.df.describe().loc['mean'].mean()
    #13- symbols sum
    def symbolsSum(self):
        return self.df.describe().loc['sum'].sum()
    #14- symbols std
    def symbolsStd(self):
        return self.df.describe().loc['std'].std()
    #15- skewness min
    def skewnessMin(self):
        return self.df.skew().min()
    #16- skewness max
    def skewnessMax(self):
        return self.df.skew().max()
    #17- skewness mean
    def skewnessMean(self):
        return self.df.skew().mean()
    #18- skewness std
    def skewnessStd(self):
        return self.df.skew().std()
    #19- kurtosis min
    def kurtosisMin(self):
        return self.df.kurtosis().min()
    #20- kurtosis max
    def kurtosisMax(self):
        return self.df.kurtosis().max()
    #21- kurtosis mean
    def kurtosisMean(self):
        return self.df.kurtosis().mean()
    #22- kurtosis std
    def kurtosisStd(self):
        return self.df.kurtosis().std()
    #23- Dataset Ratio of number of features to number of instances
    def DatasetRatioofNumberofFeaturestoNumberofInstances(self):
        return len(self.df.columns)/len(self.df)
    def getMetaFeatures(self):
        numberofFeatres,logofFeatures=self.getNumberOfFeaturesandTheirlog()
        numberofInstances,logofInstances=self.getNumberOfInstancesandTheirlog()
        numberofClasses=self.getnumberofClasses()
        numberofNumericalFeatures=self.getnumberofNumericalFeatures()
        numberofCategoricalFeatures=self.getnumberofCategoricalFeatures()
        ratio=self.ratioofNumericaltoCategoricalFeatures()
        entropy=self.classeEntropy()
        classprobmax=self.classProbabilityMax()
        classprobmin=self.classProbabilityMin()
        classprobmean=self.classProbabilityMean()
        classprobstd=self.classProbabilityStd()
        symbolsmean=self.symbolsMean()
        symbolssum=self.symbolsSum()
        symbolsstd=self.symbolsStd()
        skewnessmin=self.skewnessMin()
        skewnessmax=self.skewnessMax()
        skewnessmean=self.skewnessMean()
        skewnessstd=self.skewnessStd()
        kurtosismin=self.kurtosisMin()
        kurtosismax=self.kurtosisMax()
        kurtosismean=self.kurtosisMean()
        kurtosisstd=self.kurtosisStd()
        DatasetRatioofNumberofFeaturestoNumberofInstances=self.DatasetRatioofNumberofFeaturestoNumberofInstances()
        return numberofFeatres,logofFeatures,numberofInstances,logofInstances,numberofClasses,numberofNumericalFeatures,numberofCategoricalFeatures,ratio,entropy,\
                classprobmax,classprobmin,classprobmean,classprobstd,symbolsmean,symbolssum,symbolsstd,skewnessmin,skewnessmax,skewnessmean,skewnessstd,\
                kurtosismin,kurtosismax,kurtosismean,kurtosisstd,DatasetRatioofNumberofFeaturestoNumberofInstances
    def addToKnowledgeBase(self):
        numberofFeatres,logofFeatures,numberofInstances,logofInstances,numberofClasses,numberofNumericalFeatures,numberofCategoricalFeatures,ratio,entropy\
        ,classprobmax,classprobmin,classprobmean,classprobstd,symbolsmean,symbolssum,symbolsstd,skewnessmin,skewnessmax,skewnessmean,skewnessstd\
        ,kurtosismin,kurtosismax,kurtosismean,kurtosisstd,DatasetRatioofNumberofFeaturestoNumberofInstances=self.getMetaFeatures()
        # print(self.KBPath)
        #print kb type
        # print(type(self.KBPath))
        #if file is not exist
        try:
            knowledgeBase=pd.read_csv(self.KBPath)
        except:
            knowledgeBase=pd.DataFrame(columns=['numberofFeatres','logofFeatures','numberofInstances','logofInstances'\
                                                ,'numberofClasses','numberofNumericalFeatures','numberofCategoricalFeatures','ratio','entropy','classprobmax','classprobmin'\
                                                    ,'classprobmean','classprobstd','symbolsmean','symbolssum','symbolsstd','skewnessmin','skewnessmax','skewnessmean','skewnessstd','kurtosismin'\
                                                        ,'kurtosismax','kurtosismean','kurtosisstd','DatasetRatioofNumberofFeaturestoNumberofInstances'])
        newrow=pd.DataFrame([[numberofFeatres,logofFeatures,numberofInstances,logofInstances,numberofClasses,
                              numberofNumericalFeatures,numberofCategoricalFeatures,ratio,entropy,classprobmax,classprobmin,classprobmean,classprobstd,symbolsmean,symbolssum,symbolsstd,skewnessmin,skewnessmax,
                              skewnessmean,skewnessstd,kurtosismin,kurtosismax,kurtosismean,kurtosisstd,DatasetRatioofNumberofFeaturestoNumberofInstances]]
                            ,columns=['numberofFeatres','logofFeatures','numberofInstances','logofInstances',
                                      'numberofClasses','numberofNumericalFeatures','numberofCategoricalFeatures','ratio','entropy'
                                      ,'classprobmax','classprobmin','classprobmean','classprobstd','symbolsmean','symbolssum','symbolsstd','skewnessmin','skewnessmax','skewnessmean','skewnessstd','kurtosismin'
                                      ,'kurtosismax','kurtosismean','kurtosisstd','DatasetRatioofNumberofFeaturestoNumberofInstances'])
        knowledgeBase=pd.concat([knowledgeBase,newrow],ignore_index=True)
        #if the file exists
        try:
            knowledgeBase.to_csv(self.KBPath,index=False)
        except:
            #delete the file and create a new one
            os.remove(self.KBPath)
            knowledgeBase.to_csv(self.KBPath,index=False)