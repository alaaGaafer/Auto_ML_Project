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
    def getMetaFeatures(self):
        numberofFeatres,logofFeatures=self.getNumberOfFeaturesandTheirlog()
        numberofInstances,logofInstances=self.getNumberOfInstancesandTheirlog()
        numberofClasses=self.getnumberofClasses()
        numberofNumericalFeatures=self.getnumberofNumericalFeatures()
        numberofCategoricalFeatures=self.getnumberofCategoricalFeatures()
        ratio=self.ratioofNumericaltoCategoricalFeatures()
        entropy=self.classeEntropy()
        return numberofFeatres,logofFeatures,numberofInstances,logofInstances,numberofClasses,numberofNumericalFeatures,numberofCategoricalFeatures,ratio,entropy
    def addToKnowledgeBase(self):
        numberofFeatres,logofFeatures,numberofInstances,logofInstances,numberofClasses,numberofNumericalFeatures,numberofCategoricalFeatures,ratio,entropy=self.getMetaFeatures()
        # print(self.KBPath)
        #print kb type
        # print(type(self.KBPath))
        #if file is not exist
        try:
            knowledgeBase=pd.read_csv(self.KBPath)
        except:
            knowledgeBase=pd.DataFrame(columns=['numberofFeatres','logofFeatures','numberofInstances','logofInstances','numberofClasses','numberofNumericalFeatures','numberofCategoricalFeatures','ratio','entropy'])
        newrow=pd.DataFrame([[numberofFeatres,logofFeatures,numberofInstances,logofInstances,numberofClasses,
                              numberofNumericalFeatures,numberofCategoricalFeatures,ratio,entropy]]
                            ,columns=['numberofFeatres','logofFeatures','numberofInstances','logofInstances',
                                      'numberofClasses','numberofNumericalFeatures','numberofCategoricalFeatures','ratio','entropy'])
        knowledgeBase=pd.concat([knowledgeBase,newrow],ignore_index=True)
        #if the file exists
        try:
            knowledgeBase.to_csv(self.KBPath,index=False)
        except:
            #delete the file and create a new one
            os.remove(self.KBPath)
            knowledgeBase.to_csv(self.KBPath,index=False)