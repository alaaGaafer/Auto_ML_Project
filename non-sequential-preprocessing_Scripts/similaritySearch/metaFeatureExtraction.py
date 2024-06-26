import os
import pandas as pd
import numpy as np


class metafeatureExtraction:
    def __init__(self, df, dataset_name):
        self.df = df
        self.df.columns = list(self.df.columns[:-1]) + ['Target']
        self.KBPath = 'knowledgeBaseReg.csv'
        self.dataset_name = dataset_name

    def getNumberOfFeaturesandTheirlog(self):
        numberofFeatres = len(self.df.columns)
        logofFeatures = np.log(numberofFeatres)
        return numberofFeatres, logofFeatures

    def getNumberOfInstancesandTheirlog(self):
        numberofInstances = len(self.df)
        logofInstances = np.log(numberofInstances)
        return numberofInstances, logofInstances

    def getnumberofClasses(self):
        numberofClasses = len(self.df['Target'].unique())
        return numberofClasses

    def getnumberofNumericalFeatures(self):
        numberofNumericalFeatures = len(self.df.select_dtypes(include=[np.number]).columns)
        return numberofNumericalFeatures

    def getnumberofCategoricalFeatures(self):
        numberofCategoricalFeatures = len(self.df.select_dtypes(exclude=[np.number]).columns)
        return numberofCategoricalFeatures

    def ratioofNumericaltoCategoricalFeatures(self):
        numberofNumericalFeatures = len(self.df.select_dtypes(include=[np.number]).columns)
        numberofCategoricalFeatures = len(self.df.select_dtypes(exclude=['number']).columns)
        ratio = numberofNumericalFeatures / numberofCategoricalFeatures if numberofCategoricalFeatures != 0 else 0
        return ratio

    def classeEntropy(self):
        entropy = -np.sum(
            self.df['Target'].value_counts(normalize=True) * np.log2(self.df['Target'].value_counts(normalize=True)))
        return entropy

    def classProbabilityMax(self):
        return self.df['Target'].value_counts(normalize=True).max()

    def classProbabilityMin(self):
        return self.df['Target'].value_counts(normalize=True).min()

    def classProbabilityMean(self):
        return self.df['Target'].value_counts(normalize=True).mean()

    def classProbabilityStd(self):
        return self.df['Target'].value_counts(normalize=True).std()

    def symbolsMean(self):
        return self.df.describe().loc['mean'].mean()

    def symbolsSum(self):
        return self.df.drop(columns='Target').sum().sum()

    def symbolsStd(self):
        return self.df.describe().loc['std'].std()

    def skewnessMin(self):
        return self.df.skew().min()

    def skewnessMax(self):
        return self.df.skew().max()

    def skewnessMean(self):
        return self.df.skew().mean()

    def skewnessStd(self):
        return self.df.skew().std()

    def kurtosisMin(self):
        return self.df.kurtosis().min()

    def kurtosisMax(self):
        return self.df.kurtosis().max()

    def kurtosisMean(self):
        return self.df.kurtosis().mean()

    def kurtosisStd(self):
        return self.df.kurtosis().std()

    def DatasetRatioofNumberofFeaturestoNumberofInstances(self):
        return len(self.df.columns) / len(self.df)

    def getMetaFeatures(self):
        numberofFeatres, logofFeatures = self.getNumberOfFeaturesandTheirlog()
        numberofInstances, logofInstances = self.getNumberOfInstancesandTheirlog()
        numberofClasses = self.getnumberofClasses()
        numberofNumericalFeatures = self.getnumberofNumericalFeatures()
        numberofCategoricalFeatures = self.getnumberofCategoricalFeatures()
        ratio = self.ratioofNumericaltoCategoricalFeatures()
        entropy = self.classeEntropy()
        classprobmax = self.classProbabilityMax()
        classprobmin = self.classProbabilityMin()
        classprobmean = self.classProbabilityMean()
        classprobstd = self.classProbabilityStd()
        symbolsmean = self.symbolsMean()
        symbolssum = self.symbolsSum()
        symbolsstd = self.symbolsStd()
        skewnessmin = self.skewnessMin()
        skewnessmax = self.skewnessMax()
        skewnessmean = self.skewnessMean()
        skewnessstd = self.skewnessStd()
        kurtosismin = self.kurtosisMin()
        kurtosismax = self.kurtosisMax()
        kurtosismean = self.kurtosisMean()
        kurtosisstd = self.kurtosisStd()
        DatasetRatioofNumberofFeaturestoNumberofInstances = self.DatasetRatioofNumberofFeaturestoNumberofInstances()
        return [
            self.dataset_name, numberofFeatres, logofFeatures, numberofInstances, logofInstances, numberofClasses,
            numberofNumericalFeatures, numberofCategoricalFeatures, ratio, entropy, classprobmax, classprobmin,
            classprobmean, classprobstd, symbolsmean, symbolssum, symbolsstd, skewnessmin, skewnessmax, skewnessmean,
            skewnessstd, kurtosismin, kurtosismax, kurtosismean, kurtosisstd, DatasetRatioofNumberofFeaturestoNumberofInstances
        ]

    def addToKnowledgeBase(self):
        meta_features = self.getMetaFeatures()
        try:
            knowledgeBase = pd.read_csv(self.KBPath, encoding='latin1')
        except (FileNotFoundError, pd.errors.EmptyDataError):
            knowledgeBase = pd.DataFrame(columns=[
                'Dataset_Name', 'numberofFeatres', 'logofFeatures', 'numberofInstances', 'logofInstances', 'numberofClasses',
                'numberofNumericalFeatures', 'numberofCategoricalFeatures', 'ratio', 'entropy', 'classprobmax',
                'classprobmin', 'classprobmean', 'classprobstd', 'symbolsmean', 'symbolssum', 'symbolsstd',
                'skewnessmin', 'skewnessmax', 'skewnessmean', 'skewnessstd', 'kurtosismin', 'kurtosismax',
                'kurtosismean', 'kurtosisstd', 'DatasetRatioofNumberofFeaturestoNumberofInstances'
            ])
        newrow = pd.DataFrame([meta_features], columns=knowledgeBase.columns)
        knowledgeBase = pd.concat([knowledgeBase, newrow], ignore_index=True)
        knowledgeBase.to_csv(self.KBPath, index=False, encoding='latin1')


def process_folder(Folder_path):
    for filename in os.listdir(Folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(Folder_path, filename)
            df = pd.read_csv(file_path, encoding='latin1')
            dataset_name = os.path.splitext(filename)[0]
            meta_extractor = metafeatureExtraction(df, dataset_name)
            meta_extractor.addToKnowledgeBase()


folder_path = r"C:\Users\Alaa\Desktop\Auto_ML_Project\non-sequential-preprocessing_Scripts\similaritySearch\Regression data\Regression"
process_folder(folder_path)
