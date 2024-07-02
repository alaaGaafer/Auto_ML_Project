import os
import pandas as pd
import numpy as np


class metafeatureExtraction:
    def __init__(self, df, dataset_name, KBPath, Target):
        self.df = df
        # self.df.columns = list(self.df.columns[:-1]) + ['Target']
        self.Target = Target
        self.KBPath = KBPath
        self.dataset_name = dataset_name
        self.numeric_df = self.df.select_dtypes(include=[np.number])
        self.categorical_df = self.df.select_dtypes(exclude=[np.number]).columns

    def getNumberOfFeaturesandTheirlog(self):
        numberofFeatres = len(self.df.columns)
        logofFeatures = np.log(numberofFeatres)
        return numberofFeatres, logofFeatures

    def getNumberOfInstancesandTheirlog(self):
        numberofInstances = len(self.df)
        logofInstances = np.log(numberofInstances)
        return numberofInstances, logofInstances

    def getnumberofClasses(self):
        numberofClasses = len(self.df[self.Target].unique())
        return numberofClasses

    def getnumberofNumericalFeatures(self):
        numberofNumericalFeatures = len(self.numeric_df)
        return numberofNumericalFeatures

    def getnumberofCategoricalFeatures(self):
        numberofCategoricalFeatures = len(self.categorical_df)
        return numberofCategoricalFeatures

    def ratioofNumericaltoCategoricalFeatures(self):
        numberofNumericalFeatures = len(self.numeric_df)
        numberofCategoricalFeatures = len(self.categorical_df)
        ratio = numberofNumericalFeatures / numberofCategoricalFeatures if numberofCategoricalFeatures != 0 else 0
        return ratio

    def classeEntropy(self):
        entropy = -np.sum(
            self.df[self.Target].value_counts(normalize=True) * np.log2(self.df[self.Target].value_counts(normalize=True)))
        return entropy

    def classProbabilityMax(self):
        return self.df[self.Target].value_counts(normalize=True).max()

    def classProbabilityMin(self):
        return self.df[self.Target].value_counts(normalize=True).min()

    def classProbabilityMean(self):
        return self.df[self.Target].value_counts(normalize=True).mean()

    def classProbabilityStd(self):
        return self.df[self.Target].value_counts(normalize=True).std()

    def symbolsMean(self):
        return self.df.describe().loc['mean'].mean()

    def symbolsSum(self):
        return self.numeric_df.sum().sum()

    def symbolsStd(self):
        return self.df.describe().loc['std'].std()

    def skewnessMin(self):
        return self.numeric_df.skew().min()

    def skewnessMax(self):
        return self.numeric_df.skew().max()

    def skewnessMean(self):
        return self.numeric_df.skew().mean()

    def skewnessStd(self):
        return self.numeric_df.skew().std()

    def kurtosisMin(self):
        return self.numeric_df.kurtosis().min()

    def kurtosisMax(self):
        return self.numeric_df.kurtosis().max()

    def kurtosisMean(self):
        return self.numeric_df.kurtosis().mean()

    def kurtosisStd(self):
        return self.numeric_df.kurtosis().std()

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
            skewnessstd, kurtosismin, kurtosismax, kurtosismean, kurtosisstd,
            DatasetRatioofNumberofFeaturestoNumberofInstances
        ]

    def addToKnowledgeBase(self, meta_features):
        try:
            knowledgeBase = pd.read_csv(self.KBPath, encoding='latin1')
        except (FileNotFoundError, pd.errors.EmptyDataError):
            knowledgeBase = pd.DataFrame(columns=[
                'Dataset_Name', 'numberofFeatres', 'logofFeatures', 'numberofInstances', 'logofInstances',
                'numberofClasses',
                'numberofNumericalFeatures', 'numberofCategoricalFeatures', 'ratio', 'entropy', 'classprobmax',
                'classprobmin', 'classprobmean', 'classprobstd', 'symbolsmean', 'symbolssum', 'symbolsstd',
                'skewnessmin', 'skewnessmax', 'skewnessmean', 'skewnessstd', 'kurtosismin', 'kurtosismax',
                'kurtosismean', 'kurtosisstd', 'DatasetRatioofNumberofFeaturestoNumberofInstances'
            ])
        newrow = pd.DataFrame([meta_features], columns=knowledgeBase.columns)
        knowledgeBase = pd.concat([knowledgeBase, newrow], ignore_index=True)
        knowledgeBase.to_csv(self.KBPath, index=False, encoding='latin1')


# This was for extracting the meta feature of our DB , won't be used again
def process_folder(Folder_path, KBPath):
    for filename in os.listdir(Folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(Folder_path, filename)
            df = pd.read_csv(file_path, encoding='latin1')
            dataset_name = os.path.splitext(filename)[0]
            meta_extractor = metafeatureExtraction(df, dataset_name, KBPath)
            meta_features = meta_extractor.getMetaFeatures()
            meta_extractor.addToKnowledgeBase(meta_features)
