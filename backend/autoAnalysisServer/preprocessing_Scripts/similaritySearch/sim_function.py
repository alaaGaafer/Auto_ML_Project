import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class MetaFeatureSimilarity:
    def __init__(self, uploaded_meta_features, knowledge_base_path):
        self.uploaded_meta_features = uploaded_meta_features
        self.knowledge_base = pd.read_csv(knowledge_base_path)

    def prepare_knowledge_base_meta_features(self):
        numerical_columns = ['numberofFeatres', 'logofFeatures', 'numberofInstances', 'logofInstances', 'numberofClasses',
                             'numberofNumericalFeatures', 'numberofCategoricalFeatures', 'ratio', 'entropy', 'classprobmax',
                             'classprobmin', 'classprobmean', 'classprobstd', 'symbolsmean', 'symbolssum', 'symbolsstd',
                             'skewnessmin', 'skewnessmax', 'skewnessmean', 'skewnessstd', 'kurtosismin', 'kurtosismax',
                             'kurtosismean', 'kurtosisstd', 'DatasetRatioofNumberofFeaturestoNumberofInstances']

        meta_features = []
        for index, row in self.knowledge_base.iterrows():
            features = row[numerical_columns].values
            meta_features.append(features)
        return np.array(meta_features)

    def find_top_similar_datasets_knn(self, top_n=3):
        knowledge_base_meta_features = self.prepare_knowledge_base_meta_features()

        knn = NearestNeighbors(n_neighbors=top_n, metric='euclidean')
        knn.fit(knowledge_base_meta_features)

        distances, indices = knn.kneighbors([self.uploaded_meta_features])

        similar_datasets = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            row = self.knowledge_base.iloc[idx]
            dataset_info = {
                'Dataset_Name': row['Dataset_Name'],
                'Best_model': row['Best_model'],
                'Distance': distances[0][i]
            }
            similar_datasets.append(dataset_info)

        return similar_datasets

    def get_best_models(self):
        top_similar_datasets = self.find_top_similar_datasets_knn()
        best_models = [dataset['Best_model'] for dataset in top_similar_datasets]
        dataset_names = [dataset['Dataset_Name'] for dataset in top_similar_datasets]

        return best_models, dataset_names
