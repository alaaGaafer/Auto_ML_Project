import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.neighbors import NearestNeighbors

from metaFeatureExtraction import metafeatureExtraction


class MetaFeatureSimilarity:
    def __init__(self, Uploaded_dataset_path, Knowledge_base_path):
        self.uploaded_df = pd.read_csv(Uploaded_dataset_path)
        self.knowledge_base = pd.read_csv(Knowledge_base_path)
        self.meta_extractor = metafeatureExtraction(self.uploaded_df)

    def extract_uploaded_meta_features(self):
        return self.meta_extractor.getMetaFeatures()

    @staticmethod
    def calculate_similarity(meta1, meta2):
        return euclidean(meta1, meta2)

    def find_top_similar_datasets_euclidean(self, top_n=3):
        uploaded_meta_features = self.extract_uploaded_meta_features()
        similarities = []

        for index, row in self.knowledge_base.iterrows():
            knowledge_meta_features = row[
                ['numberofFeatres', 'logofFeatures', 'numberofInstances', 'logofInstances', 'numberofClasses',
                 'numberofNumericalFeatures', 'numberofCategoricalFeatures', 'ratio', 'entropy', 'classprobmax',
                 'classprobmin', 'classprobmean', 'classprobstd', 'symbolsmean', 'symbolssum', 'symbolsstd',
                 'skewnessmin', 'skewnessmax', 'skewnessmean', 'skewnessstd', 'kurtosismin', 'kurtosismax',
                 'kurtosismean', 'kurtosisstd', 'DatasetRatioofNumberofFeaturestoNumberofInstances']
            ].values
            similarity = self.calculate_similarity(uploaded_meta_features, knowledge_meta_features)
            dataset_info = {
                'DataSet_Name': row['DataSet_Name'],
                'Best_model': row['Best_model'],
                'Similarity': similarity  # Optionally, include similarity for clarity
            }
            similarities.append(dataset_info)

        similarities.sort(key=lambda x: x['Similarity'])
        return similarities[:top_n]

    def prepare_knowledge_base_meta_features(self):
        meta_features = []
        for index, row in self.knowledge_base.iterrows():
            features = [
                row['numberofFeatres'], row['logofFeatures'], row['numberofInstances'], row['logofInstances'],
                row['numberofClasses'], row['numberofNumericalFeatures'], row['numberofCategoricalFeatures'],
                row['ratio'], row['entropy'], row['classprobmax'], row['classprobmin'], row['classprobmean'],
                row['classprobstd'], row['symbolsmean'], row['symbolssum'], row['symbolsstd'], row['skewnessmin'],
                row['skewnessmax'], row['skewnessmean'], row['skewnessstd'], row['kurtosismin'], row['kurtosismax'],
                row['kurtosismean'], row['kurtosisstd'], row['DatasetRatioofNumberofFeaturestoNumberofInstances']
            ]
            meta_features.append(features)
        return np.array(meta_features)

    def find_top_similar_datasets_knn(self, top_n=3):
        uploaded_meta_features = np.array(self.extract_uploaded_meta_features()).reshape(1, -1)
        knowledge_base_meta_features = self.prepare_knowledge_base_meta_features()

        knn = NearestNeighbors(n_neighbors=top_n, metric='euclidean')
        knn.fit(knowledge_base_meta_features)

        distances, indices = knn.kneighbors(uploaded_meta_features)

        similar_datasets = []
        for idx in indices[0]:
            row = self.knowledge_base.iloc[idx]
            dataset_info = {
                'DataSet_Name': row['DataSet_Name'],
                'Best_model': row['Best_model'],
                'Distance': distances[0][idx]  # Optionally, include distance for clarity
            }
            similar_datasets.append(dataset_info)

        return similar_datasets

    def get_best_models(self, top_n=3, method='knn'):
        if method == 'knn':
            top_similar_datasets = self.find_top_similar_datasets_knn(top_n)
        else:
            top_similar_datasets = self.find_top_similar_datasets_euclidean(top_n)

        best_models = [{
            'DataSet_Name': dataset['DataSet_Name'],
            'Best_model': dataset['Best_model']
        } for dataset in top_similar_datasets]

        return best_models


uploaded_dataset_path = 'uploaded_dataset.csv'
knowledge_base_path = 'knowledgeBase.csv'

similarity_checker = MetaFeatureSimilarity(uploaded_dataset_path, knowledge_base_path)

top_3_models_knn = similarity_checker.get_best_models(method='knn')
print("Top 3 models using k-NN:", top_3_models_knn)

# Find top 3 models using Euclidean distance
top_3_models_euclidean = similarity_checker.get_best_models(method='euclidean')
print("Top 3 models using Euclidean distance:", top_3_models_euclidean)
