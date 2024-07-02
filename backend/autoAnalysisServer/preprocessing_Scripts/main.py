from sklearn.model_selection import train_test_split

from .similaritySearch.functions import *
from .similaritySearch.metaFeatureExtraction import *
from .similaritySearch.sim_function import *
from .bestmodel import *


class AutoClean:
    def __init__(self, historical_df, y_column, problem, date_col=None, pred_df=None):
        #self.historical_file_path = historical_file_path
        self.historical_df = historical_df
        self.pred_df = pred_df
        self.y_column = y_column
        self.date_col = date_col
        self.problem = problem.lower()
        # print("problem", self.problem)

    @staticmethod
    def _read_file(file_path: str) -> pd.DataFrame:
        """Read a CSV or Excel file into a DataFrame."""
        file_extension = file_path.split('.')[-1]
        if file_extension == 'csv':
            return pd.read_csv(file_path)
        elif file_extension in ['xls', 'xlsx']:
            return pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format")

    def extract_and_search_features(self, historical_df, dataset_name= "train.csv"):
        # print(self.problem)
        knowledge_base_path = {
            "regression": "similaritySearch/Knowledge bases/new_knowledgeBaseReg.csv",
            "classification": "preprocessing_Scripts/similaritySearch/Knowledge bases/new_knowledgeBaseCls.csv",
            "timeseries": "similaritySearch/Knowledge bases/knowledgeBaseTime.csv"
        }.get(self.problem)

        if not knowledge_base_path:
            raise ValueError(f"Unsupported problem type: {self.problem}")

        # dataset_name = self.historical_file_path.split('.')[0]
        meta_extractor = metafeatureExtraction(historical_df, dataset_name, knowledge_base_path, self.y_column)
        meta_features = meta_extractor.getMetaFeatures()
        meta_features_numerical = meta_features[1:]
        print("meta_features: ", meta_features)

        similarity_checker = MetaFeatureSimilarity(meta_features_numerical, knowledge_base_path)
        best_models, dataset_names = similarity_checker.get_best_models()
        best_models = list(set(best_models))
        print("best_models", best_models, "dataset_names", dataset_names)

        return meta_extractor, meta_features, best_models

    def Detections(self):
        # Check file extension
        # historical_df = self._read_file(self.historical_file_path)
        # Remove ID column and store categorical and numerical columns
        self.historical_df = RemoveIDColumn.remove_high_cardinality_columns(self.historical_df)
        self.historical_df = MissingValues().del_high_null_cols(self.historical_df)
        categorical_columns = self.historical_df.select_dtypes(include=['object']).columns.tolist()
        # numerical_columns = historical_df.select_dtypes(include=['number']).columns.tolist()

        if self.date_col:
            self.historical_df[self.date_col] = pd.to_datetime(self.historical_df[self.date_col])

        nulls_columns = MissingValues().detect_nulls(self.historical_df)
        cols_with_outliers = Outliers().detect_outliers(self.historical_df)
        historical_df = Duplicates().handle_dub(self.historical_df)
        imbalance_detected = HandlingImbalanceClasses().detect_class_imbalance(historical_df, self.y_column)
        df_without_y = historical_df.drop(columns=[self.y_column])
        low_variance_columns, low_variance_info = HandlingColinearity().detect_low_variance(df_without_y)

        return historical_df, nulls_columns, cols_with_outliers, imbalance_detected, low_variance_columns, categorical_columns

    @staticmethod
    def _process_data(df: pd.DataFrame, fill_na_dict: dict, outliers_methods_input: tuple,
                      Norm_method: str) -> pd.DataFrame:
        """Handle missing values, outliers, normalization, and encoding."""
        df = MissingValues().handle_nan(df, fill_na_dict)
        cols_with_outliers = Outliers().detect_outliers(df)
        df = Outliers().handle_outliers(df, cols_with_outliers, outliers_methods_input)
        df = DataNormalization().normalize_data(df, Norm_method)

        return df

    # historical_df is the same one returned from detections
    def Handling_calls(self, fill_na_dict, outliers_methods_input, imb_instruction, Norm_method,
                       lowvariance_actions, encoding_dict, reduce, auto_reduce, num_components_to_keep):
        if self.pred_df:
            # pred_df = self._read_file(self.pred_file_path)
            self.pred_df, carednality_messages = RemoveIDColumn.remove_high_cardinality_columns(self.pred_df)
            self.pred_df, deletion_messages = MissingValues().del_high_null_cols(self.pred_df)

            if self.problem == "timeseries":
                self.pred_df.rename(columns={self.date_col: 'ds', self.y_column: 'y'}, inplace=True)

            self.pred_df = self.pred_df.reset_index(drop=True)

            # Handle missing values, duplicates, outliers, normalization, and co-linearity in pred_data
            pred_fill_na_dict = {}
            null_info = df.isnull().sum()
            pred_nulls_columns = null_info[null_info > 0]
            for col in pred_nulls_columns:
                for H_col, method in fill_na_dict.items():
                    if col == H_col:
                        pred_fill_na_dict[col] = method
                    else:
                        pred_fill_na_dict[col] = "auto"

            pred_df = self._process_data(self.pred_df, pred_fill_na_dict, outliers_methods_input, Norm_method)

            pred_df_without_y = pred_df.drop(columns=[self.y_column])
            pred_df_without_y = HandlingColinearity().handle_low_variance(pred_df_without_y, lowvariance_actions)
            pred_df_without_y, handling_info = HandlingColinearity().handling_colinearity(pred_df_without_y)
            pred_df_without_y[self.y_column] = pred_df[self.y_column]
            pred_df = pred_df_without_y.copy()
            pred_df = EncodeCategorical().Encode(pred_df, encoding_dict)
            pred_df = HandlingImbalanceClasses().handle_class_imbalance(pred_df, self.y_column, imb_instruction)

        # ----------------------------------------------------------------------------------------------------------
        if self.problem == "timeseries":
            self.historical_df.rename(columns={self.date_col: 'ds', self.y_column: 'y'}, inplace=True)
        # لو عاصم هيجرب ويكنسل الdetections
        # historical_df = RemoveIDColumn.remove_high_cardinality_columns(historical_df)
        # historical_df = MissingValues().del_high_null_cols(historical_df)
        # historical_df = Duplicates().handle_dub(historical_df)
        historical_df = self.historical_df.reset_index(drop=True)

        train_data, test_data = train_test_split(historical_df, test_size=0.2, random_state=42,
                                                 stratify=historical_df[self.y_column])

        train_data = train_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)

        train_data = self._process_data(train_data, fill_na_dict, outliers_methods_input, Norm_method)
        test_data = self._process_data(test_data, fill_na_dict, outliers_methods_input, Norm_method)

        # ----------------------------------------------------------------------------------------------------------
        # concat test and train to remake historical_df to pass to extract_and_search_features
        historical_df_copy = pd.concat([train_data, test_data])
        original_columns = set(historical_df.columns)

        # Remove co-linearity & low_variance
        historical_df_without_y = historical_df_copy.drop(columns=[self.y_column])
        historical_df_without_y = HandlingColinearity().handle_low_variance(historical_df_without_y,
                                                                            lowvariance_actions)
        historical_df_without_y = HandlingColinearity().handling_colinearity(historical_df_without_y)
        historical_df_without_y[self.y_column] = historical_df_copy[self.y_column]

        # Update historical_df_copy with processed columns
        historical_df_copy = historical_df_without_y.copy()

        # Identify removed columns
        processed_columns = set(historical_df_copy.columns)
        removed_columns = list(original_columns - processed_columns)

        # Remove identified columns from train_data
        train_data = train_data.drop(columns=removed_columns)
        test_data = test_data.drop(columns=removed_columns)

        # Extract features and perform similarity search
        meta_extractor, meta_features, best_models = self.extract_and_search_features(historical_df_copy)
        # ----------------------------------------------------------------------------------------------------------
        # resuming cleaning both df (because of extraction)
        # Encode categorical features and reduce dimensions if needed
        train_data = EncodeCategorical().Encode(train_data, encoding_dict)
        test_data = EncodeCategorical().Encode(test_data, encoding_dict)
        if imb_instruction:
            train_data = HandlingImbalanceClasses().handle_class_imbalance(train_data, self.y_column, imb_instruction)
            test_data = HandlingImbalanceClasses().handle_class_imbalance(test_data, self.y_column, imb_instruction)

        # handling_reduction = HandlingReduction()

        # X_train = train_data.drop(columns=[self.y_column]).select_dtypes(include=[np.number])
        # explained_variability, _ = handling_reduction.explainedVariability(X_train)
        # num_components_to_keep = handling_reduction.NumberOfComponents(X_train, explained_variability)
        # pca_model = PCA(n_components=num_components_to_keep).fit(X_train)

        # # Apply the fitted PCA model to both train and test data
        # reduced_train_df = handling_reduction.feature_reduction(train_data.drop(columns=[self.y_column]),
        #                                                         num_components_to_keep, pca_model)
        # reduced_train_df[self.y_column] = train_data[self.y_column]

        # reduced_test_df = handling_reduction.feature_reduction(test_data.drop(columns=[self.y_column]), num_components_to_keep,
        #                                                        pca_model)
        # reduced_test_df[self.y_column] = test_data[self.y_column]

        # if self.pred_df:
        #     self.pred_df = handling_reduction.feature_reduction(self.pred_df.drop(columns=[self.y_column]),
        #                                                             num_components_to_keep, pca_model)
        #     self.pred_df[self.y_column] = self.pred_df[self.y_column]
        #     print("reduced_test_df", self.pred_df)

        # print("reduced_train_df",reduced_train_df)

        # print("reduced_test_df",reduced_test_df)

        # --------------------------------------------------------------------------------------------
        # return cleand df
        historical_df_copy = pd.concat([train_data, test_data])

        # Split train_data into features and labels
        x_train = train_data.drop(columns=[self.y_column])
        y_train = train_data[self.y_column]

        # Split test_data into features and labels
        x_test = test_data.drop(columns=[self.y_column])
        y_test = test_data[self.y_column]

        Bestmodelobj = Bestmodel(ProblemType.CLASSIFICATION, ["KNN","LR","RF"], x_train, x_test, y_train, y_test)
        Bestmodelobj.splitTestData()
        Bestmodelobj.TrainModel()
        print(Bestmodelobj.modelobj)
        # meta_features.append(Bestmodelobj.modelobj)
        # meta_extractor.addToKnowledgeBase(meta_features)
        if self.pred_df:
            return historical_df_copy, self.pred_df
        else:
            return historical_df_copy

    @staticmethod
    def featureReduction(df, Num_components_to_keep):
        numerical_columns = df.select_dtypes(include=['number']).columns
        numericalDf = df[numerical_columns]
        reducedFeatures = HandlingReduction().feature_reduction(numericalDf, Num_components_to_keep)
        non_numerical_columns = df.select_dtypes(exclude=['number']).columns
        non_numerical_df = df[non_numerical_columns].reset_index(drop=True)
        concatenated_features = pd.concat([non_numerical_df, reducedFeatures], axis=1)
        return concatenated_features


# this part is just for testing ( will be removed)


def user_interaction():
    # file_path = "similaritySearch/train.csv"
    y_column = 'Survived'
    df = pd.read_csv("similaritySearch/train.csv")
    # Step 2: Call the Detections function to get detection results
    try:
        autoclean = AutoClean(df, y_column, "classification")
        df_copy, nulls_columns, outlier_columns, imbalance, low_variance_columns, categorical_columns = autoclean.Detections()

        print("Detection Results:")
        print(f"Nulls columns: {nulls_columns}")
        print(f"Outlier columns: {outlier_columns}")
        print(f"Imbalance: {imbalance}")
        print(f"Low Variance Columns: {low_variance_columns}")
        print(f"Categorical Columns: {categorical_columns}")

        # Handling missing values
        fill_na_dict = {}
        for col in nulls_columns:
            fill_na_dict[col] = 'auto'

        # Handling outliers
        outliers_method_input = ('z_score', 'auto', 3)
        if imbalance:
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
        processed_data = autoclean.Handling_calls(fill_na_dict, outliers_method_input,
                                                  imb_instruction, Norm_method,
                                                  low_actions, encoding_dict, reduce, auto_reduce,
                                                  num_components_to_keep)
        # print(processed_data)
        print("Data preprocessing and handling completed successfully.")
        return processed_data

    except ValueError as ve:
        print(f"Error occurred: {ve}")
        return None


# Example usage:
if __name__ == "__main__":
    user_interaction()
