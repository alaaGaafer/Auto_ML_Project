from sklearn.model_selection import train_test_split

from similaritySearch.functions import *
from similaritySearch.metaFeatureExtraction import *
from similaritySearch.sim_function import *
from bestmodel import *


class AutoClean:
    def __init__(self, historical_file_path, y_column, problem, date_col=None, pred_file_path=None):
        self.historical_file_path = historical_file_path
        self.pred_file_path = pred_file_path
        self.y_column = y_column
        self.date_col = date_col
        self.problem = problem.lower()

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

    def extract_and_search_features(self, historical_df):
        knowledge_base_path = {
            "regression": "similaritySearch/Knowledge bases/new_knowledgeBaseReg.csv",
            "classification": "similaritySearch/Knowledge bases/new_knowledgeBaseCls.csv",
            "timeseries": "similaritySearch/Knowledge bases/knowledgeBaseTime.csv"
        }.get(self.problem)

        if not knowledge_base_path:
            raise ValueError(f"Unsupported problem type: {self.problem}")

        dataset_name = self.historical_file_path.split('.')[0]
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
        historical_df = self._read_file(self.historical_file_path)

        # Remove ID column and store categorical and numerical columns
        historical_df = RemoveIDColumn.remove_high_cardinality_columns(historical_df)
        historical_df = MissingValues().del_high_null_cols(historical_df)
        categorical_columns = historical_df.select_dtypes(include=['object']).columns.tolist()
        # numerical_columns = historical_df.select_dtypes(include=['number']).columns.tolist()

        if self.date_col:
            historical_df[self.date_col] = pd.to_datetime(historical_df[self.date_col])

        nulls_columns = MissingValues().detect_nulls(historical_df)
        cols_with_outliers = Outliers().detect_outliers(historical_df)
        historical_df = Duplicates().handle_dub(historical_df)
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

    def apply_feature_reduction(self,data, y_column, reduce, auto_reduce, feature_reduction_method, handling_reduction):
        if reduce == 'True':
            # Extract features only (excluding target column)
            data_features = data.drop(columns=[y_column])

            # Apply feature reduction on the features
            explained_variability, _ = handling_reduction.explainedVariability(
                data_features.select_dtypes(include=['number'])
            )

            num_components_to_keep = None
            if auto_reduce == 'True':
                num_components_to_keep = handling_reduction.NumberOfComponents(
                    data_features.select_dtypes(include=['number']), explained_variability
                )

            # Apply feature reduction
            reduced_data_features = feature_reduction_method(data_features, num_components_to_keep)

            # Reattach the target column
            reduced_data = reduced_data_features.copy()
            reduced_data[y_column] = data[y_column]
        else:
            reduced_data = data.copy()

        return reduced_data

    # historical_df is the same one returned from detections
    def Handling_calls(self, historical_df, fill_na_dict, outliers_methods_input, imb_instruction, Norm_method,
                       lowvariance_actions, encoding_dict, reduce, auto_reduce, num_components_to_keep):
        if self.pred_file_path:
            pred_df = self._read_file(self.pred_file_path)
            pred_df, carednality_messages = RemoveIDColumn.remove_high_cardinality_columns(pred_df)
            pred_df, deletion_messages = MissingValues().del_high_null_cols(pred_df)

            if self.problem == "timeseries":
                pred_df.rename(columns={self.date_col: 'ds', self.y_column: 'y'}, inplace=True)

            pred_df = pred_df.reset_index(drop=True)

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

            pred_df = self._process_data(pred_df, pred_fill_na_dict, outliers_methods_input, Norm_method)

            pred_df_without_y = pred_df.drop(columns=[self.y_column])
            pred_df_without_y = HandlingColinearity().handle_low_variance(pred_df_without_y, lowvariance_actions)
            pred_df_without_y, handling_info = HandlingColinearity().handling_colinearity(pred_df_without_y)
            pred_df_without_y[self.y_column] = pred_df[self.y_column]
            pred_df = pred_df_without_y.copy()
            pred_df = EncodeCategorical().Encode(pred_df, encoding_dict)
            pred_df = HandlingImbalanceClasses().handle_class_imbalance(pred_df, self.y_column, imb_instruction)

            # pred_df = self.apply_feature_reduction(
            #     pred_df, self.y_column, reduce, auto_reduce, self.featureReduction, HandlingReduction()
            # )
        # ----------------------------------------------------------------------------------------------------------
        if self.problem == "timeseries":
            historical_df.rename(columns={self.date_col: 'ds', self.y_column: 'y'}, inplace=True)
        historical_df = historical_df.reset_index(drop=True)

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

        # train_data = self.apply_feature_reduction(
        #     train_data, self.y_column, reduce, auto_reduce, self.featureReduction, HandlingReduction()
        # )
        #
        # test_data = self.apply_feature_reduction(
        #     test_data, self.y_column, reduce, auto_reduce, self.featureReduction, HandlingReduction()
        # )

        # --------------------------------------------------------------------------------------------
        # return cleand df
        historical_df_copy = pd.concat([train_data, test_data])

        # Split train_data into features and labels
        x_train = train_data.drop(columns=[self.y_column])
        y_train = train_data[self.y_column]

        # Split test_data into features and labels
        x_test = test_data.drop(columns=[self.y_column])
        y_test = test_data[self.y_column]

        Bestmodelobj = Bestmodel(ProblemType.CLASSIFICATION, ["KNN",'LR','RF'], x_train, x_test, y_train, y_test)
        Bestmodelobj.splitTestData()
        Bestmodelobj.TrainModel()
        print(Bestmodelobj.modelobj)
        # meta_features.append(Bestmodelobj.modelobj)
        # meta_extractor.addToKnowledgeBase(meta_features)
        if self.pred_file_path:
            return historical_df_copy, pred_df
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
    file_path = "similaritySearch/train.csv"
    y_column = 'Survived'

    # Step 2: Call the Detections function to get detection results
    try:
        autoclean = AutoClean(file_path, y_column, "classification")
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

        processed_data = autoclean.Handling_calls(df_copy, fill_na_dict, outliers_method_input,
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
