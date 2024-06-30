from .functions import *
from .metaFeatureExtraction import *
from .sim_function import *
from enum import Enum


class ProblemType(Enum):
    REGRESSION = "Regression"
    CLASSIFICATION = "Classification"
    TIME_SERIES = "Time series"


def split_dataset(file_path, y_column, pred_data_path=None):
    if pred_data_path:
        train_data = pd.read_csv(file_path)
        pred_data = pd.read_csv(pred_data_path)
    else:
        data = pd.read_csv(file_path)
        train_data = data.dropna(subset=[y_column])
        pred_data = data[data[y_column].isna()]

    return train_data, pred_data


def Detections(df, y_column, date_col=None):
    # Check file extension
    # file_extension = file_path.split('.')[-1]
    # if file_extension == 'csv':
    #     df = pd.read_csv(file_path)
    # elif file_extension in ['xls', 'xlsx']:
    #     df = pd.read_excel(file_path)
    # else:
    #     raise ValueError("Unsupported file format")


    df_copy = df.copy()

    # Remove ID column and store categorical and numerical columns
    # those two messages are lists
    df_copy, carednality_messages = RemoveIDColumn.remove_high_cardinality_columns(df_copy)
    df_copy, deletion_messages = MissingValues().del_high_null_cols(df_copy)
    categorical_columns = df_copy.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = df_copy.select_dtypes(include=['number']).columns.tolist()

    if date_col:
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])

    nulls_dict = MissingValues().detect_nulls(df_copy)
    outlier_info, cols_with_outliers = Outliers().detect_outliers(df_copy)
    duplicates = df.duplicated().any()
    imbalance_info, imbalance_detected = HandlingImbalanceClasses().detect_class_imbalance(df_copy, y_column)
    df_without_y = df_copy.drop(columns=[y_column])
    low_variance_columns, low_variance_info = HandlingColinearity().detect_low_variance(df_without_y)

    return df_copy, nulls_dict, outlier_info, duplicates, imbalance_info, numerical_columns, low_variance_columns, low_variance_info, categorical_columns, deletion_messages, carednality_messages


def extract_and_search_features(df_copy, file_path, problem, y_column):
    knowledge_base_path = {
        ProblemType.REGRESSION.value: "Knowledge bases\\new_knowledgeBaseReg.csv",
        ProblemType.CLASSIFICATION.value: "Knowledge bases\\new_knowledgeBaseCls.csv",
        ProblemType.TIME_SERIES.value: "Knowledge bases\\knowledgeBaseTime.csv"
    }.get(problem)

    dataset_name = file_path.split('.')[0]
    meta_extractor = metafeatureExtraction(df_copy, dataset_name, knowledge_base_path, y_column)
    meta_features = meta_extractor.getMetaFeatures()
    meta_features_numerical = meta_features[1:]
    print("meta_features: ", meta_features)

    similarity_checker = MetaFeatureSimilarity(meta_features_numerical, knowledge_base_path)
    best_models, dataset_names = similarity_checker.get_best_models()
    best_models = list(set(best_models))
    print("best_models", best_models, "dataset_names", dataset_names)
    # cash &best model
    # add best model to metafeatures
    # meta_extractor.addToKnowledgeBase(meta_features)

    return meta_features, best_models


class AutoClean:

    def Handling_calls(self, file_path, df_copy, problem, y_column, fill_na_dict, outliers_methods_input,
                       imb_instruction, Norm_method, lowvariance_actions,
                       encoding_dict, reduce, auto_reduce, num_components_to_keep, date_col=None):

        df_copy = df_copy.reset_index(drop=True)
        # making the y column the last col for the meta_extraction
        # y_column_content = df_copy[y_column]
        # df_copy.drop(y_column, axis=1, inplace=True)
        # df_copy[y_column] = y_column_content

        if problem == ProblemType.TIME_SERIES.value:
            df_copy.rename(columns={date_col: 'ds', y_column: 'y'}, inplace=True)

        # Handle missing values, duplicates, outliers, normalization, and colinearity
        df_copy = MissingValues().handle_nan(df_copy, fill_na_dict)
        df_copy = Duplicates().handle_dub(df_copy)
        outlier_info, cols_with_outliers = Outliers().detect_outliers(df_copy)
        df_copy = Outliers().handle_outliers(df_copy, cols_with_outliers, outliers_methods_input)
        df_copy = DataNormalization().normalize_data(df_copy, Norm_method)
        df_without_y = df_copy.drop(columns=[y_column])
        df_without_y = HandlingColinearity().handle_low_variance(df_without_y, lowvariance_actions)
        df_without_y, handling_info = HandlingColinearity().handling_colinearity(df_without_y)
        df_without_y[y_column] = df_copy[y_column]
        df_copy = df_without_y.copy()

        # Extract features and perform similarity search
        # pass train instead of df_copy
        meta_features, best_models = extract_and_search_features(df_copy, file_path, problem, y_column)

        # Encode categorical features and reduce dimensions if needed
        df_copy = EncodeCategorical().Encode(df_copy, encoding_dict)
        df_copy = HandlingImbalanceClasses().handle_class_imbalance(df_copy, y_column, imb_instruction)

        if reduce == 'True':
            explainedVariability, explanation_list = HandlingReduction().explainedVariability(
                df_copy.select_dtypes(include=['number']))
            if auto_reduce == 'True':
                num_components_to_keep = HandlingReduction().NumberOfComponents(
                    df_copy.select_dtypes(include=['number']), explainedVariability)
            df_copy = self.featureReduction(df_copy, num_components_to_keep)
        # train the best model on data
        # return model name & predictions & hyperparameters

        return df_copy

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
    file_path = "train.csv"
    y_column = 'Survived'

    # Step 2: Call the Detections function to get detection results
    try:
        df_copy, nulls_dict, outlier_info, duplicates, imbalance_info, numerical_columns, low_variance_columns, \
        low_variance_info, categorical_columns, deletion_messages, carednality_messages = Detections(file_path,
                                                                                                     y_column)

        print("Detection Results:")
        print(f"Nulls Dictionary: {nulls_dict}")
        print(f"Outlier Info: {outlier_info}")
        print(f"Duplicates Detected: {duplicates}")
        print(f"Imbalance Info: {imbalance_info}")
        print(f"Numerical Columns: {numerical_columns}")
        print(f"Low Variance Columns: {low_variance_columns}")
        print(f"Categorical Columns: {categorical_columns}")
        print(f"deletion_messages: {deletion_messages}")
        print(f"deletion_messages: {carednality_messages}")

        problem_type = ProblemType.CLASSIFICATION.value

        # Handling missing values
        fill_na_dict = {}
        for col in nulls_dict.keys():
            fill_na_dict[col] = 'auto'

        # Handling outliers
        outliers_method_input = ('z_score', 'auto', 3)

        imb_instruction = "auto"
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

        auto_clean_instance = AutoClean()
        processed_data = auto_clean_instance.Handling_calls(file_path, df_copy, problem_type, y_column, fill_na_dict,
                                                            outliers_method_input, imb_instruction, Norm_method,
                                                            low_actions, encoding_dict, reduce, auto_reduce,
                                                            num_components_to_keep)

        print("Data preprocessing and handling completed successfully.")
        return processed_data

    except ValueError as ve:
        print(f"Error occurred: {ve}")
        return None


# Example usage:
if __name__ == "__main__":
    user_interaction()
