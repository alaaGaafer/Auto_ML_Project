from functions import *


def Detections(File_path, Problem, Y_column):
    # Check file extension
    file_extension = File_path.split('.')[-1]
    if file_extension == 'csv':
        df = pd.read_csv(File_path)
    elif file_extension in ['xls', 'xlsx']:
        df = pd.read_excel(File_path)
    else:
        raise ValueError("Unsupported file format")
    df_copy = df.copy()
    if Problem != "Clustering":
        df_copy = RemoveIDColumn.remove_id_column(df_copy)

    # Store categorical and numerical columns
    categorical_columns = df_copy.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = df_copy.select_dtypes(include=['number']).columns.tolist()

    df_copy = ConvertToDatetime().convert(df_copy)
    nulls_dict = MissingValues().detect_nulls(df_copy)
    outlier_info = Outliers().detect_outliers(df_copy)
    duplicates = df.duplicated().any()
    df_without_y = df_copy.drop(columns=[Y_column])
    low_variance_columns, low_variance_info = HandlingColinearity().detect_low_variance(df_without_y)

    return nulls_dict, outlier_info, duplicates, numerical_columns, low_variance_columns, low_variance_info, categorical_columns


class AutoClean:

    def Handling_calls(self, File_path, Problem, Y_column, fillNA_dict, methods_input, method, actions, encoding_dict,
                       reduce, auto_reduce, num_components_to_keep):
        # Read CSV file
        file_extension = File_path.split('.')[-1]
        if file_extension == 'csv':
            df = pd.read_csv(File_path)
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(File_path)
        else:
            raise ValueError("Unsupported file format")

        df = df.reset_index(drop=True)
        df_copy = df.copy()

        # removing Id column
        if Problem != "Clustering":
            df_copy = RemoveIDColumn.remove_id_column(df_copy)

        # next line should be called from the front-end after entering the user choices
        df_copy = MissingValues().handle_nan(df_copy, fillNA_dict)
        df_copy = Duplicates().handle_dub(df_copy)
        df_copy = Outliers().handle_outliers(df_copy, methods_input)
        df_copy = DataNormalization().normalize_data(df_copy, method)
        df_without_y = df_copy.drop(columns=[Y_column])
        df_without_y = HandlingColinearity().handle_low_variance(df_without_y, actions)
        # Detect and handle co-linearity
        df_without_y, handling_info = HandlingColinearity().handling_colinearity(df_without_y)
        df_without_y[Y_column] = df_copy[Y_column]
        df_copy = df_without_y.copy()
        df_copy = EncodeCategorical().Encode(df_copy, encoding_dict)
        # Reduce dimensions
        numerical_columns = df_copy.select_dtypes(include=['number']).columns
        numericalDf = df_copy[numerical_columns]

        if reduce == 'True':
            # next line better be called in front end and pass explainedVariability in order to display explanation_list
            explainedVariability, explanation_list = HandlingReduction().explainedVariability(numericalDf)
            if auto_reduce == 'True':
                num_components_to_keep = HandlingReduction().NumberOfComponents(numericalDf, explainedVariability)
            df_copy = self.featureReduction(df_copy, num_components_to_keep)
        return df_copy

    @staticmethod
    def featureReduction(df, num_components_to_keep):
        numerical_columns = df.select_dtypes(include=['number']).columns
        numericalDf = df[numerical_columns]
        reducedFeatures = HandlingReduction().feature_reduction(numericalDf, num_components_to_keep)
        non_numerical_columns = df.select_dtypes(exclude=['number']).columns
        non_numerical_df = df[non_numerical_columns]
        non_numerical_df.reset_index(drop=True, inplace=True)
        concatenated_features = pd.concat([non_numerical_df, reducedFeatures], axis=1)
        return concatenated_features

# if __name__ == "__main__":
#     # file_path = "data.csv"
#     file_path = "train.csv"
#
#     problem = "Clustering"
#     y_column = "Survived"
#     Detections(file_path, y_column)
#     # AutoClean.Handling_calls(file_path, problem, y_column)
