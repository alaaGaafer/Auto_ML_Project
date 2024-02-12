from functions import *


class AutoClean:

    @staticmethod
    def clean_data(File_path, Problem, Y_column):
        # Read CSV file
        df = pd.read_csv(File_path)
        print(df)
        df = df.reset_index(drop=True)
        df_copy = df.copy()

        # Convert to datetime
        df_copy = AutoClean.convert_to_datetime(df_copy)

        # removing Id column
        if Problem != "Clustering":
            df_copy = RemoveIDColumn.remove_id_column(df_copy)

        # Handle missing values , next line will be a notification in front-end
        nulls_dict = MissingValues().detect_nulls(df_copy)

        # next line should be called from the front-end after entering the user choices
        df_copy = AutoClean.handle_missing_values(df_copy, nulls_dict)
        print(df_copy.head(20))

        # Handle duplicates, notify the user
        df_copy = AutoClean.handle_duplicates(df_copy)

        # Detect outliers and inform the user
        outlier_info = Outliers().detect_outliers(df_copy)
        # next line should be called from the front-end after entering the user choices
        df_copy = AutoClean.handle_outliers(df_copy, outlier_info)

        # Normalize numerical columns
        method = 'auto'
        df_copy = DataNormalization().normalize_data(df_copy, method)

        # Handle low variance columns, detect it and inform the user
        df_without_y = df_copy.drop(columns=[Y_column])
        low_variance_columns, low_variance_info = HandlingColinearity().detect_low_variance(df_without_y)
        actions = {c: "auto" for c in low_variance_columns}
        # next line should be called from the front-end after entering the user choices
        df_without_y = HandlingColinearity().handle_low_variance(df_without_y, actions)
        # Detect and handle co-linearity
        df_without_y, handling_info = HandlingColinearity().handling_colinearity(df_without_y)
        df_without_y[Y_column] = df_copy[Y_column]
        df_copy = df_without_y.copy()
        print("df_copy",df_copy)

        # Encode categorical variables
        encoding_dict = EncodeCategorical().categorical_columns(df_copy)
        df_copy = AutoClean.encode_categorical(df_copy, encoding_dict)
        # Reduce dimensions
        df_copy = AutoClean.reduceDimensions(df_copy)

        print("Final data after cleaning: ")
        print(df_copy)
        print(df_copy.columns)

    @staticmethod
    def convert_to_datetime(df):
        print("info before", df.info())
        df = ConvertToDatetime().convert(df)
        print("info after", df.info())
        return df

    # Taking input from user , will be taken from the front end later
    @staticmethod
    def get_fill_method_input(df, col_name):
        col_type = df.dtypes[col_name]

        if pd.api.types.is_numeric_dtype(col_type):  # Check if the column is numeric (int or float)
            fill_method = input(
                f"Enter method for handling NaN in '{col_name}' (mean, median, mode, delete, auto): ").lower()
            valid_methods = ["mean", "median", "mode", "delete", "auto"]
        elif pd.api.types.is_object_dtype(col_type):  # Check if the column is an object
            fill_method = input(f"Enter method for handling NaN in '{col_name}' (mode, delete,auto): ").lower()
            valid_methods = ["mode", "delete", "auto"]
        else:
            fill_method = "auto"  # For date columns, assign it as "auto"
            valid_methods = ["auto"]

        while fill_method not in valid_methods:
            print(f"Invalid input. Please enter one of: {', '.join(valid_methods)}")
            fill_method = input(f"Enter method for handling NaN in '{col_name}': ").lower()

        return fill_method

    @staticmethod
    def handle_missing_values(df, nulls_dict):
        print('nan before: ', df.isna().sum())
        fillNA_dict = {}
        for col in nulls_dict.keys():
            fill_method = AutoClean.get_fill_method_input(df, col)
            fillNA_dict[col] = fill_method

        print("fillNA_dict", fillNA_dict)

        df = MissingValues().handle_nan(df, fillNA_dict)
        print('nan after: ', df.isna().sum())
        return df

    @staticmethod
    def handle_duplicates(df):
        print('dub before: ', df.duplicated().sum())
        df = Duplicates().handle_dub(df)
        print('dub after: ', df.duplicated().sum())
        return df

    # Taking input from user , will be taken from the front-end later
    @staticmethod
    def get_outliers_handling_input(outlier_info):
        methods_input = {}
        for col in outlier_info.keys():
            print(f"Handling options for '{col}':")
            print("1. Method: z_score, Handle: auto, Threshold: 3")
            print("2. Method: IQR, Handle: delete, Threshold: 1.5")
            user_choice = input("Enter your choice (1 or 2): ")

            if user_choice == '1':
                methods_input[col] = ('z_score', 'auto', 3)
            elif user_choice == '2':
                methods_input[col] = ('IQR', 'delete', 1.5)
            else:
                print("Invalid choice. Using default handling (Method: z_score, Handle: auto, Threshold: 3).")
                methods_input[col] = ('z_score', 'auto', 3)
        return methods_input

    @staticmethod
    def handle_outliers(df, outlier_info):
        methods_input = AutoClean.get_outliers_handling_input(outlier_info)
        df = Outliers().handle_outliers(df, methods_input)
        return df

    @staticmethod
    def encode_categorical(df, encoding_dict):
        return EncodeCategorical().Encode(df, encoding_dict)
    @staticmethod
###### should be modified to handle only numerical dataFrame############################
    def reduceDimensions(df):
        #get the number of dataframes numerical columns
        numerical_columns = df.select_dtypes(include=['number']).columns
        numericalDf=df[numerical_columns]
        #####
        print("")
        print("////////////////////////Features Reduction////////////////////////")
        
        print(f"The number of numerical columns is: {len(numerical_columns)}")
        print("Do you want to reduce the number of dimensions? (yes/no)")
        choice = input().lower()
        if choice == "yes" or choice == "y":
            explainedVariability, explanation_list = HandlingReduction().explainedVariability(numericalDf)
            for explanation in explanation_list:
                print(explanation)
            print("Do you want to handle the reduction automatically? (yes/no)")
            choice = input().lower()
            if choice == "yes" or choice == "y":
                num_components_to_keep = HandlingReduction().NumberOfComponents(numericalDf, explainedVariability)
                reducedFeatures = HandlingReduction().feature_reduction(numericalDf, num_components_to_keep)
            else:
                choice = int(input("Enter the number of components you want to keep: "))
                reducedFeatures = HandlingReduction().feature_reduction(numericalDf, choice)
        else:
            print("The number of dimensions will not be reduced")
            return df
        # print('the reduced features are: ',reducedFeatures)
        non_numerical_columns = df.select_dtypes(exclude=['number']).columns
        non_numerical_df = df[non_numerical_columns]
        non_numerical_df.reset_index(drop=True, inplace=True)
        # print("the non numerical columns are: ",non_numerical_df)
        # print("the reduced features are: ",reducedFeatures)
        concatenated_features = pd.concat([non_numerical_df, reducedFeatures], axis=1)
        return concatenated_features


if __name__ == "__main__":
    # file_path = "data.csv"
    file_path = r"C:\Users\Alaa\Downloads\titanic\train.csv"

    problem = "Clustering"
    y_column = "Survived"
    AutoClean.clean_data(file_path, problem, y_column)
