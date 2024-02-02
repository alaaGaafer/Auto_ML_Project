from functions import *


class AutoClean:

    @staticmethod
    def clean_data(File_path, Is_clustering):
        # Read CSV file
        df = pd.read_csv(File_path)
        print(df)
        df = df.reset_index(drop=True)
        df_copy = df.copy()

        # Convert to datetime
        # df_copy = AutoClean.convert_to_datetime(df_copy)

        # removing Id column
        if not Is_clustering:
            df_copy = RemoveIDColumn.remove_id_column(df_copy)

        # Handle missing values , next line will be a notification in front-end
        columns_with_nulls = MissingValues().detect_nulls(df_copy)

        # next line should be called from the front-end after entering the user choices
        df_copy = AutoClean.handle_missing_values(df_copy, columns_with_nulls)
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
        low_variance_columns = HandlingColinearity().detect_low_variance(df_copy)
        actions = {c: "auto" for c in low_variance_columns}
        # next line should be called from the front-end after entering the user choices
        df_copy = HandlingColinearity().handle_low_variance(df_copy, actions)
        # Detect and handle co-linearity
        df_copy = HandlingColinearity().handling_colinearity(df_copy)

        # Encode categorical variables
        encoding_dict = EncodeCategorical().categorical_columns(df_copy)
        df_copy = AutoClean.encode_categorical(df_copy, encoding_dict)
        # Reduce dimensions
        # df_copy = AutoClean.reduceDimensions(df_copy)

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
        elif pd.api.types.is_string_dtype(col_type):  # Check if the column is string
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
    def handle_missing_values(df, columns_with_nulls):
        print('nan before: ', df.isna().sum())
        fillNA_dict = {}

        for col, null_count in columns_with_nulls.items():
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
        print("Encoded: ")
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
            explainedVariability,num_components_to_keep=HandlingReduction().explainedVariability(numericalDf)
            for i,value in enumerate(explainedVariability):
                
                print(f"The explained variability with {i+1} components is: {value}")
            
            print("Do you want to handle the reduction automatically? (yes/no)")
            choice = input().lower()
            if choice == "yes" or choice == "y":
                reducedFeatures=HandlingReduction().feature_reduction(numericalDf,num_components_to_keep)
            else:
                choice=int(input("Enter the number of components you want to keep: "))
                reducedFeatures=HandlingReduction().feature_reduction(numericalDf,choice)
        else:
            print("The number of dimensions will not be reduced")
            return df
        # print('the reduced features are: ',reducedFeatures)
        non_numerical_columns = df.select_dtypes(exclude=['number']).columns
        non_numerical_df = df[non_numerical_columns]
        concatenated_features = pd.concat([non_numerical_df, pd.DataFrame(reducedFeatures)], axis=1)
        return concatenated_features


if __name__ == "__main__":
    # file_path = "data.csv"
    is_clustering = False
    AutoClean.clean_data("data.csv", is_clustering)
