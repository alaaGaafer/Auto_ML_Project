from functions import *


class AutoClean:

    @staticmethod
    def clean_data(file_path):
        # Read CSV file
        df = pd.read_csv(file_path)
        df = df.reset_index(drop=True)

        # Convert to datetime
        df = AutoClean.convert_to_datetime(df)
        # Handle missing values
        df = AutoClean.handle_missing_values(df)
        print(df.head(20))
        # Handle duplicates
        df = AutoClean.handle_duplicates(df)

        # Detect and handle outliers
        df = AutoClean.handle_outliers(df)

        # Encode categorical variables
        encoding_dict = {"Gender": "onehot"}
        df = AutoClean.encode_categorical(df, encoding_dict)

        # Handle low variance columns
        low_variance_columns = HandlingColinearity().detect_low_variance(df)
        actions = {c: "auto" for c in low_variance_columns}
        df = HandlingColinearity().handle_low_variance(df, actions)

        # Detect and handle colinearity
        df = HandlingColinearity().handling_colinearity(df)

        print(df)

    @staticmethod
    def convert_to_datetime(df):
        print("info before", df.info())
        df = Convert_to_datetime().convert(df)
        print("info after", df.info())
        return df

    @staticmethod
    def get_fill_method_input(df, col_name):
        col_type = df.dtypes[col_name]

        if pd.api.types.is_numeric_dtype(col_type):  # Check if the column is numeric (int or float)
            fill_method = input(f"Enter method for handling NaN in '{col_name}' (mean, median, mode, delete): ").lower()
            valid_methods = ["mean", "median", "mode", "delete"]
        elif pd.api.types.is_string_dtype(col_type):  # Check if the column is string
            fill_method = input(f"Enter method for handling NaN in '{col_name}' (mode, delete): ").lower()
            valid_methods = ["mode", "delete"]
        else:
            fill_method = "auto"  # For date columns, assign it as "auto"
            valid_methods = ["auto"]

        while fill_method not in valid_methods:
            print(f"Invalid input. Please enter one of: {', '.join(valid_methods)}")
            fill_method = input(f"Enter method for handling NaN in '{col_name}': ").lower()

        return fill_method

    @staticmethod
    def handle_missing_values(df):
        print('nan before: ', df.isna().sum())
        columns_with_nulls = MissingValues().detect_nulls(df)
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
    def handle_outliers(df):
        outlier_info = Outliers().detect_outliers(df)

        print("\nOutliers Information:")
        for col, info in outlier_info.items():
            print(f"Column: {col}")
            print(f"Outlier Locations: {info['locations']}")
            print(f"Message: {info['message']}")
            print("\n")

        methods_input = AutoClean.get_outliers_handling_input(outlier_info)
        df = Outliers().handle_outliers(df, methods_input)
        return df

    @staticmethod
    def encode_categorical(df, encoding_dict):
        print("Encoded: ")
        return EncodeCategorical().Encode(df, encoding_dict)


if __name__ == "__main__":
    file_path = "data.csv"
    AutoClean.clean_data(file_path)
