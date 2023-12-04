import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype, is_integer_dtype, is_float_dtype
from pandas.core.dtypes.common import is_datetime64_any_dtype
from sklearn.preprocessing import StandardScaler, LabelEncoder


class Convert_to_datetime:
    """
    A class for converting object-type columns to datetime in a pandas DataFrame.

    Methods:
        - convert(df): Converts object-type columns to datetime.
    """

    def convert(self, df):
        """
        Convert object-type columns to datetime.

        Parameters:
            - df: pandas DataFrame

        Returns:
            - df: pandas DataFrame with converted datetime columns
        """

        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col])
                except ValueError:
                    pass

        return df


class MissingValues:
    """
    A class for handling missing values in a pandas DataFrame.

    Methods:
        - handle_nan(df, handle='auto'): Handles missing values based on the specified method.
        - fill_datetime_na(series): Fills missing values in a datetime series with sequential dates if it is sequential.
    """

    def detect_nulls(self, df):
        """
        Detects null values in each column and informs the user.

        Parameters:
            - df: pandas DataFrame

        Returns:
            - None
        """
        null_info = df.isnull().sum()
        columns_with_nulls = null_info[null_info > 0]

        if columns_with_nulls.empty:
            print("No null values detected.")
        else:
            print("Null values detected:")
            for col, count in columns_with_nulls.items():
                print(f"Column: {col}, Number of Nulls: {count}")
                print(f"Locations of Nulls: {df.index[df[col].isnull()].tolist()}")
                print("\n")
        return columns_with_nulls

    def handle_nan(self, df, fillNA_dict):
        """
        Handles missing values in the data.

        Parameters:
            - df: pandas DataFrame
            - handle: 'auto' or 'delete'

        Returns:
            - df: pandas DataFrame with missing values handled
        """
        for col, method in fillNA_dict.items():
            if method == 'auto':
                if not df[col].isnull().any():
                    continue
                if is_float_dtype(df[col]):
                    df[col].fillna(df[col].median(), inplace=True)
                elif is_string_dtype(df[col]) or is_integer_dtype(df[col]):
                    df[col].fillna(df[col].mode()[0], inplace=True)
                elif is_datetime64_any_dtype(df[col]):
                    df[col], status = self.fill_datetime_na(df[col])
                    if status == "failed":
                        df[col].dropna(inplace=True)
            elif method == 'median':
                if is_float_dtype(df[col]) or is_integer_dtype(df[col]):
                    df[col].fillna(df[col].median(), inplace=True)
            elif method == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif method == 'mean':
                if is_float_dtype(df[col]) or is_integer_dtype(df[col]):
                    df[col].fillna(df[col].mean()[0], inplace=True)
            elif method == 'delete':
                df.dropna(subset = [col], inplace=True)
        return df

    def fill_datetime_na(self, series):
        """
        Fills missing values in a datetime series based on the dynamically calculated pattern of dates.

        Parameters:
            - series: pandas Series
                The input datetime series with missing values to be filled.

        Returns:
            - filled_series: pandas Series
                The datetime series with filled missing values based on the dynamically calculated pattern.
            - status: str
                The status of the filling operation, either 'success' or 'failed'.
        """
        series_copy = series.copy()
        print("Nan locations: ", series_copy.index[series_copy.isnull()])
        status = "failed"
        old_index = series_copy.index
        series_copy = series_copy.reindex(range(old_index.min(), old_index.max() + 1))
        added_indices = series_copy.index.difference(old_index).tolist()
        print("added_indices",added_indices)

        if series_copy.isnull().any():
            max_sequential_non_nulls = 4
            max_count = 0
            count = 0
            start_index = 0

            # Find the index where the sequential non-null dates start
            for i, date in enumerate(series_copy):
                if pd.notnull(date):
                    count += 1
                    if count > max_count:
                        max_count = count
                        start_index = i - count + 1
                else:
                    count = 0

            # Check if the max number of sequential non-null dates is sufficient
            if max_count >= max_sequential_non_nulls:
                prev_gap = (series_copy[start_index + 1] - series_copy[start_index]).total_seconds() / (
                        60 * 60 * 24)  # Convert to days
                c = 0

                # Check if the gaps between sequential non-null dates are consistent
                for i in range(start_index + 1, max_sequential_non_nulls + start_index - 1):
                    gap = (series_copy[i + 1] - series_copy[i]).total_seconds() / (60 * 60 * 24)  # Convert to days
                    if gap == prev_gap:
                        c += 1

                dynamic_pattern = (series_copy[start_index + 1] - series_copy[start_index]).total_seconds() / (60 * 60 * 24)

                # If the gaps are consistent, fill NaN values using the dynamic pattern
                if c == (max_sequential_non_nulls - 2):
                    dynamic_pattern = (series_copy[start_index + 1] - series_copy[start_index]).total_seconds() / (60 * 60 * 24)
                    print("Gap Pattern = ", dynamic_pattern)

                    # Fill NaN values before the sequence
                    for i in range(start_index - 1, -1, -1):
                        series_copy.loc[i] = series_copy.loc[i + 1] - pd.to_timedelta(dynamic_pattern, unit='D')

                    # Fill NaN values after the sequence
                    for i in range(start_index + max_sequential_non_nulls, len(series_copy)):
                        series_copy.loc[i] = series_copy.loc[i - 1] + pd.to_timedelta(dynamic_pattern, unit='D')

                    status = "success"
                    series_copy = series_copy.drop(added_indices)

                else:
                    print("Failed to detect a pattern. All nulls in the date column were dropped")
                    status = "failed"

            else:
                # If the max number of sequential non-null dates is not sufficient, drop the row
                print("The Date column has too many nulls. Maybe you want to refill it manually and re-upload the data")
                print("All nulls in the date column were dropped")
                status = "failed"

        return series_copy, status


class Duplicates:
    """
       A class for handling duplicate values in a pandas DataFrame.

       Methods:
       - handle_dub(df, method='auto'): Handles duplicate values based on the specified method.
    """

    def handle_dub(self, df):
        """
        Handle duplicate values in the data.

        Parameters:
            - df: pandas DataFrame
        Returns:
            - df: pandas DataFrame with duplicate values handled
        """
        df.drop_duplicates(inplace=True)
        return df


class Outliers:
    """
    A class for handling outliers in a pandas DataFrame.

    Methods:
        - handle_outliers(df, method='z_score', handle='auto', threshold=3): Handles outliers based on the specified method.
    """

    def detect_outliers(self, df, threshold=3):
        """
        Detect outliers in the data and inform the user.

        Parameters:
            - df: pandas DataFrame
            - method: 'z_score' or specific method for detecting outliers
            - threshold: Z-score threshold for identifying outliers

        Returns:
            - None
        """
        outlier_info = {}

        for col in df.select_dtypes(include=['float64']).columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers_mask = z_scores > threshold

            if outliers_mask.any():
                outlier_info[col] = {
                    'locations': df.index[outliers_mask].tolist(),
                    'message': 'Extreme outliers detected. Consider handling them.'
                }
            elif np.any((z_scores > 1) & (z_scores <= threshold)):
                outlier_info[col] = {
                    'locations': df.index[z_scores > 1].tolist(),
                    'message': 'Mild outliers detected. Consider handling them.'
                }


        return outlier_info

    def handle_outliers(self, df, methods):
        """
        Handle outliers in the data.

        Parameters:
            - df: pandas DataFrame
            - methods: Dictionary with keys as column names, values as tuples (method, handle, threshold)
                       where method can be 'z_score' or 'IQR', handle can be 'auto', 'delete', 'median', 'mean', and threshold is the outlier threshold.

        Returns:
            - df: pandas DataFrame with outliers handled
        """
        # in the website threshold will be a default of 1.5 in case of IQR and 3 in case of z-score if the user doesn't enter another one.
        for col, (method, handle, threshold) in methods.items():
            outliers_mask = None
            if method == 'z_score':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers_mask = z_scores > threshold
            elif method == 'IQR':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)

            if handle == 'auto' or handle == 'median':
                df.loc[outliers_mask, col] = df[col].median()
            elif handle == 'delete':
                deletion_mask = ~outliers_mask
                df = df[deletion_mask]
            elif handle == 'mean':
                df.loc[outliers_mask, col] = df[col].mean()

        return df


class DataNormalization:
    """
        A class for normalizing numeric data in a pandas DataFrame.

        Methods:
        - normalize_data(df, method='standard'): Normalizes numeric data based on the specified method.
    """

    def normalize_data(self, df, method='standard'):
        """
        Normalize numeric data in the DataFrame.

        Parameters:
            - df: pandas DataFrame
            - method: 'standard' or specific method for normalization

        Returns:
            - df: pandas DataFrame with normalized numeric data
        """
        if method == 'standard':
            scaler = StandardScaler()
            df[df.select_dtypes(include=['float64']).columns] = scaler.fit_transform(
                df.select_dtypes(include=['float64']))
        return df


class EncodeCategorical:
    """
        A class for encoding categorical columns in a pandas DataFrame.

        Methods:
        - Encode(df, encoding_dict): Encodes categorical columns based on the specified encoding method.
    """

    def Encode(self, df, encoding_dict):
        """
        Encode categorical columns in a DataFrame based on the specified encoding method wanted by the user.

            Parameters:
            - df: pandas DataFrame
            - encoding_dict: a dictionary where keys are column names and values are encoding methods ("label" or "onehot")

        Returns:
            - df_encoded: pandas DataFrame with encoded categorical columns
        """

        df_encoded = df.copy()

        for col, method in encoding_dict.items():
            if method == "label":
                # Use LabelEncoder for ordinal encoding
                label_encoder = LabelEncoder()
                df_encoded[col] = label_encoder.fit_transform(df_encoded[col])

            elif method == "onehot":
                # Use OneHotEncoder for one-hot encoding
                df_encoded = pd.get_dummies(df_encoded, columns=[col])

        return df_encoded


class HandlingColinearity:
    """
    A class for handling co-linearity in numeric columns of a pandas DataFrame.

    Methods:
        - detect_low_variance(df_numeric, var_threshold=0.1): Detects and informs about columns with low variance.
        - handle_low_variance(df, actions): Handles low variance based on user actions.
        - detect_colinearity(df): Detects and informs about colinearity between numeric variables.
        - handle_colinearity(df, actions): Handles colinearity based on user actions.
    """

    def detect_low_variance(self, df, var_threshold=0.1):
        """
        Detect and inform about columns with low variance in numeric columns.

        Parameters:
            - df_numeric: pandas DataFrame with numeric columns
            - var_threshold: Variance threshold for detecting low variance

        Returns:
            - low_variance_columns: List of columns with low variance
        """
        df_numeric = df.select_dtypes(include=['number'])
        low_variance_columns = [column for column in df_numeric.columns if df_numeric[column].var() < var_threshold]
        if low_variance_columns:
            print("The following columns have low variance:")
            for col in low_variance_columns:
                print(f"- {col}")
        else:
            print("No columns with low variance detected.")
        return low_variance_columns

    def handle_low_variance(self, df, actions):
        """
        Handle low variance in numeric columns based on user actions.

        Parameters:
            - df: pandas DataFrame
            - actions: Dictionary with column names as keys and values 'keep', 'remove', or 'auto'

        Returns:
            - df: pandas DataFrame with low variance handled
        """
        for column, action in actions.items():
            if action == 'remove' or action == 'auto' and column in df.columns:
                df.drop(columns=column, inplace=True)
        return df

    def handling_colinearity(self, df, auto_handling=True):
        # Select numeric variables only
        numeric_df = df.select_dtypes(include='number')

        # Build Correlation Matrix and choose the correlated variables
        corr = round(df.corr(), 2)
        high_corr_pairs = (corr.abs() > 0.5) & (corr != 1)
        high_corr_variables = corr[high_corr_pairs].stack()
        variables_cleaned = high_corr_variables.drop_duplicates()

        if auto_handling:
            columns_to_drop = [variables_cleaned.index[i][0] for i in range(len(variables_cleaned))]
            columns_to_keep = [variables_cleaned.index[i][1] for i in range(len(variables_cleaned))]
            print("I have removed this/those column/s: ", columns_to_drop)
            print("and kept this/those column/s: ", columns_to_keep)
            df.drop(columns=columns_to_drop, inplace=True)
        else:
            columns_to_drop = []
            for i in range(len(variables_cleaned)):
                print("Which variable do you want to remove?")
                print(f"1-{variables_cleaned.index[i][0]}, 2-{variables_cleaned.index[i][1]}")
                choice = int(input())
                if choice == 1:
                    columns_to_drop.append(variables_cleaned.index[i][0])
                elif choice == 2:
                    columns_to_drop.append(variables_cleaned.index[i][1])
            df.drop(columns=columns_to_drop, inplace=True)

        return df

class test_server:
    def change_parameters(self,parameters):
        parameters['test']="the server connected to scripts successfully"
        return parameters



