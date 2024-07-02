import pandas as pd
import numpy as np
from pandas.api.types import is_object_dtype, is_integer_dtype, is_float_dtype
from pandas.core.dtypes.common import is_datetime64_any_dtype
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt


class RemoveIDColumn:
    """
    A class for automatically identifying and removing potential ID columns
    based on either column names containing "id" or "ID" or high cardinality.

    Methods:
        - remove_id_column(df): Identifies and removes ID columns.
    """

    @classmethod
    def remove_high_cardinality_columns(cls, df):
        """
        Identifies and removes potential ID columns based on high cardinality.

        Parameters:
            - df: pandas DataFrame
        Returns:
            - df: pandas DataFrame with removed high cardinality columns
        """
        messages = []
        # Select only categorical columns
        categorical_columns = df.select_dtypes(include='object').columns

        for col in categorical_columns:
            # Calculate the unique ratio after dropping NaNs in the column
            unique_ratio = df[col].nunique() / len(df[col].dropna())

            # If the ratio is 95% unique, consider it for removal
            if unique_ratio >= 0.95:
                df = df.drop(columns=col)
                messages.append(f"Removed high cardinality column: {col}")

        return df


class ConvertToDatetime:
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
        # date_formats = [
        #     "%Y-%m-%dT%H:%M:%S",
        #     "%Y-%m-%dT%H:%M:%SZ",
        #     "%B %d, %Y %I:%M:%S %p",
        #     "%m/%d/%Y %H:%M",
        #     "%A, %B %d, %Y %H:%M",
        #     "%b %d, %Y %H:%M",
        #     "%d/%m/%Y %H:%M",
        #     "%d-%b-%Y %H:%M",
        #     "%Y-%m-%dT%H:%M:%S.%f"
        # ]

        # Get columns with object dtype
        object_cols = df.select_dtypes(include=['object']).columns

        # Iterate through each column
        for col in object_cols:
            if 'date' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    break  # Break the loop if conversion is successful
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

    # detection will appear as a notification to the user
    def detect_nulls(self, df):
        """
        Detects null values in each column and returns a dictionary with columns with nulls,
        their type, number of nulls, and locations.

        Parameters:
            - df: pandas DataFrame

        Returns:
            - dictionary containing columns with nulls, their type, number of nulls, and locations
        """
        null_info = df.isnull().sum()
        columns_with_nulls = null_info[null_info > 0]

        nulls_dict = {}

        if columns_with_nulls.empty:
            return None
        else:
            for col, count in columns_with_nulls.items():
                nulls_dict[col] = {
                    "type": df[col].dtype,
                    "number_of_nulls": count,
                    "locations_of_nulls": df.index[df[col].isnull()].tolist()
                }
        return list(columns_with_nulls.index)

    def del_high_null_cols(self, df):
        messages = []
        null_info = df.isnull().sum()
        columns_with_nulls = null_info[null_info > 0].index

        for col in columns_with_nulls:
            null_percentage = df[col].isnull().mean() * 100
            if null_percentage > 75:
                messages.append(
                    f"Column '{col}' has too many null values ({null_percentage:.2f}%). Dropping the column.")
                df.drop(columns=[col], inplace=True)
                continue

            if is_datetime64_any_dtype(df[col]):
                df[col], status, date_message = self.fill_datetime_na(df[col])
                messages.append(date_message)
                if status == "failed":
                    df[col].dropna(inplace=True)

        return df

    def handle_nan(self, df, fillNA_dict):
        """
        Handles missing values in the data.

        Parameters:
            - df: pandas DataFrame
            - fillNA_dict: A dictionary contains columns as keys and handling methods chosen by the user (if not specified it gets auto handled)

        Returns:
            - df: pandas DataFrame with missing values handled
        """
        for col, method in fillNA_dict.items():
            if not df[col].isnull().any():
                continue

            if method == 'auto':
                if is_float_dtype(df[col]):
                    df[col].fillna(df[col].median(), inplace=True)
                elif is_object_dtype(df[col]) or is_integer_dtype(df[col]):
                    mode_values = df[col].mode()
                    if not mode_values.empty:
                        chosen_mode = mode_values.iloc[0].split()[0]
                        df[col].fillna(chosen_mode, inplace=True)
                # elif is_datetime64_any_dtype(df[col]):
                #     df[col], status = self.fill_datetime_na(df[col])
                #     if status == "failed":
                #         df[col].dropna(inplace=True)
            elif method == 'median':
                if is_float_dtype(df[col]) or is_integer_dtype(df[col]):
                    df[col].fillna(df[col].median(), inplace=True)
            elif method == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif method == 'mean':
                if is_float_dtype(df[col]) or is_integer_dtype(df[col]):
                    df[col].fillna(df[col].mean(), inplace=True)
            elif method == 'delete':
                df.dropna(subset=[col], inplace=True)
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
        # print("Nan locations: ", series_copy.index[series_copy.isnull()])
        status = "failed"
        old_index = series_copy.index
        series_copy = series_copy.reindex(range(old_index.min(), old_index.max() + 1))
        added_indices = series_copy.index.difference(old_index).tolist()
        # print("added_indices", added_indices)

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

                dynamic_pattern = (series_copy[start_index + 1] - series_copy[start_index]).total_seconds() / (
                        60 * 60 * 24)

                # If the gaps are consistent, fill NaN values using the dynamic pattern
                if c == (max_sequential_non_nulls - 2):
                    dynamic_pattern = (series_copy[start_index + 1] - series_copy[start_index]).total_seconds() / (
                            60 * 60 * 24)
                    # print("Gap Pattern = ", dynamic_pattern)

                    # Fill NaN values before the sequence
                    for i in range(start_index - 1, -1, -1):
                        series_copy.loc[i] = series_copy.loc[i + 1] - pd.to_timedelta(dynamic_pattern, unit='D')

                    # Fill NaN values after the sequence
                    for i in range(start_index + max_sequential_non_nulls, len(series_copy)):
                        series_copy.loc[i] = series_copy.loc[i - 1] + pd.to_timedelta(dynamic_pattern, unit='D')

                    status = "success"
                    series_copy = series_copy.drop(added_indices)

                else:
                    message = "Failed to detect a pattern. All nulls in the date column were dropped"
                    status = "failed"

            else:
                # If the max number of sequential non-null dates is not sufficient, drop the row
                message = "The Date column has too many nulls. Maybe you want to refill it manually and re-upload the data", "All nulls in the date column were dropped"
                status = "failed"

        return series_copy, status, message


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
        - detect_outliers(df, threshold=3): Detects outliers using z-score and informs the user.
        - handle_outliers(df, methods): Handles outliers based on the specified method.
    """

    def detect_outliers(self, df, threshold=3):
        """
        Detect outliers in the data and inform the user.

        Parameters:
            - df: pandas DataFrame
            - method: 'z_score'
            - threshold: Z-score threshold for identifying outliers

        Returns:
            - outlier_info
        """
        outlier_info = {}
        cols_with_outliers = []

        for col in df.select_dtypes(include=['float64']).columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers_mask = z_scores > threshold

            if outliers_mask.any():
                outlier_info[col] = {
                    'locations': df.index[outliers_mask].tolist(),
                    'status': 'Extreme'
                }
                cols_with_outliers.append(col)
            elif np.any((z_scores > 1) & (z_scores <= threshold)):
                outlier_info[col] = {
                    'locations': df.index[z_scores > 1].tolist(),
                    'status': 'Mild'
                }
                cols_with_outliers.append(col)

        # print("\nOutliers Information:")
        # for col, info in outlier_info.items():
        #     print(f"Column: {col}")
        #     print(f"Outlier Locations: {info['locations']}")
        #     print(f"Status: {info['status']}")
        #     print("\n")

        return cols_with_outliers

    def handle_outliers(self, df, cols, choices):
        """
        Handle outliers in the data.

        Parameters:
            - df: pandas DataFrame
            - choices: A dict with 3 keys and their values (method, handle, threshold)where method can be
            'z_score' or 'IQR', handle can be 'auto', 'delete', 'median', 'mean', and threshold is the outlier threshold.

        Returns:
            - df: pandas DataFrame with outliers handled
        """
        # in the website threshold will be a default of 1.5 in case of IQR and 3 in case of z-score if the user doesn't enter another one.
        method = choices[0]
        handle = choices[1]
        threshold = choices[2]

        if method not in ['z_score', 'IQR']:
            raise ValueError("Method must be either 'z_score' or 'IQR'")
        if handle not in ['auto', 'delete', 'median', 'mean']:
            raise ValueError("Handle must be one of 'auto', 'delete', 'median', 'mean'")

        # Setting default thresholds if not provided
        if threshold is None:
            threshold = 1.5 if method == 'IQR' else 3

        for col in cols:
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


class HandlingImbalanceClasses:
    """
    Class for handling class imbalance in a dataset.

    Methods:
        detect_class_imbalance(self, df, target_column):
            Function to detect class imbalance and inform the user.
        handle_class_imbalance(self, df, target_column, instruction='auto'):
            Function to handle class imbalance based on user instruction.
    """

    def detect_class_imbalance(self, df, target_column):
        """
        Function to detect class imbalance in a dataset.

        Parameters:
            df (DataFrame): The input dataset.
            target_column (str): The name of the target column.

        Returns:
            str: Information about class imbalance in the dataset.
            bool: Whether class imbalance is detected.
        """
        class_distribution = df[target_column].value_counts()
        imbalance_info = "Class Imbalance Information:\n"
        imbalance_detected = False

        for class_label, count in class_distribution.items():
            imbalance_ratio = count / len(df)
            if imbalance_ratio < 0.05 or imbalance_ratio > 0.95:
                imbalance_info += f" - Class '{class_label}' has an imbalance ratio of {imbalance_ratio:.2f}\n"
                imbalance_detected = True

        if not imbalance_detected:
            imbalance_info += "No significant class imbalance detected."

        return imbalance_detected

    def handle_class_imbalance(self, df, target_column, instruction='auto'):
        """
        Function to handle class imbalance based on user instruction.

        Parameters:
            df (DataFrame): The input dataset.
            target_column (str): The name of the target column.
            instruction (str): User instruction on how to handle class imbalance.
                Options: 'oversampling', 'undersampling', 'auto' (default).

        Returns:
            DataFrame: Resampled dataset.
        """

        if instruction == 'oversampling':
            oversampling = RandomOverSampler(sampling_strategy='minority')
            X_resampled, y_resampled = oversampling.fit_resample(df.drop(columns=[target_column]), df[target_column])
            resampled_data = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled, columns=[target_column])],
                                       axis=1)
            return resampled_data

        elif instruction == 'undersampling':
            undersampling = RandomUnderSampler(sampling_strategy='majority')
            X_resampled, y_resampled = undersampling.fit_resample(df.drop(columns=[target_column]),
                                                                  df[target_column])
            resampled_data = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled, columns=[target_column])],
                                       axis=1)
            return resampled_data
        elif instruction == 'auto':
            smote = SMOTE()
            X_resampled, y_resampled = smote.fit_resample(df.drop(columns=[target_column]), df[target_column])
            resampled_data = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled, columns=[target_column])],
                                       axis=1)
            return resampled_data

        else:
            return df


class DataNormalization:
    """
        A class for normalizing numeric data in a pandas DataFrame.

        Methods:
        - normalize_data(df, method): Normalizes numeric data based on the specified method either with standard scaler(auto) or Minmax scaler.
    """

    def normalize_data(self, df, method='auto'):
        """
        Normalize numeric data in the DataFrame.

        Parameters:
            - df: pandas DataFrame
            - method: 'standard' or specific method for normalization

        Returns:
            - df: pandas DataFrame with normalized numeric data
        """
        if method == 'standard' or 'auto':
            scaler = StandardScaler()
            df[df.select_dtypes(include=['float64']).columns] = scaler.fit_transform(
                df.select_dtypes(include=['float64']))
        elif method == 'MinMax':
            scaler = MinMaxScaler()
            df[df.select_dtypes(include=['float64']).columns] = scaler.fit_transform(
                df.select_dtypes(include=['float64']))
        return df


class EncodeCategorical:
    """
        A class for encoding categorical columns in a pandas DataFrame.

        Methods:
        - Encode(df, encoding_dict): Encodes categorical columns based on the specified encoding methods or auto encoded if not specified.
    """

    def Encode(self, df, encoding_dict):
        """
        Encode categorical columns in a DataFrame based on the specified encoding method wanted by the user.

        Parameters:
            - df: pandas DataFrame
            - encoding_dict: a dictionary where keys are column names and values are encoding methods ("label" or "onehot" or auto)

        Returns:
            - df_encoded: pandas DataFrame with encoded categorical columns
        """
        # Create a copy of the original DataFrame
        df_encoded = df.copy()

        for col, method in encoding_dict.items():
            if method == "label" or (method == "auto" and df[col].nunique() >= 5):
                # Use LabelEncoder for ordinal encoding
                label_encoder = LabelEncoder()
                df_encoded[col] = label_encoder.fit_transform(df_encoded[col])

            elif method == "onehot" or (method == "auto" and df[col].nunique() < 5):
                # Use OneHotEncoder for one-hot encoding
                df_encoded = pd.get_dummies(df_encoded, columns=[col], drop_first=True)

        return df_encoded


class HandlingColinearity:
    """
    A class for handling co-linearity in numeric columns of a pandas DataFrame.

    functions:
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
        low_variance_info = {}

        if low_variance_columns:
            low_variance_info['low_variance_detected'] = True
            low_variance_info['low_variance_columns'] = low_variance_columns
        else:
            low_variance_info['low_variance_detected'] = False

        return low_variance_columns, low_variance_info

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

    def handling_colinearity(self, df):
        # Select numeric variables only
        numeric_df = df.select_dtypes(include='number')

        # Build Correlation Matrix and choose the correlated variables
        corr = round(numeric_df.corr(), 2)
        high_corr_pairs = (corr.abs() > 0.5) & (corr != 1)
        high_corr_variables = corr[high_corr_pairs].stack()
        variables_cleaned = high_corr_variables.drop_duplicates()
        handling_info = {}

        if not variables_cleaned.empty:
            columns_to_drop = [variables_cleaned.index[i][0] for i in range(len(variables_cleaned))]
            columns_to_keep = [variables_cleaned.index[i][1] for i in range(len(variables_cleaned))]
            handling_info['co-linearity_detected'] = True
            handling_info['removed_columns'] = columns_to_drop
            handling_info['kept_columns'] = columns_to_keep
            df.drop(columns=columns_to_drop, inplace=True)

        return df


class HandlingReduction:
    """
    A class for handling dimensionality reduction in a pandas DataFrame.

    functions:
        - feature_reduction(df, num_components_to_keep): Reduces the dimensionality of the DataFrame using PCA.
        - explainedVariability(df): Returns the cumulative explained variance and the number of components to keep.
        - plotExplainedVariance(X_pca): Plots the explained variance of the PCA components.
        - apply_feature_reduction(data, y_column, reduce, auto_reduce, feature_reduction_method, handling_reduction): Applies feature reduction.
    """

    def feature_reduction(self, df, num_components_to_keep, pca_model=None):
        X = df.copy()
        X = X.select_dtypes(include=[np.number])
        if pca_model is None:
            pca = PCA(n_components=num_components_to_keep)
            X_pca = pca.fit_transform(X)
        else:
            X_pca = pca_model.transform(X)
        reduced_df = pd.DataFrame(data=X_pca, columns=[f'PC_{i + 1}' for i in range(num_components_to_keep)])
        return reduced_df

    def explainedVariability(self, df):
        explanation_list = []
        X = df.copy()
        X = X.select_dtypes(include=[np.number])
        pca = PCA()
        pca.fit(X)
        explainVariance = pca.explained_variance_ratio_
        cumsum = np.cumsum(explainVariance)
        for i, value in enumerate(cumsum):
            explanation_list.append(f"The explained variability with {i + 1} components is: {value}")
        return cumsum, explanation_list

    def NumberOfComponents(self, df, cumsum):
        explainVariancethreshold = 0.9
        num_components_to_keep = np.argmax(cumsum >= explainVariancethreshold) + 1
        return num_components_to_keep

    def plotExplainedVariance(self, X_pca):
        numberofcomponents = len(X_pca)
        if numberofcomponents <= 2:
            plt.plot(X_pca)
            plt.xlabel('number of components')
            plt.ylabel('cumulative explained variance')
            plt.show()
        # else:
        #     return "Number of components is greater than 2, so cannot plot the explained variance"
    # def selectFeatures(self,df):

    #     Y=df.iloc[:,-1]
    #     X=df.iloc[:,:-1]
    #     X= X.select_dtypes(include=[np.number]).interpolate().dropna()
    #     model = RandomForestClassifier(n_estimators=100, random_state=42)
    #     model.fit(X, Y)
    #     sfm = SelectFromModel(model, prefit=True)
    #     selected_feature_indices = np.where(sfm.get_support())[0]
    #     get_feature_names = X.columns[selected_feature_indices]
    #     return get_feature_names

# class test_server:
#     def change_parameters(self, parameters):
#         parameters['test'] = "the server connected to scripts successfully"
#         return parameters