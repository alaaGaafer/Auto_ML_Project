import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype, is_integer_dtype, is_float_dtype
from pandas.core.dtypes.common import is_datetime64_any_dtype
from sklearn.preprocessing import StandardScaler, LabelEncoder


class Convert_to_datetime:

    def convert(self, df):

        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col])
                except ValueError:
                    pass

        return df


class MissingValues:

    def handle_nan(self, df, handle='auto'):
        # Function for handling missing values in the data
        if handle == 'auto':
            for col in df.columns:
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

        elif handle == 'delete':
            df.dropna(inplace=True)
        return df

    def fill_datetime_na(self, series):
        # Fill missing values in a datetime series with sequential dates if it is sequential
        date_diffs = series.diff()

        if date_diffs.min() == date_diffs.max() == pd.Timedelta(days=1):
            start_date = series.min()
            end_date = series.max()
            date_range = pd.date_range(start_date, end_date, freq='D')
            series_filled = series.combine_first(pd.Series(date_range))
            print("Nan locations: ", series.index[series.isnull()])
            status = "success"
        else:
            # If the datetime values are not sequential, drop the row
            status = "failed"

        return series_filled, status


class Duplicates:

    def handle_dub(self, df, method="auto"):
        # Function for handling Duplications values in the data
        if method == "auto":
            for col in df.columns:
                df.drop_duplicates(inplace=True)
        return df


class Outliers:

    def handle_outliers(self, df, method='z_score', handle='auto', threshold=3):
        if method == 'z_score':
            for col in df.select_dtypes(include=['float64']).columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers_mask = z_scores > threshold
                if handle == 'auto':
                    df.loc[outliers_mask, col] = df[col].median()
                elif handle == 'delete':
                    deletion_mask = ~outliers_mask
                    df = df[deletion_mask]
                    print(df.head(61))

        return df


class DataNormalization:

    def normalize_data(self, df, method='standard'):
        if method == 'standard':
            scaler = StandardScaler()
            df[df.select_dtypes(include=['float64']).columns] = scaler.fit_transform(
                df.select_dtypes(include=['float64']))
        return df


class EncodeCategorical:

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
    def remove_low_variance(self, df_numeric, var_threshold=0.1):
        columns_to_drop = []

        for column in df_numeric.columns:
            if df_numeric[column].var() < var_threshold:
                columns_to_drop.append(column)

        df_numeric.drop(columns=columns_to_drop, inplace=True)
        return df_numeric

    def handling_colinearity(self, df, auto_handling=True):
        # Select numeric variables only
        numeric_df = df.select_dtypes(include='number')

        # Remove variables with low variance
        new_df = self.remove_low_variance(numeric_df)

        # Build Correlation Matrix and choose the correlated variables
        corr = round(new_df.corr(), 2)
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




