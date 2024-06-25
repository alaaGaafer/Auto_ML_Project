import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

folder_path = r"C:\Users\Alaa\Desktop\Auto_ML_Project\non-sequential-preprocessing_Scripts\similaritySearch\Regression - reg41\Regression"

csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]


def remove_high_cardinality_columns(Df, threshold=90):
    for column in Df.columns:
        if (Df[column].dtype == 'object' and
            Df[column].nunique() > threshold and
            ('name' in column.lower() or 'id' in column.lower())):
            Df.drop(columns=[column], inplace=True)
            print(f"Removed high cardinality column: {column}")
    return Df


def convert_categorical_to_numerical(Df):
    # Identify categorical columns
    categorical_cols = Df.select_dtypes(include=['object']).columns
    df_encoded = df.copy()

    for col in categorical_cols:
        if df[col].nunique() >= 5:
            # Use LabelEncoder for ordinal encoding
            label_encoder = LabelEncoder()
            df_encoded[col] = label_encoder.fit_transform(df_encoded[col])

        elif df[col].nunique() < 5:
            df_encoded = pd.get_dummies(df_encoded, columns=[col], drop_first=True)
    return df_encoded


for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)

    df = remove_high_cardinality_columns(df)

    df = convert_categorical_to_numerical(df)

    output_file_path = os.path.join(folder_path, f'modified_{file}')
    df.to_csv(output_file_path, index=False)
