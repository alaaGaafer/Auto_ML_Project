import argparse
import pandas as pd
from functions import *
class AutoClean:

    @staticmethod
    def clean_data(file_path):
        df = pd.read_csv(file_path)
        df = df.reset_index(drop=True)

        print("info before", df.info())
        df = Convert_to_datetime().convert(df)
        print("info after", df.info())

        print('nan before: ', df.isna().sum())
        df = MissingValues().handle_nan(df, "auto")
        print('nan after: ', df.isna().sum())

        print('dub before: ', df.duplicated().sum())
        df = Duplicates().handle_dub(df)
        print('dub after: ', df.duplicated().sum())

        # print("before handling outliers: ")
        # print(df[df["Calories"] > 1250])
        df = Outliers().handle_outliers(df)
        # print("after handling outliers: ",df[df["Calories"] > 1250])
        encoding_dict = {"Gender":"onehot"}
        df = EncodeCategorical().Encode(df, encoding_dict)
        print("Encoded: ")
        print(df.head(20))
        df = HandlingColinearity().handling_colinearity(df)
        print(df.head())


if __name__ == "__main__":
    ### when called from cmd
    # parser = argparse.ArgumentParser()
    # parser.add_argument("file_path", type=str, help="Path to the CSV file to be cleaned.")
    # args = parser.parse_args()
    #
    # AutoClean.clean_data(args.file_path)
    file_path = "data.csv"
    AutoClean.clean_data(file_path)