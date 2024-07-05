from similaritySearch.functions import *
from similaritySearch.metaFeatureExtraction import *
from similaritySearch.sim_function import *
from bestmodel import *


def calculate_date_frequency(series):
    """
    Calculate the most common frequency (interval) of a datetime series.

    Parameters:
    series (pd.Series): A pandas Series of datetime objects.

    Returns:
    str: A string representing the most common frequency (e.g., 'D' for days, 'H' for hours).
    """
    # Drop NaN values to avoid errors in calculations
    series = series.dropna()
    # Calculate the differences between consecutive dates
    diffs = series.diff().dropna()
    # Calculate the most common frequency
    freq = diffs.mode()[0]
    # Convert the frequency to a string representation
    series = series.dropna()
    # Calculate the differences between consecutive dates
    diffs = series.diff().dropna()
    # Calculate the most common frequency
    freq = diffs.mode()[0]
    # Convert the frequency to a string representation
    if freq == pd.Timedelta(days=1):
        return 'D'  # Daily
    elif freq >= pd.Timedelta(days=7):
        return 'W'  # Weekly
    elif freq >= pd.Timedelta(days=30):
        return 'M'  # Monthly
    elif freq >= pd.Timedelta(days=90):
        return 'Q'  # Quartely
    elif freq >= pd.Timedelta(days=365):
        return 'A'  # Annually
    else:
        return freq  # Return the exact frequency if it's not a common one


class AutoClean:
    def __init__(self, historical_df, y_column, problem, date_col=None):
        self.historical_df = historical_df
        self.y_column = y_column
        self.date_col = date_col
        self.problem = problem.lower()

    def extract_and_search_features(self, historical_df, dataset_name="train.csv"):
        knowledge_base_path = {
            "regression": "similaritySearch/Knowledge bases/new_knowledgeBaseReg.csv",
            "classification": "similaritySearch/Knowledge bases/new_knowledgeBaseCls.csv",
        }.get(self.problem)

        if not knowledge_base_path:
            raise ValueError(f"Unsupported problem type: {self.problem}")

        # dataset_name = self.historical_file_path.split('.')[0]
        meta_extractor = metafeatureExtraction(historical_df, dataset_name, knowledge_base_path, self.y_column)
        meta_features = meta_extractor.getMetaFeatures()
        meta_features_numerical = meta_features[1:]
        print("meta_features: ", meta_features)

        similarity_checker = MetaFeatureSimilarity(meta_features_numerical, knowledge_base_path)
        best_models, dataset_names = similarity_checker.get_best_models()
        best_models = list(set(best_models))
        print("best_models", best_models, "dataset_names", dataset_names)

        return meta_extractor, meta_features, best_models

    def Detections(self):
        if self.date_col:
            self.historical_df[self.date_col] = pd.to_datetime(self.historical_df[self.date_col])

        if self.problem == "timeseries":
            self.historical_df.rename(columns={self.date_col: 'ds', self.y_column: 'y'}, inplace=True)
            self.date_col = 'ds'
            self.y_column = 'y'
            self.historical_df['y'] = self.historical_df['y'].str.replace('[^0-9\.]', '', regex=True)
            self.historical_df['y'] = pd.to_numeric(self.historical_df['y'], errors='coerce')
            self.historical_df['y'] = self.historical_df['y'].astype(float)
        elif self.problem == "regression":
            self.historical_df['y'] = self.historical_df['y'].str.replace('[^0-9\.]', '', regex=True)
            self.historical_df['y'] = pd.to_numeric(self.historical_df['y'], errors='coerce')
            self.historical_df['y'] = self.historical_df['y'].astype(float)

        self.historical_df = RemoveIDColumn.remove_high_cardinality_columns(self.historical_df)
        self.historical_df = MissingValues().del_high_null_cols(self.historical_df)
        categorical_columns = self.historical_df.select_dtypes(include=['object']).columns.tolist()

        nulls_columns = MissingValues().detect_nulls(self.historical_df)
        cols_with_outliers = Outliers().detect_outliers(self.historical_df)
        self.historical_df = Duplicates().handle_dub(self.historical_df)
        imbalance_detected = None
        if self.problem == "classification":
            imbalance_detected = HandlingImbalanceClasses().detect_class_imbalance(self.historical_df, self.y_column)
        df_without_y = self.historical_df.drop(columns=[self.y_column])
        low_variance_columns, low_variance_info = HandlingColinearity().detect_low_variance(df_without_y)

        return self.historical_df, nulls_columns, cols_with_outliers, imbalance_detected, low_variance_columns, categorical_columns

    @staticmethod
    def _process_data(df: pd.DataFrame, fill_na_dict: dict, outliers_methods_input: tuple,
                      Norm_method: str) -> pd.DataFrame:
        """Handle missing values, outliers, normalization, and encoding."""
        df = MissingValues().handle_nan(df, fill_na_dict)
        cols_with_outliers = Outliers().detect_outliers(df)
        df = Outliers().handle_outliers(df, cols_with_outliers, outliers_methods_input)
        df = DataNormalization().normalize_data(df, Norm_method)

        return df

    def prediction_data(self, pred_df, fill_na_dict, outliers_methods_input, Norm_method, lowvariance_actions,
                        encoding_dict,
                        imb_instruction, Bestmodelobj, num_components_to_keep=None, pca_model=None):
        if self.date_col:
            pred_df[self.date_col] = pd.to_datetime(pred_df[self.date_col])

        if self.problem == "timeseries":
            pred_df.rename(columns={self.date_col: 'ds', self.y_column: 'y'}, inplace=True)
            self.date_col = 'ds'
            self.y_column = 'y'
            pred_df['y'] = pred_df['y'].str.replace('[^0-9\.]', '', regex=True)
            pred_df['y'] = pd.to_numeric(pred_df['y'], errors='coerce')
            pred_df['y'] = pred_df['y'].astype(float)
        elif self.problem == "regression":
            self.historical_df['y'] = self.historical_df['y'].str.replace('[^0-9\.]', '', regex=True)
            self.historical_df['y'] = pd.to_numeric(self.historical_df['y'], errors='coerce')
            self.historical_df['y'] = self.historical_df['y'].astype(float)

        pred_df, carednality_messages = RemoveIDColumn.remove_high_cardinality_columns(pred_df)
        pred_df, deletion_messages = MissingValues().del_high_null_cols(pred_df)

        # Handle missing values, duplicates, outliers, normalization, and co-linearity in pred_data
        pred_fill_na_dict = {}
        null_info = df.isnull().sum()
        pred_nulls_columns = null_info[null_info > 0]
        for col in pred_nulls_columns:
            for H_col, method in fill_na_dict.items():
                if col == H_col:
                    pred_fill_na_dict[col] = method
                else:
                    pred_fill_na_dict[col] = "auto"

        pred_df = self._process_data(pred_df, pred_fill_na_dict, outliers_methods_input, Norm_method)

        pred_df_without_y = pred_df.drop(columns=[self.y_column])
        pred_df_without_y = HandlingColinearity().handle_low_variance(pred_df_without_y, lowvariance_actions)
        pred_df_without_y, handling_info = HandlingColinearity().handling_colinearity(pred_df_without_y)
        pred_df_without_y[self.y_column] = pred_df[self.y_column]
        pred_df = pred_df_without_y.copy()
        pred_df = EncodeCategorical().Encode(pred_df, encoding_dict)
        if pca_model:
            numerical_columns = pred_df.drop(columns=[self.y_column]).select_dtypes(include=[np.number]).columns
            pred_df = HandlingReduction().feature_reduction(pred_df[numerical_columns],
                                                            num_components_to_keep, pca_model)
        if self.problem == "classification":
            pred_df = HandlingImbalanceClasses().handle_class_imbalance(pred_df, self.y_column, imb_instruction)
        y_pred = Bestmodelobj.basePredictModel(pred_df)
        return y_pred

    # historical_df is the same one returned from detections
    def Handling_calls(self, fill_na_dict, outliers_methods_input, imb_instruction, Norm_method,
                       lowvariance_actions, encoding_dict, reduce=None, auto_reduce=None, num_components_to_keep=None):

        if self.problem == "timeseries":
            frequency = calculate_date_frequency(self.historical_df[self.date_col])
            df.set_index('ds', inplace=True)
            train_data, test_data = train_test_split(self.historical_df, test_size=0.2, random_state=42)
        elif self.problem == "classification":
            train_data, test_data = train_test_split(self.historical_df, test_size=0.2, random_state=42,
                                                     stratify=self.historical_df[self.y_column])
            train_data = train_data.reset_index(drop=True)
            test_data = test_data.reset_index(drop=True)
        else:
            self.historical_df['timestamp'] = self.historical_df[self.date_col].astype('int64') / 10 ** 9
            self.historical_df['year'] = self.historical_df[self.date_col].dt.year
            self.historical_df['month'] = self.historical_df[self.date_col].dt.month
            self.historical_df['day'] = self.historical_df[self.date_col].dt.day
            self.historical_df.drop(self.date_col, axis=1, inplace=True)
            train_data, test_data = train_test_split(self.historical_df, test_size=0.2, random_state=42)
            train_data = train_data.reset_index(drop=True)
            test_data = test_data.reset_index(drop=True)

        train_data = self._process_data(train_data, fill_na_dict, outliers_methods_input, Norm_method)
        test_data = self._process_data(test_data, fill_na_dict, outliers_methods_input, Norm_method)

        # ----------------------------------------------------------------------------------------------------------
        # concat test and train to remake historical_df to pass to extract_and_search_features
        historical_df_copy = pd.concat([train_data, test_data])
        original_columns = set(self.historical_df.columns)

        # Remove co-linearity & low_variance
        historical_df_without_y = historical_df_copy.drop(columns=[self.y_column])
        historical_df_without_y = HandlingColinearity().handle_low_variance(historical_df_without_y,
                                                                            lowvariance_actions)
        historical_df_without_y = HandlingColinearity().handling_colinearity(historical_df_without_y)
        historical_df_without_y[self.y_column] = historical_df_copy[self.y_column]

        # Update historical_df_copy with processed columns
        historical_df_copy = historical_df_without_y.copy()

        # Identify removed columns
        processed_columns = set(historical_df_copy.columns)
        removed_columns = list(original_columns - processed_columns)

        # Remove identified columns from train_data
        train_data = train_data.drop(columns=removed_columns)
        test_data = test_data.drop(columns=removed_columns)
        if self.problem != "timeseries":
            # Extract features and perform similarity search
            meta_extractor, meta_features, best_models = self.extract_and_search_features(historical_df_copy)
        # ----------------------------------------------------------------------------------------------------------
        # resuming cleaning both df (because of extraction)
        # Encode categorical features and reduce dimensions if needed
        train_data = EncodeCategorical().Encode(train_data, encoding_dict)
        test_data = EncodeCategorical().Encode(test_data, encoding_dict)
        if imb_instruction:
            train_data = HandlingImbalanceClasses().handle_class_imbalance(train_data, self.y_column, imb_instruction)
            test_data = HandlingImbalanceClasses().handle_class_imbalance(test_data, self.y_column, imb_instruction)
        pca_model = None
        if reduce:
            handling_reduction = HandlingReduction()
            numerical_columns = train_data.drop(columns=[self.y_column]).select_dtypes(include=[np.number]).columns

            explained_variability, _ = handling_reduction.explainedVariability(train_data[numerical_columns])
            if auto_reduce:
                num_components_to_keep = handling_reduction.NumberOfComponents(train_data[numerical_columns],
                                                                               explained_variability)
            pca_model = PCA(n_components=num_components_to_keep).fit(train_data[numerical_columns])
            # Apply the fitted PCA model to both train and test data
            reduced_train_data = handling_reduction.feature_reduction(train_data[numerical_columns],
                                                                      num_components_to_keep, pca_model)
            reduced_train_data[self.y_column] = train_data[self.y_column]
            reduced_test_data = handling_reduction.feature_reduction(test_data[numerical_columns],
                                                                     num_components_to_keep, pca_model)
            reduced_test_data[self.y_column] = test_data[self.y_column]
            train_data = reduced_train_data
            test_data = reduced_test_data
        # --------------------------------------------------------------------------------------------
        # return cleand df
        historical_df_copy = pd.concat([train_data, test_data])
        historical_df_copy = historical_df_copy.sort_index()

        if self.problem != "timeseries":
            x_train = train_data.drop(columns=[self.y_column])
            y_train = train_data[self.y_column]

            # Split test_data into features and labels
            x_test = test_data.drop(columns=[self.y_column])
            y_test = test_data[self.y_column]

            print('best_models', best_models)
            Bestmodelobj = Bestmodel(ProblemType.CLASSIFICATION, best_models, x_train, x_test, y_train, y_test)
            Bestmodelobj.splitTestData()
            Bestmodelobj.TrainModel()
            print(Bestmodelobj.modelobj)
            # meta_features.append(Bestmodelobj.modelobj)
            # meta_extractor.addToKnowledgeBase(meta_features)

        else:
            x_train = "any"
            x_test = "any"
            print('best_models', best_models)
            Bestmodelobj = Bestmodel(ProblemType.CLASSIFICATION, best_models, x_train, x_test, train_data, test_data,
                                     frequency)
            Bestmodelobj.splitTestData()
            Bestmodelobj.TrainModel()
            print(Bestmodelobj.modelobj)
            # meta_features.append(Bestmodelobj.modelobj)
            # meta_extractor.addToKnowledgeBase(meta_features)

        return historical_df_copy, Bestmodelobj,num_components_to_keep, pca_model

    @staticmethod
    def featureReduction(df, Num_components_to_keep):
        numerical_columns = df.select_dtypes(include=['number']).columns
        numericalDf = df[numerical_columns]
        reducedFeatures = HandlingReduction().feature_reduction(numericalDf, Num_components_to_keep)
        non_numerical_columns = df.select_dtypes(exclude=['number']).columns
        non_numerical_df = df[non_numerical_columns].reset_index(drop=True)
        concatenated_features = pd.concat([non_numerical_df, reducedFeatures], axis=1)
        return concatenated_features


def user_interaction():
    # file_path = "similaritySearch/train.csv"
    y_column = 'Survived'
    df = pd.read_csv("train.csv")
    # Step 2: Call the Detections function to get detection results
    try:
        autoclean = AutoClean(df, y_column, "classification")
        df_copy, nulls_columns, outlier_columns, imbalance, low_variance_columns, categorical_columns = autoclean.Detections()

        print("Detection Results:")
        print(f"Nulls columns: {nulls_columns}")
        print(f"Outlier columns: {outlier_columns}")
        print(f"Imbalance: {imbalance}")
        print(f"Low Variance Columns: {low_variance_columns}")
        print(f"Categorical Columns: {categorical_columns}")

        # Handling missing values
        fill_na_dict = {}
        for col in nulls_columns:
            fill_na_dict[col] = 'auto'

        # Handling outliers
        outliers_method_input = ('z_score', 'auto', 3)
        if imbalance:
            imb_instruction = "auto"
        else:
            imb_instruction = None
        Norm_method = "auto"
        low_actions = {}
        encoding_dict = {}

        for col in categorical_columns:
            encoding_dict[col] = 'auto'
        for col in low_variance_columns:
            low_actions[col] = 'auto'

        reduce = 'True'
        auto_reduce = 'True'
        num_components_to_keep = 3
        processed_data, Bestmodelobj,num_components_to_keep, pca_model = autoclean.Handling_calls(fill_na_dict, outliers_method_input,
                                                                           imb_instruction, Norm_method,
                                                                           low_actions, encoding_dict, reduce,
                                                                           auto_reduce, num_components_to_keep)

        # y_pred = self.prediction_data(X_pred,fill_na_dict, outliers_methods_input, Norm_method, lowvariance_actions, encoding_dict,
        #                  imb_instruction, Bestmodelobj, num_components_to_keep,pca_model)

        print("Data preprocessing and handling completed successfully.")
        return processed_data

    except ValueError as ve:
        print(f"Error occurred: {ve}")
        return None


# Example usage:
if __name__ == "__main__":
    user_interaction()
