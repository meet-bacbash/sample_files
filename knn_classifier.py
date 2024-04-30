# Import general requirements
import time
import warnings
# from icecream import ic
from logger import logex
# from ....Generics.generics import print_execution_time

# Import model specific requirements
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # Enable iterative imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_row', 10)
pd.set_option('display.width', 10)


class KnnClassifier:
    def __init__(self,
                 input_dataframe: pd.DataFrame,
                 targeted_column_name: str,
                 selected_features=None,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 params=None
                 ):
        """

        :param input_dataframe: pandas dataframe
        :param targeted_column_name: name of column which is targeted to fill missing values
        :param selected_features: list of column names to be considered for filing target column
        :param test_size: float value to use in train_test_split
        :param random_state: float value to use in train_test_split
        :param params: advance parameters according selected models
        """
        if params is None:
            params = {}
        if selected_features is None:
            selected_features = input_dataframe.columns.values.tolist()
            selected_features.remove(targeted_column_name)

        # ic(input_dataframe.shape)
        logex.info(f"Input Dataframe:{input_dataframe.shape}")

        self.targeted_column_name = targeted_column_name
        self.selected_features = selected_features
        self.input_dataframe2 = input_dataframe

        # impute all the possible columns for better accuracy except targeted column.
        self.input_dataframe = self.label_encoding(self.input_dataframe2)
        self.input_dataframe = self.impute_empty_columns(self.input_dataframe[selected_features])
        self.input_dataframe[targeted_column_name] = self.input_dataframe2[targeted_column_name]

        self.updated_dataframe = input_dataframe
        self.test_size = test_size
        self.random_state = random_state
        self.params = params
        self.empty_target_records = None

    def get_empty_target_records(self) -> pd.DataFrame:
        """
        :return: dataframe with null records for targeted column,
        also keeping new dataframe in self for productions space
        """
        self.input_dataframe[self.targeted_column_name].replace('', None, inplace=True)
        self.empty_target_records = self.input_dataframe[self.input_dataframe[self.targeted_column_name].isnull()]
        # ic(self.empty_target_records.shape)
        logex.info(f"Empty Target Records Shape:{self.empty_target_records.shape}")
        print(self.empty_target_records.head())
        return self.empty_target_records
    
    # @print_execution_time
    def preprocess_data(self) -> pd.DataFrame:
        """
        :return: updated_df: Updated dataframe after applying operations
        """

        # drop nulls from the subset of relevant columns
        relevant_columns = self.selected_features
        self.updated_dataframe = (self.input_dataframe[relevant_columns + [self.targeted_column_name]].
                                  dropna(subset=relevant_columns))

        # Label encoding for categorical columns
        self.updated_dataframe = self.label_encoding(self.updated_dataframe)
        return self.updated_dataframe

    def label_encoding(self, df_encode) -> pd.DataFrame:
        """
        Encodes categorical and date/datetime columns into int
        :param df_encode: dataframe to encode (input dataframe in most cases)
        :return: dataframe with encoded columns
        """
        label_encoders = {}
        for column in df_encode.drop(self.targeted_column_name, axis=1).select_dtypes(
                include=['object', 'datetime64[ns]']).columns:
            label_encoders[column] = LabelEncoder()
            df_encode[column] = label_encoders[column].fit_transform(
                df_encode[column].astype(str))
        return df_encode

    def impute_empty_columns(self, predict_data) -> pd.DataFrame:
        """
        Imputes missing values in all columns except the target column.
        :param predict_data: dataframe
        :return: dataframe with imputed records
        """

        target_column = self.targeted_column_name  # Replace with the actual target column name

        # Select columns for imputation (excluding the target column)
        impute_columns = predict_data.columns[predict_data.columns != target_column]

        # Choose an imputation method
        imputation_method = IterativeImputer(max_iter=10, random_state=0)  # Iterative imputer

        # Impute missing values into training dataset
        predict_data[impute_columns] = imputation_method.fit_transform(predict_data[impute_columns])

        return predict_data

    def get_train_test_and_predict_data(self) -> [pd.DataFrame, pd.DataFrame]:
        """
        This method will divide entire dataframe into
        predict_dataframe: all the target column cells are null
        train_test_dataframe: all the target column cells are having values
        :return: train_test_data, predict_data
        """
        df = self.updated_dataframe
        logex.info(f"self.selected_features : {self.selected_features} \n self.targeted_column_name : {self.targeted_column_name}")
        predict_data = self.empty_target_records[self.selected_features + [self.targeted_column_name]]
        # predict_data = self.label_encoding(predict_data)

        # predict_data = self.impute_empty_columns(predict_data)
        train_test_data = df[df[self.targeted_column_name].notnull()]

        return train_test_data, predict_data

    # @print_execution_time
    def train_and_apply_model(self, x_train, y_train) -> KNeighborsClassifier:
        """

        This method will train the KNN classifier model

        :param x_train: 2-dimensional array with training records
        :param y_train: 1-dimensional array (labels)
        :return: _model:KNeighborsClassifier Trained Knn classifier model with input data
        """

        _model = KNeighborsClassifier(**self.params)
        _model.fit(x_train, y_train)
        return _model

    @staticmethod
    def calculate_accuracy(y_test, test_predictions) -> float:
        """
        Calculate accuracy score by comparing real and predicted labels and return computed subset accuracy
        :param y_test: real labels
        :param test_predictions: predicted labels
        :return: float value (accuracy score)
        """
        accuracy_ = accuracy_score(y_test, test_predictions)
        accuracy_ = round(accuracy_, 2)
        return accuracy_

    @staticmethod
    def generate_confusion_matrix(y_test, test_predictions) -> np.ndarray:
        """
        Calculate confusion matrix: 2-dimensional np.ndarray

        :param y_test: real labels
        :param test_predictions: predicted labels
        :return: confusion matrix
        """
        matrix_ = confusion_matrix(y_test, test_predictions)
        return matrix_

    # @print_execution_time
    def calculate_prediction_results(self, y_test, test_predictions) -> dict:
        """
        calculate accuracy score and generate confusion matrix, wrapper for calculating accuracy and confusion matrix

        :param y_test: real label array
        :param test_predictions: predicted labels
        :return: dict [accuracy_score, confusion_matrix]
        """
        accuracy_score_ = self.__class__.calculate_accuracy(
            y_test,
            test_predictions
        )
        # ic(accuracy_score_)
        logex.info(f"Accuracy Score:{accuracy_score_}")

        confusion_matrix_ = self.__class__.generate_confusion_matrix(
            y_test,
            test_predictions
        )
        return {
            "accuracy_score": accuracy_score_,
            "confusion_matrix": confusion_matrix_
        }

    def prepare_final_dataframe(self, predicted_column: pd.DataFrame) -> pd.DataFrame:
        """
        :param predicted_column: dataframe with only single column (targeted_column)
        :return: final_dataframe: dataframe with shape of input_dataframe
        """
        final_dataframe: pd.DataFrame = pd.concat(
            [self.empty_target_records.drop(self.targeted_column_name, axis=1).reset_index(drop=True),
             predicted_column.reset_index(drop=True)], axis=1)

        df2 = self.input_dataframe[
            self.input_dataframe[self.targeted_column_name].notnull() & self.input_dataframe[
                self.targeted_column_name].ne("")]

        final_dataframe = final_dataframe.append(df2)

        return final_dataframe

    def transform(self):
        """
        Run all the operation in order:
        -separate null target records
        -data preprocessing
        -split train, test, predict data
        -train the model
        -calculate accuracy
        -prepare final output dataframe

        :return: output: dict
        """

        logex.info(f"[{self.targeted_column_name}] {self.__class__.__name__}: Preprocessing data.")
        empty_targets_records = self.get_empty_target_records()
        self.preprocess_data()

        # Split data into train, test, to be predicted
        logex.info(f"[{self.targeted_column_name}] {self.__class__.__name__}: Splitting data.")
        test_train_df, need_to_predict_df = self.get_train_test_and_predict_data()

        x = test_train_df.drop(self.targeted_column_name, axis=1)
        y = test_train_df[self.targeted_column_name]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size,
                                                            random_state=self.random_state)
        # Train & make predictions
        logex.info(f"[{self.targeted_column_name}] {self.__class__.__name__}: Training model.")
        _knn = self.train_and_apply_model(x_train, y_train)
        test_predictions = _knn.predict(x_test)

        # Calculate accuracy score & confusion matrix
        logex.info(f"[{self.targeted_column_name}] {self.__class__.__name__}: Calculating accuracy.")
        model_result = self.calculate_prediction_results(y_test, test_predictions)

        # Make predications for actual missing values for targeted data
        logex.info(f"[{self.targeted_column_name}] {self.__class__.__name__}: Making actual predictions.")
        actual_predictions = _knn.predict(need_to_predict_df.drop(self.targeted_column_name, axis=1))
        predicted_df = pd.DataFrame(actual_predictions, columns=[self.targeted_column_name])

        # Prepare final dataframe, join predicted cells into empty cells.
        logex.info(f"[{self.targeted_column_name}] {self.__class__.__name__}: Preparing final dataframe.")
        final_dataframe = self.prepare_final_dataframe(predicted_df)

        # ic(predicted_df.shape)
        logex.info(f"Predicted DF shape:{predicted_df.shape}")
        # ic(final_dataframe.shape)
        logex.info(f"Final dataframe shape:{final_dataframe.shape}")
        print(f"[{self.targeted_column_name}] {self.__class__.__name__}: Finish.")
        return {
            "output_dataframe": final_dataframe,
            **model_result
        }


# Testing for local env only
def run(df, targeted_column_name, selected_features=None, params=None):
    if params is None:
        params = {}
    if selected_features is None:
        selected_features = []
    knn_obj = KnnClassifier(
        input_dataframe=df,
        targeted_column_name=targeted_column_name,
        selected_features=selected_features,
        params=params
    )
    results = knn_obj.transform()
    return results


if __name__ == "__main__":
    # Testing for local only
    params = {
        "n_neighbors": 5
    }
    input_file = ""

    selected_columns = ['Assignment group', 'State', 'Duration','Close code', 'Impacted Country']
    df = pd.read_excel(input_file)
    output_business_data = run(df, "Priority", selected_columns, params)
    print(output_business_data.get("output_dataframe"))
    output_business_data.get("output_dataframe").to_csv("/home/bacancy/Downloads/imputed_knn_classifier.csv")
