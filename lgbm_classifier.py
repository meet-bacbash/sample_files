# Import general requirements
import time

# Import model specific requirements
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from .knn_classifier import KnnClassifier


class LightGBMClassifier(KnnClassifier):

    def train_and_apply_model(self, x_train: np.ndarray, y_train: np.ndarray) -> LGBMClassifier:
        """

        This method will train the KNN classifier model

        :param x_train: 2-dimensional array with training records
        :param y_train: 1-dimensional array (labels)
        :return: _model:KNeighborsClassifier Trained Knn classifier model with input data
        """

        _model = LGBMClassifier(**self.params)
        _model.fit(x_train, y_train)
        return _model


# Testing in local env ony
def run(df, targeted_column_name, selected_features=None, params: dict = {}):
    if selected_features is None:
        selected_features = []
    lgbm_obj = LightGBMClassifier(
        input_dataframe=df,
        targeted_column_name=targeted_column_name,
        selected_features=selected_features,
        params=params
    )
    results = lgbm_obj.transform()
    return results


if __name__ == "__main__":

    selected_columns = ['Assignment group', 'State', 'Duration','Close code', 'Impacted Country']
    df = pd.read_excel('/home/bacancy/Downloads/All_Tickets_2023_With_MCAR.xlsx')
    output_business_data = run(df, "Priority", selected_columns, {})
    print(output_business_data.get("output_dataframe"))
    output_business_data.get("output_dataframe").to_csv("/home/bacancy/Downloads/imputed_lgbm.csv")
