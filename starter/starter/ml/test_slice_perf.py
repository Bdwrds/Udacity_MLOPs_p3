import pandas as pd
import pytest
import os
import joblib
from sklearn.model_selection import train_test_split
from .data import process_data
from .model import compute_model_metrics, inference, train_model

FP_CWD = os.getcwd()
FP_DATA = 'starter/data/census.csv'

@pytest.fixture
def data():
    df = pd.read_csv(os.path.join(FP_CWD, FP_DATA))
    return df

@pytest.fixture
def train_test_data(data):
    train, test = train_test_split(data, test_size=0.20, random_state=50)
    return train, test

@pytest.fixture
def cat_features():
    cat_feat = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    return cat_feat

@pytest.fixture
def y_values(train_test_data, cat_features):
    X_train, y_train, encoder, lb = process_data(
        train_test_data[0], categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, encoder, lb = process_data(
        train_test_data[1], categorical_features=cat_features, encoder=encoder, lb=lb, label="salary", training=False
    )
    FP_MODEL = "starter/model/random_forest.joblib"
    model = joblib.load(os.path.join(FP_CWD, FP_MODEL))
    predictions = inference(model, X_test)
    return predictions, y_test, X_test

def test_slice_inference(train_test_data, cat_features, y_values):
    """ Test to see if our mean per categorical slice is in the range 1.5 to 2.5."""
    test_data = train_test_data[1].reset_index(drop=True)
    for feature in cat_features:
        if feature == "workclass":
            slice_values = test_data[feature].value_counts().index
            for slice in slice_values:
                X_test = y_values[2]
                _idx = test_data.loc[test_data[feature] == slice].index
                print(len(_idx))
                predictions = y_values[0][_idx]
                y_test = y_values[1][_idx]

                # compute scores from predictions
                precision, recall, f1_score = compute_model_metrics(y_test, predictions)

                # print results from inference test
                print(f"{slice} value | Precision score: {precision}")
                print(f"{slice} value | Recall score: {recall}")
                print(f"{slice} value | F1 score: {f1_score}")
                assert (
                        f1_score > 0.5
                        ), f"For {slice}, slice inf score"

