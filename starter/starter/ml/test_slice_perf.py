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
    """ Test to see if our mean f1 score per categorical per slice is greater than 0.6. Training is ~0.68"""
    test_data = train_test_data[1].reset_index(drop=True)
    cols = ['feature', 'slice', 'instances', 'precision', 'recall', 'f1']
    df_perf = pd.DataFrame(columns=cols)
    for feature in cat_features:
        df_sliced_perf = pd.DataFrame(columns = cols)
        slice_values = test_data[feature].value_counts().index
        for slice in slice_values:
            _idx = test_data.loc[test_data[feature] == slice].index
            predictions = y_values[0][_idx]
            y_test = y_values[1][_idx]
            slice_len = len(_idx)
            if slice_len < 25:
                continue
            precision, recall, f1_score = compute_model_metrics(y_test, predictions)
            slice_performance = pd.DataFrame([[feature, slice, slice_len, precision, recall, f1_score]], columns = cols)
            df_sliced_perf = df_sliced_perf.append(slice_performance, ignore_index=True)
        df_perf = df_perf.append(df_sliced_perf)
    assert (df_perf.f1.mean() > 0.6), f"For {slice}, slice inf score"
