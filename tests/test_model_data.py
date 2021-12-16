import pytest
import pandas as pd
import os
from starter.ml.data import process_data

FP_CWD = os.getcwd()
FP_DATA = 'data/census.csv'


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
def data():
    df = pd.read_csv(os.path.join(FP_CWD, FP_DATA))
    return df


def test_cat_features(data, cat_features):
    """ Checks the categorical features are categorical. """
    assert set(cat_features).issubset(set(data.columns))


def test_cat_features_valid(data, cat_features):
    """ Check that none of the categorical variables are numeric """
    for variable in cat_features:
        print(data.loc[:, variable].dtypes)
        assert pd.api.types.is_numeric_dtype(data.loc[:, variable].dtypes) != True


@pytest.fixture
def x_train(data, cat_features):
    """  """
    df_x, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    return df_x


def test_data_size(data):
    """ Checks the size of the dataset is between 100 and 100k. """
    assert 100 < data.shape[0] < 100000


def test_data_shape(data):
    """ If your data is assumed to have no null values then this is a valid test. """
    assert data.shape == data.dropna().shape, "Dropping null changes shape."
