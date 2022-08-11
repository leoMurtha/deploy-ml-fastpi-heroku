"""
This module includes unit tests for the ML model
"""
import pytest
from src.ml import utils
from src.ml.model import inference
from src.ml.data import cat_features, process_data, load_data


DATA_PATH = 'data/census_no_spaces.csv'
MODEL_PATH = 'model/model.pkl'
ENCODER_PATH = 'model/encoder.pkl'
LB_PATH = 'model/lb.pkl'


@pytest.fixture(name='data')
def data():
    """
    Fixture data will be used by the unit tests.
    """
    yield load_data(DATA_PATH)


def test_load_data(data):
    """ Test the data from load_data """
    assert data.shape[0] > 0
    assert data.shape[1] > 0


def test_model_predict(data):
    """ Test model predict function """

    model = utils.get_object(MODEL_PATH)
    encoder = utils.get_object(ENCODER_PATH)
    lb = utils.get_object(LB_PATH)

    X, _, _, _ = process_data(
        data.head(10), categorical_features=cat_features,
        label="salary", training=False, encoder=encoder, lb=lb
    )

    pred = inference(model, X)[0]

    assert pred is not None


def test_process_data(data):
    """ Test process_data function, if returns the same len for x and y """

    X, y, _, _ = process_data(data.head(100), cat_features, label='salary')
    assert len(X) == len(y)
