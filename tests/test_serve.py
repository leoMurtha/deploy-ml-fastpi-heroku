""" This module tests serve api for the ML Model """
from fastapi.testclient import TestClient
import sys

sys.path.append(".")  # noqa: E402
sys.path.append("./")  # noqa: E402

# Import our app from main.py.
from serve import app

# Instantiate the testing client with our app.
client = TestClient(app)


def test_get_root():
    """ Test the root page get a succesful response"""
    r = client.get("/")
    print(r)
    assert r.status_code == 200
    assert r.json() == {
        "200": "Greetings use post method with data to predict Income Class"}


def test_predict_high_income():
    """ An example when income greater than 50K """

    r = client.post("/inference", json={
        "age": 46,
        "workclass": "Private",
        "fnlgt": 400000,
        "education": "Some-college",
        "education_num": 10,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "Black",
        "sex": "Male",
        "capital_gain": 1000,
        "capital_loss": -100,
        "hours_per_week": 90,
        "native_country": "United-States"
    })

    assert r.status_code == 200
    assert r.json() == {"Income": ">50K"}


def test_predict_lower_income():
    """ Example when income is lower than 50K """
    r = client.post("/inference", json={
        "age": 27,
        "workclass": "Private",
        "fnlgt": 77516,
        "education": "Masters",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Tech-support",
        "relationship": "Unmarried",
        "race": "White",
        "sex": "Male",
        "capital_gain": 1600,
        "capital_loss": 0,
        "hours_per_week": 46,
        "native_country": "Cambodia"
    })

    assert r.status_code == 200
    assert r.json() == {"Income": "<=50K"}
