"""
Test the live prediction endpoint on Heroku
"""
import requests


features = {
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
}


app_url = "https://udacity-income-mle.herokuapp.com/inference"

r = requests.post(app_url, json=features)
print(r)
assert r.status_code == 200

print(f"Response body: {r.json()}")
