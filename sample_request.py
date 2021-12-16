"""
Flexible script to query the FastAPI
author: Ben E
date: 2021-12-16
"""
import requests
import json
import argparse
import sys

data = {
    "neg": {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    },
    "pos": {
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 45,
        "native-country": "United-States"
    }
}

# return the results of the api query
def query_api(argue):
    if argue.api == 'local':
        address = "http://127.0.0.1:8000/infer/"
    elif argue.api == 'remote':
        address = "https://umlops-task.herokuapp.com/infer/"
    else:
        print("Must specify local or remote option!")
        sys.exit()
    sample_data = data[argue.classified]
    r = requests.post(address, data=json.dumps(sample_data))
    print("status_code: ", r.status_code)
    print("Json response: ", r.json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('classified', default='neg',
                        help='Sample data either pos or neg.')
    parser.add_argument('api', default='local',
                        help='Specify local or remote api')
    args = parser.parse_args()
    query_api(args)
