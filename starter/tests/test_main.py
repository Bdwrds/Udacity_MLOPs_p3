import json
from fastapi.testclient import TestClient
import importlib
import os

#from starter.main import app
#from starter.sample_request import data, data2
print(os.getcwd())
import sys
sys.path.append(os.path.join(os.getcwd()))#, 'starter'))
print("sys.paths", sys.path)
main = importlib.import_module('main')
sample_request = importlib.import_module('sample_request')

client = TestClient(main.app)

# verify the message response and status code
def test_get_local_root():
    r = client.get('/')
    assert r.status_code == 200
    assert r.json() == ["Welcome to my model page. POST fields to /infer/ for running inference"]

# verify the results from sample 1 - <=$50k/ 0
def test_post_local_pred_neg():
    r = client.post('/infer/', data=json.dumps(sample_request.data))
    assert r.status_code == 200
    assert r.json() == '{"prediction": 0}'

# verify the results from sample 2 - >$50k/ 1
def test_post_local_pred_pos():
    r = client.post('/infer/', data=json.dumps(sample_request.data2))
    assert r.status_code == 200
    assert r.json() == '{"prediction": 1}'