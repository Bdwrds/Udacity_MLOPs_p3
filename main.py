"""
Main file for the FastAPI
author: Ben E
date: 2021-12-16
"""
import json
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from typing import Dict
from pickle import load
import os
from pydantic import BaseModel, Field
import pandas as pd
import importlib
import sys

# as specified in readme.md
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


sys.path.append(os.path.join(os.getcwd(), 'starter'))
data = importlib.import_module('ml.data')
model = importlib.import_module('ml.model')

process_data = data.process_data
inference = model.inference
# from bla.bla.ml.data import process_data
# from bla.bla.ml.model import inference

app = FastAPI()

FP_CWD = os.getcwd()
FP_MODEL = "model/random_forest.pkl"
FP_ENCODER = "model/encoder.pkl"
encoder = load(open(os.path.join(FP_CWD, FP_ENCODER), 'rb'))
model = load(open(os.path.join(FP_CWD, FP_MODEL), 'rb'))


class inferData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")


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


@app.get("/")
async def create_welcome():
    return {"Welcome to my model page. POST fields to /infer/ for running inference"}


@app.post("/infer/")
async def read_data(json_data: inferData):
    df = pd.DataFrame(jsonable_encoder(json_data), index=[0])
    XX, _, _, _ = process_data(
        df, categorical_features=cat_feat, encoder=encoder, training=False
    )
    preds = inference(model, XX)
    return json.dumps({"prediction": int(preds)})
