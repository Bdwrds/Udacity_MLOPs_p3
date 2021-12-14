import json

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from typing import Optional
from pickle import load
import os
from pydantic import BaseModel, Field
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from starter.starter.ml.data import process_data
from starter.starter.ml.model import inference

app = FastAPI()

FP_CWD = os.getcwd()
FP_MODEL = "starter/model/random_forest.pkl"
FP_ENCODER = "starter/model/encoder.pkl"
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
async def read_data(data: inferData):
    df = pd.DataFrame(jsonable_encoder(data), index=[0])
    XX, _, _, _ = process_data(
        df, categorical_features=cat_feat, encoder=encoder, training=False
    )
    preds = inference(model, XX)
    return json.dumps({"prediction": int(preds)})

@app.post("/items/")
async def create_item(item: int):
    item_dict = item.dict()
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict