# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import joblib
import os
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data
# Add code to load in the data.
FP_CWD = os.getcwd()
FP_DATA = 'starter/data/census.csv'
data = pd.read_csv(os.path.join(FP_CWD, FP_DATA))


# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False
)

# Train and save a model.
model = train_model(X_train, y_train)

# save model in existing dir
FP_MODEL = "starter/model/random_forest.joblib"
joblib.dump(model, os.path.join(FP_CWD, FP_MODEL))

# run inference on the latest data
predictions = inference(model, X_test)

# compute scores from predictions
precision, recall, f1_score = compute_model_metrics(y_test, predictions)

# print results from inference test
print(f"Precision score: {precision}")
print(f"Recall score: {recall}")
print(f"F1 score: {f1_score}")
