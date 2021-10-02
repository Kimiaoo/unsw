import submission
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

## Parameters settings
past_cases_interval = 10
past_weather_interval = 10

## Read training data
train_file = './data/COVID_train_data.csv'
train_df = pd.read_csv(train_file)

## Read Training labels
train_label_file = './data/COVID_train_labels.csv'
train_labels_df = pd.read_csv(train_label_file)

## Read testing Features
test_fea_file = './data/test_features.csv'
test_features = pd.read_csv(test_fea_file)

## Set hyper-parameters for the SVM Model
svm_model = SVR()
svm_model.set_params(**{'kernel': 'rbf', 'degree': 1, 'C': 5000,
                        'gamma': 'scale', 'coef0': 0.0, 'tol': 0.001, 'epsilon': 10})

## Generate Prediction Results
predicted_cases_part1 = []
for idx in range(len(test_features)):
    test_feature = test_features.loc[idx]
    prediction = submission.predict_COVID_part1(svm_model, train_df, train_labels_df,
                                                past_cases_interval, past_weather_interval, test_feature)
    predicted_cases_part1.append(prediction)

print(predicted_cases_part1)
