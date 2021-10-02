import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, accuracy_score
import math


# reshape train_df to match test_df
def reshape_df(x_train_df, cols):
    train_set = pd.DataFrame(columns=cols)
    for col in cols:
        link_loc = col.find('-')
        if col == 'day':
            train_set[col] = x_train_df[col].values.tolist()[30:]
        else:
            day = int(col[link_loc + 1:])
            train_set[col] = x_train_df[col[:link_loc]].values.tolist()[30 - day:len(x_train_df) - day]
    return train_set


## Project-Part1
def predict_COVID_part1(svm_model, train_df, train_labels_df, past_cases_interval, past_weather_interval, test_feature):
    train_cols = test_feature.index.tolist()
    train_set = reshape_df(train_df, train_cols)

    x_train = pd.DataFrame()
    x_test = pd.Series()

    # get past_weather_interval, past_cases_interval of information
    for col in train_cols:
        if col == "day":
            x_train[col] = train_set[col]
            x_test[col] = test_feature[col]
        else:
            link_loc = col.find('-')
            day = int(col[link_loc + 1:])
            if col[:link_loc] == "max_temp" or col[:link_loc] == "max_dew" or col[:link_loc] == "max_humid":
                if day <= past_weather_interval:
                    x_train[col] = train_set[col]
                    x_test[col] = test_feature[col]
            elif col[:link_loc] == "dailly_cases":
                if day <= past_cases_interval:
                    x_train[col] = train_set[col]
                    x_test[col] = test_feature[col]

    x_train.set_index("day", inplace=True)

    # train model
    x_train = np.array(x_train)
    y_train = np.array(train_labels_df.loc[30:, :].set_index("day"))
    svm_model.fit(x_train, y_train.ravel())

    return math.floor(svm_model.predict([np.array(x_test[1:])]))


## Project-Part2
def predict_COVID_part2(train_df, train_labels_df, test_feature):
    past_cases_interval = 16

    train_cols = test_feature.index.tolist()
    train_set = reshape_df(train_df, train_cols)

    x_train = pd.DataFrame()
    x_test = pd.Series()

    # get past_weather_interval, past_cases_interval of information
    for col in train_cols:
        if col == "day":
            x_train[col] = train_set[col]
            x_test[col] = test_feature[col]
        else:
            link_loc = col.find('-')
            day = int(col[link_loc + 1:])
            if col[:link_loc] == "dailly_cases":
                if day <= past_cases_interval:
                    x_train[col] = train_set[col]
                    x_test[col] = test_feature[col]
    x_train.set_index("day", inplace=True)

    # train model
    x_train = np.array(x_train[60:])
    y_train = np.array(train_labels_df.loc[90:, :].set_index("day"))

    # set hyper-parameters for the SVM Model
    svm_model = SVR()
    svm_model.set_params(**{'kernel': 'poly', 'degree': 1, 'C': 8500,
                            'gamma': 'scale', 'coef0': 0.0, 'tol': 0.001, 'epsilon': 10})
    svm_model.fit(x_train, y_train.ravel())

    return math.floor(svm_model.predict([np.array(x_test[1:])]))
