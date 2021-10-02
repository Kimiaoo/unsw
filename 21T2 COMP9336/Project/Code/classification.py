import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# set label and obtain formatted data
def format_data(df, la):
    df["Time"] = df["Time"].map(lambda x: int(x))
    time = df["Time"].values.tolist()
    rss = df["RSS"].values.tolist()

    interval = 5
    temp, ds_rss = [], []
    for j in range(len(time)):
        if time[j] < interval and len(temp) < 45:
            temp.append(rss[j])
        else:
            ds_rss.append(temp)
            interval += 5
            temp = []

    ds = pd.DataFrame({"RSS": ds_rss, "label": [la] * len(ds_rss)})
    return ds


# read gesture data
df1 = pd.read_excel("../Dataset/pull-push.xlsx")
df2 = pd.read_excel("../Dataset/up-down.xlsx")
df3 = pd.read_excel("../Dataset/wave.xlsx")

# set labels
df1_ds = format_data(df1, "pull-push")
df2_ds = format_data(df2, "up-down")
df3_ds = format_data(df3, "wave")

# concat three excel file to a dataset
dataset = pd.concat([df1_ds, df2_ds, df3_ds], axis=0, ignore_index=True)

# obtain x: rss_list and y: label_list
rss_list = dataset["RSS"].values.tolist()
label_list = dataset["label"].values.tolist()

# Normalization
rss_list = MinMaxScaler().fit_transform(rss_list)

# Standardized data
rss_list = StandardScaler().fit(rss_list).transform(rss_list)

# split train set and test set
x_train, x_test, y_train, y_test = train_test_split(rss_list, label_list, test_size=0.2, shuffle=True, random_state=2)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# ########################### GridSearchCV ###########################
# k_fold = KFold(n_splits=10)
# knn = GridSearchCV(KNeighborsClassifier(), param_grid={'leaf_size': list([15, 30]), 'n_neighbors': list([3, 10])},
#                    cv=k_fold, scoring="accuracy")
# knn.fit(x_train, y_train)
# print(knn.best_estimator_)
# y_pred = knn.predict(x_test)
# #####################################################################

# fit data
knn_classification = KNeighborsClassifier(leaf_size=15, n_neighbors=3)
knn_classification.fit(x_train, y_train)

# prediction
y_pred = knn_classification.predict(x_test)

# performance
con_matrix = confusion_matrix(y_test, y_pred, labels=["pull-push", "up-down", "wave"])
accuracy = accuracy_score(y_test, y_pred)

print("confusion matrix: ", con_matrix)
print("accuracy: ", accuracy)
