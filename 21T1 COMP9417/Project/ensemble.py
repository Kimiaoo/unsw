import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

# ----------------------------------------------------------------------------------------------------------------------

# [8346 rows x 128 columns]
raw_x_train = pd.read_csv('./data/X_train.csv', header=None)
# [8346 rows x 1 columns] 1.0 2.0 3.0 4.0 5.0 6.0
raw_y_train = pd.read_csv('./data/y_train.csv', header=None)
y_train = raw_y_train.values.reshape(len(raw_y_train), )

# [2782 rows x 128 columns]
raw_x_val = pd.read_csv('./data/X_val.csv', header=None)
raw_y_val = pd.read_csv('./data/y_val.csv', header=None)
y_val = raw_y_val.values.reshape(len(raw_y_val), )

# [2782 rows x 128 columns]
raw_x_test = pd.read_csv('./data/X_test.csv', header=None)

# ----------------------------------------------------------------------------------------------------------------------

# preprocessing
scaler = StandardScaler()
x_train = scaler.fit_transform(raw_x_train)
x_val = scaler.transform(raw_x_val)
x_test = scaler.transform(raw_x_test)

# delete outlier
x_train = pd.DataFrame(x_train)
y_train = pd.DataFrame(y_train)
x_train = x_train[(x_train.index != 8202) & (x_train.index != 6911)
                  & (x_train.index != 2900) & (x_train.index != 7814) & (x_train.index != 3101)
                  & (x_train.index != 7685) & (x_train.index != 288)]
y_train = y_train[(y_train.index != 8202) & (y_train.index != 6911)
                  & (y_train.index != 2900) & (y_train.index != 7814) & (y_train.index != 3101)
                  & (y_train.index != 7685) & (y_train.index != 288)]
x_train = np.array(x_train)
y_train = np.array(y_train).reshape(len(y_train), )

# PCA
pca = PCA(n_components=0.99)
x_train = pca.fit_transform(x_train)
x_val = pca.transform(x_val)
x_test = pca.transform(x_test)

# Base Model
# knn: 0.9931, 19/2782
knn = BaggingClassifier(KNeighborsClassifier(n_neighbors=1), random_state=5)

# rf: 0.9928, 20/2782
rf = RandomForestClassifier(n_estimators=150, max_features=3, random_state=0)

# dt: 0.9924, 21/2782
dt = BaggingClassifier(DecisionTreeClassifier(max_depth=26, splitter='random', min_samples_leaf=1,
                                              min_impurity_decrease=0.0, criterion='entropy', random_state=42),
                       random_state=85)

# svc: 0.9910, 25/2782
svc = SVC(kernel='rbf', C=100, probability=True, random_state=0, decision_function_shape='ovo')

# ----- Ensemble: Major Voting -----
estimators = [('knn', knn), ('rf', rf), ('dt', dt), ('svc', svc)]
ensemble = VotingClassifier(estimators, voting='soft', weights=[1, 1, 1, 1])

ensemble.fit(x_train, y_train)
predict_train = ensemble.predict(x_train)
predict_val = ensemble.predict(x_val)

# Base Model evaluate
# knn / rf / dt / knn5 / svc
# model = svc
# model.fit(x_train, y_train)
# predict_train = model.predict(x_train)
# predict_val = model.predict(x_val)

# ----- Weighted F1 Score -----
f1_train = f1_score(y_train, predict_train, average='weighted')
print('train set f1_score', f1_train)
f1_val_weighted = f1_score(y_val, predict_val, average='weighted')
print('val set f1_val_weighted', f1_val_weighted)

# precision_val = precision_score(y_val, predict_val, average=None)
# print('val set precision_val', np.mean(precision_val))
# recall_val = recall_score(y_val, predict_val, average=None)
# print('val set recall_val', np.mean(recall_val))
# f1_val_micro = f1_score(y_val, predict_val, average='micro')
# print('val set f1_val_micro', f1_val_micro)
# f1_val_macro = f1_score(y_val, predict_val, average='macro')
# print('val set f1_val_macro', f1_val_macro)

count = 0
for (a, b) in zip(predict_train, y_train):
    if a != b:
        # print(a, b)
        count += 1
print(f'train set mis-classify: {count}/{len(y_train)}')

count = 0
for (a, b) in zip(predict_val, y_val):
    if a != b:
        # print(a, b)
        count += 1
print(f'val set mis-classify: {count}/{len(y_val)}')

# ----------------------------------------------------------------------------------------------------------------------

# ----- submission -----
predict_test = ensemble.predict(x_test)
submission = pd.DataFrame(predict_test, dtype='float64')
submission.to_csv('y_test.csv', index=False, header=None)
