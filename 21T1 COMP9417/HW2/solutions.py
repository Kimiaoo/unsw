### Question 1

# (a)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)

import time
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

from sklearn.metrics import accuracy_score, roc_auc_score


def create_dataset(n=1250, nf=2, nr=0, ni=2, random_state=125):
    '''
    generate a new dataset with 
    n: total number of samples
    nf: number of features
    nr: number of redundant features (these are linear combinatins of informative features)
    ni: number of informative features (ni + nr = nf must hold)
    random_state: set for reproducibility
    '''
    X, y = make_classification(n_samples=n,
                               n_features=nf,
                               n_redundant=nr,
                               n_informative=ni,
                               random_state=random_state,
                               n_clusters_per_class=2)
    rng = np.random.RandomState(2)
    X += 3 * rng.uniform(size=X.shape)
    X = StandardScaler().fit_transform(X)

    return X, y


# (b)

def plotter(classifier, X, X_test, y_test, title, ax=None):
    # plot decision boundary for given classifier
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    if ax:
        ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
        ax.set_title(title)
    else:
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
        plt.title(title)


# create classifier
def create_clf(cls, x_train, y_train):
    if cls == 0:
        return DecisionTreeClassifier().fit(x_train, y_train)
    elif cls == 1:
        return RandomForestClassifier().fit(x_train, y_train)
    elif cls == 2:
        return AdaBoostClassifier().fit(x_train, y_train)
    elif cls == 3:
        return LogisticRegression().fit(x_train, y_train)
    elif cls == 4:
        return MLPClassifier().fit(x_train, y_train)
    elif cls == 5:
        return SVC().fit(x_train, y_train)


# Q1 (a): Plot the decision boundaries of each of the classifiers
def Question1_a(X, x_train, x_test, y_train, y_test, labels):
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    for i, ax in enumerate(ax.flat):
        plotter(create_clf(i, x_train, y_train), X, x_test, y_test, labels[i], ax)
    plt.show()


# Q1 (b): study the performance of each of classifiers varies as increasing the size of training set
def Question1_b(x_train, x_test, y_train, y_test, labels):
    train_sizes = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    clf_acc = {}
    for i in range(6):
        for size in train_sizes:
            temp = []
            for j in range(10):
                if size != 1000:
                    xx_train, _, yy_train, _ = train_test_split(x_train, y_train, test_size=(1000 - size))
                else:
                    xx_train, yy_train = x_train, y_train
                clf = create_clf(i, xx_train, yy_train)
                y_pred = clf.predict(x_test)
                acc = accuracy_score(y_test, y_pred)
                temp.append(acc)
            if labels[i] not in clf_acc:
                clf_acc[labels[i]] = [np.mean(temp)]
            else:
                clf_acc[labels[i]].append(np.mean(temp))
    # draw the plot
    clf_legend = {'Decision Tree': 'blue', 'Random Forest': 'orange', 'AdaBoost': 'green',
                  'Logistic Regression': 'red', 'Neural Network': 'purple', 'SVM': 'tan'}

    plt.figure(figsize=(10, 8))
    plt.xlabel('Size of training set')
    plt.ylabel('Average accuracy')

    for clf in clf_acc:
        y_val = clf_acc[clf]
        plt.plot(train_sizes, y_val, label=clf, c=clf_legend[clf])
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()


# Q1 (c): the training time for each of the classifiers at each of the training set sizes
def Question1_c(x_train, x_test, y_train, y_test, labels):
    train_sizes = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    clf_time = {}
    for i in range(6):
        for size in train_sizes:
            temp = []
            for j in range(10):
                if size != 1000:
                    xx_train, _, yy_train, _ = train_test_split(x_train, y_train, test_size=(1000 - size))
                else:
                    xx_train, yy_train = x_train, y_train
                t1 = time.time()
                create_clf(i, xx_train, yy_train)
                t2 = time.time()
                temp.append(t2 - t1)
            if labels[i] not in clf_time:
                clf_time[labels[i]] = [np.log(np.mean(temp))]
            else:
                clf_time[labels[i]].append(np.log(np.mean(temp)))

    # draw the plot
    clf_legend = {'Decision Tree': 'blue', 'Random Forest': 'orange', 'AdaBoost': 'green',
                  'Logistic Regression': 'red', 'Neural Network': 'purple', 'SVM': 'tan'}

    plt.figure(figsize=(10, 8))
    plt.xlabel('Size of training set')
    plt.ylabel('Average training time for each of the classifiers')

    for clf in clf_time:
        y_val = clf_time[clf]
        plt.plot(train_sizes, y_val, label=clf, c=clf_legend[clf])
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()


# Q1 (d): calculate the accuracy on train set and test set
def Question1_d(x_train, x_test, y_train, y_test):
    DT_classifier = DecisionTreeClassifier(random_state=0).fit(x_train, y_train)

    y_train_pred = DT_classifier.predict(x_train)
    train_acc = accuracy_score(y_train, y_train_pred)

    y_test_pred = DT_classifier.predict(x_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    print("-------------------- Q1 (d) --------------------")
    print(f"Train accuracy: {train_acc}\nTest accuracy: {test_acc}")


# Q1 (e): draw plot of AUC score
def Question1_e(x_train, x_test, y_train, y_test):
    train_auc, test_auc = [], []
    for k in range(2, 131):
        DT_classifier = DecisionTreeClassifier(min_samples_leaf=k, random_state=0).fit(x_train, y_train)

        # calculate AUC score
        y_train_pred = DT_classifier.predict_proba(x_train)
        train_auc.append(roc_auc_score(y_train, y_train_pred[:, 1]))

        y_test_pred = DT_classifier.predict_proba(x_test)
        test_auc.append(roc_auc_score(y_test, y_test_pred[:, 1]))

    plt.figure(figsize=(10, 10))
    plt.xlabel('min samples leaf')
    plt.ylabel('AUC score')

    auc_score = {'train_auc': train_auc, 'test_auc': test_auc}

    for st in auc_score:
        y_val = auc_score[st]
        plt.plot(range(2, 131), y_val, label=st)
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()


# Q1 (f): perform cross validation over the min samples leaf hyperparameter from scratch
def Question1_f(x_train, x_test, y_train, y_test):
    cv_score = []
    min_leaf, highest_cv_score = 2, 0
    for k in range(2, 96):
        auc_score = []
        for i in range(10):
            # split train and test dataset
            x_cv = x_train[i * 100:(i + 1) * 100]
            xx_train = np.concatenate((x_train[:i * 100], x_train[(i + 1) * 100:]))
            y_cv = y_train[i * 100:(i + 1) * 100]
            yy_train = np.concatenate((y_train[:i * 100], y_train[(i + 1) * 100:]))

            # fit model
            DT_classifier = DecisionTreeClassifier(min_samples_leaf=k, random_state=0)
            DT_classifier.fit(xx_train, yy_train)
            # calculate auc score
            y_pred = DT_classifier.predict_proba(x_cv)
            auc_score.append(roc_auc_score(y_cv, y_pred[:, 1]))
        cv_score.append(auc_score)
        if np.mean(auc_score) > highest_cv_score:
            highest_cv_score = np.mean(auc_score)
            min_leaf = k
    # draw the boxplot
    plt.figure(figsize=(20, 10))
    plt.xlabel('min samples leaf')
    plt.ylabel('CV score')
    plt.boxplot(cv_score, labels=range(2, 96))
    plt.tight_layout()
    plt.show()
    # make the best model and calculate the accuracy on x_train and x_test
    DT_classifier = DecisionTreeClassifier(min_samples_leaf=min_leaf, random_state=0).fit(x_train, y_train)
    y_train_pred = DT_classifier.predict(x_train)
    train_acc = accuracy_score(y_train, y_train_pred)

    y_test_pred = DT_classifier.predict(x_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print("-------------------- Q1 (f) --------------------")
    print(f"min_samples_leaf: {min_leaf}\nTrain accuracy: {train_acc}\nTest accuracy: {test_acc}")


# Q1 (g): find the optimal min samples leaf chosen by GridSearchCV
def Question1_g(x_train, x_test, y_train, y_test):
    leaf_list = list(range(2, 96))
    min_leaf_dic = {'min_samples_leaf': leaf_list}
    DT_classifier = DecisionTreeClassifier(random_state=0)
    grid_search_cv = GridSearchCV(estimator=DT_classifier, param_grid=min_leaf_dic, scoring='roc_auc', cv=10)
    grid_search_cv.fit(x_train, y_train)
    best_min_leaf = leaf_list[grid_search_cv.best_index_]

    DT_classifier = DecisionTreeClassifier(min_samples_leaf=best_min_leaf, random_state=0).fit(x_train, y_train)
    y_train_pred = DT_classifier.predict(x_train)
    train_acc = accuracy_score(y_train, y_train_pred)

    y_test_pred = DT_classifier.predict(x_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print("-------------------- Q1 (g) --------------------")
    print(f"min_samples_leaf: {best_min_leaf}\nTrain accuracy: {train_acc}\nTest accuracy: {test_acc}")


# plot the data
def Question2_a(x1, x2, yval):
    pos_idx, neg_idx = [], []
    for i in range(len(yval)):
        if yval[i] == 1:
            pos_idx.append(i)
        else:
            neg_idx.append(i)

    pos_x1, neg_x1, pos_x2, neg_x2 = [], [], [], []
    for idx in pos_idx:
        pos_x1.append(x1[idx])
        pos_x2.append(x2[idx])
    for idx in neg_idx:
        neg_x1.append(x1[idx])
        neg_x2.append(x2[idx])

    plt.figure(figsize=(8, 8))
    plt.xlabel('x1', fontsize=13)
    plt.ylabel('x2', fontsize=13)

    plt.scatter(pos_x1, pos_x2, label='1', c='red')
    plt.scatter(neg_x1, neg_x2, label='-1', c='blue')

    plt.tight_layout()
    plt.legend(loc='best', fontsize=13)
    plt.show()


# try three vectors
def Question2_b(x1, x2, yval):
    # (x1^2, x2^2)
    for i in range(len(x1)):
        x1[i] = x1[i] * x1[i]
        x2[i] = x2[i] * x2[i]

    # (x1^2, √2x1x2)
    # for i in range(len(x1)):
    #     x2[i] = np.sqrt(2) * x1[i] * x2[i]
    #     x1[i] = x1[i] * x1[i]

    # (x2^2, √2x1x2)
    # for i in range(len(x1)):
    #     x1[i] = np.sqrt(2) * x1[i] * x2[i]
    #     x2[i] = x2[i] * x2[i]

    Question2_a(x1, x2, yval)


# this algorithm copyright lec PPT
# train a perceptron
def Question2_c(x1, x2, yval):
    lr = 0.2
    w = np.array([1, 1, 1, 1])
    w_dic = {'w0': '%.2f' % w[0], 'w1': '%.2f' % w[1], 'w2': '%.2f' % w[2], 'w3': '%.2f' % w[3]}
    w_df = pd.DataFrame(columns=['w0', 'w1', 'w2', 'w3'])
    w_df = w_df.append(w_dic, ignore_index=True)
    converged = False
    while converged is False:
        converged = True
        for i in range(len(x1)):
            phi_x = np.array((1, x1[i] ** 2, x2[i] ** 2, np.sqrt(2) * x1[i] * x2[i])).transpose()
            val = yval[i] * np.dot(phi_x, w)
            if val <= 0:
                w = w + np.dot(lr * yval[i], phi_x)
                w_dic = {'w0': '%.2f' % w[0], 'w1': '%.2f' % w[1], 'w2': '%.2f' % w[2], 'w3': '%.2f' % w[3]}
                w_df = w_df.append(w_dic, ignore_index=True)
                converged = False

    # show my model has converged
    r_df = pd.DataFrame(columns=['xi', 'Phi(xi)', 'yi*Phi(xi)*w'])
    for i in range(len(x1)):
        phi_x = np.array((1, x1[i] ** 2, x2[i] ** 2, np.sqrt(2) * x1[i] * x2[i])).transpose()
        r_dic = {'xi': ('%.2f' % x1[i], '%.2f' % x2[i]), 'Phi(xi)': np.around(phi_x[1:], 2),
                 'yi*Phi(xi)*w': '%.2f' % (yval[i] * np.dot(phi_x, w))}
        r_df = r_df.append(r_dic, ignore_index=True)

    print(w_df)
    print(r_df)


if __name__ == "__main__":
    # # -------------------------- Question 1 --------------------------
    # x, y = create_dataset(n=1250, nf=2, nr=0, ni=2, random_state=125)
    # xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=45)
    # titles = ['Decision Tree', 'Random Forest', 'AdaBoost', 'Logistic Regression', 'Neural Network', 'SVM']
    #
    # Question1_a(x, xtrain, xtest, ytrain, ytest, titles)
    # Question1_b(xtrain, xtest, ytrain, ytest, titles)
    # Question1_c(xtrain, xtest, ytrain, ytest, titles)
    #
    # x, y = create_dataset(n=2000, nf=20, nr=12, ni=8, random_state=25)
    # xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.5, random_state=15)
    #
    # Question1_d(xtrain, xtest, ytrain, ytest)
    # Question1_e(xtrain, xtest, ytrain, ytest)
    # Question1_f(xtrain, xtest, ytrain, ytest)
    # Question1_g(xtrain, xtest, ytrain, ytest)

    # -------------------------- Question 2 --------------------------

    x1 = [-0.8, 3.9, 1.4, 0.1, 1.2, -2.45, -1.5, 1.2]
    x2 = [1, 0.4, 1, -3.3, 2.7, 0.1, -0.5, -1.5]
    y = [1, -1, 1, -1, -1, -1, 1, 1]

    # Question2_a(x1, x2, y)
    # Question2_b(x1, x2, y)
    Question2_c(x1, x2, y)
