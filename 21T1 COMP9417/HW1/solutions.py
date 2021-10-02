import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Question 1 Pre-processing
# Q1: (a) remove rows which contain NA values
def remove_NA(dataframe):
    return dataframe.dropna()


# Q1: (a) delete columns:
# delete columns: transactiondate, latitude, longitude, price
def delete_columns(dataframe, del_name):
    return dataframe.drop(del_name, axis=1)


# Q1: (b) feature normalisation
def feature_normalisation(dataframe):
    # x_new = (x - min(x)) / (max(x) - min(x))
    dataframe = (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())
    return dataframe


# Question 2 Train and Test sets
def set_division(dataframe):
    # ---------------- training_set ---------------------')
    training = dataframe.loc[:207, :].copy()
    # ------------------ test_set ------------------')
    test = dataframe.loc[208:, :].copy()
    return training, test


# copy from hw1 details
# Generate a 3×3 grid of plots
def draw_loss_plot(losses, alphas):
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(ax.flat):
        # losses is a list of 9 elements. Each element is an array of length nIter
        # storing the loss at each iteration for
        # that particular step size
        ax.plot(losses[i])
        ax.set_title(f"step size: {alphas[i]}")  # plot titles
    plt.tight_layout()  # plot formatting
    plt.show()


# Generate a 1×1 grid of plots
def draw_weights_plot(ws, lr):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for idx in range(len(ws)):
        ax.plot(ws[idx], label='w' + str(idx))
    ax.set_title(f"step size: {lr}")  # plot titles
    plt.tight_layout()  # plot formatting
    plt.legend()
    plt.show()


# calculate the mean of loss
def cal_mean_loss(data_set, w):
    loss_sum = 0
    for row in data_set.itertuples():
        diff = np.dot(w, np.array([row[1], row[2], row[3], row[4]])) - row[5]
        loss_sum = loss_sum + (((0.25 * (diff ** 2)) + 1) ** 0.5) - 1
    return loss_sum / data_set.shape[0]


# gradient descent
def gradient_descent(data_set, nIter, lr, w):
    loss = []
    ws = [[w[0]], [w[1]], [w[2]], [w[3]]]
    for i in range(nIter):
        sum_loss = 0
        sum_der = np.array([0, 0, 0, 0])
        for row in data_set.itertuples():
            real_price = row[5]
            x = np.array([row[1], row[2], row[3], row[4]])
            predict_price = np.dot(w, x)
            diff = predict_price - real_price
            # calculate loss
            sum_loss = sum_loss + (((0.25 * (diff ** 2)) + 1) ** 0.5) - 1
            sum_der = sum_der + (x * diff) / (2 * (((diff ** 2) + 4) ** 0.5))
        # update weight vector
        w = w - lr * (sum_der / data_set.shape[0])
        mean_loss = sum_loss / data_set.shape[0]
        loss.append(mean_loss)
        for idx in range(len(ws)):
            ws[idx].append(w[idx])
    return loss, ws, w


# Question 5 Gradient Descent Implementation - (a)
def GD_loss_plot(data_set, nIter, lrs):
    losses = []
    for lr in lrs:
        w_0 = np.array([1, 1, 1, 1])  # initialise weight vector
        loss, ws, w = gradient_descent(data_set, nIter, lr, w_0)
        losses.append(loss)
    draw_loss_plot(losses, lrs)
    return w


# Question 5 Gradient Descent Implementation - (c)
def GD_weights_plot(data_set, nIter, lr):
    losses = []
    w_0 = np.array([1, 1, 1, 1])  # initialise weight vector
    loss, ws, w = gradient_descent(data_set, nIter, lr, w_0)
    losses.append(loss)
    # show weights
    draw_weights_plot(ws, lr)
    return w


# Question 6 Stochastic Gradient Descent Implementation
def stochastic_gradient_descent(data_set, ep, lr, w):
    loss = []
    ws = [[w[0]], [w[1]], [w[2]], [w[3]]]
    for i in range(ep):
        for row in data_set.itertuples():
            real_price = row[5]
            x = np.array([row[1], row[2], row[3], row[4]])
            predict_price = np.dot(w, x)
            diff = predict_price - real_price
            mean_loss = cal_mean_loss(data_set, w)
            # update weight vector
            w = w - lr * ((x * diff) / (2 * (((diff ** 2) + 4) ** 0.5)))
            for idx in range(len(ws)):
                ws[idx].append(w[idx])
            loss.append(mean_loss)
    return loss, ws, w


# Question 6 Stochastic Gradient Descent Implementation - (a)
def SGD_loss_plot(data_set, ep, lrs):
    losses = []
    for lr in lrs:
        w_0 = np.array([1, 1, 1, 1])  # initialise weight vector
        loss, ws, w = stochastic_gradient_descent(data_set, ep, lr, w_0)
        losses.append(loss)
    draw_loss_plot(losses, lrs)


# Question 6 Stochastic Gradient Descent Implementation - (c)
def SGD_weights_plot(data_set, ep, lr):
    losses = []
    w_0 = np.array([1, 1, 1, 1])  # initialise weight vector
    loss, ws, w = stochastic_gradient_descent(data_set, ep, lr, w_0)
    losses.append(loss)
    # show weights
    draw_weights_plot(ws, lr)
    return w


read_file = pd.read_csv('real_estate.csv')
df = pd.DataFrame(read_file)

print('------------------------------ Question 1 ------------------------------')
Q1_a = remove_NA(df)
idx_of_NA = list(df.drop(Q1_a.index).index)
print('Q1 (a) The indices of the removed data points: ', idx_of_NA)
del_col_name = ['transactiondate', 'latitude', 'longitude', 'price']
Q1_b = delete_columns(Q1_a, del_col_name)

Q1_b = feature_normalisation(Q1_b)
print('Q1 (b) The mean value of each feature:\n', Q1_b.describe().iloc[1:2])

print('\n------------------------------ Question 2 ------------------------------')
training_set, test_set = set_division(Q1_b)
print('Q2 The first row of training set: \n', training_set.iloc[:1])
print('Q2 The last row of training set: \n', training_set.iloc[-1:])
print('\nQ2 The first row of test set: \n', test_set.iloc[:1])
print('Q2 The last row of test set: \n', test_set.iloc[-1:])

# preprocessing
df_with_price = pd.concat([Q1_b, Q1_a['price']], axis=1)
training_set, test_set = set_division(df_with_price)
training_set.insert(0, 'col0', 1)
test_set.insert(0, 'col0', 1)

print('------------------------------ Question 5 ------------------------------')
# Q5: (a)
num_of_iteration = 400
learning_rates = [10, 5, 2, 1, 0.5, 0.25, 0.1, 0.05, 0.01]
GD_loss_plot(training_set, num_of_iteration, learning_rates)

# Q5: (c)
learning_rate = 0.3
weight = GD_weights_plot(training_set, num_of_iteration, learning_rate)
print('Q5 (c) The final weight vector:  ', weight)
Q5_training_set_loss = cal_mean_loss(training_set, weight)
Q5_test_set_loss = cal_mean_loss(test_set, weight)
print('Q5 (c) The loss of training_set: ', Q5_training_set_loss)
print('Q5 (c) The loss of test_set:     ', Q5_test_set_loss)

print('------------------------------ Question 6 ------------------------------')
# Q6: (a)
epoch = 6
learning_rates = [10, 5, 2, 1, 0.5, 0.25, 0.1, 0.05, 0.01]
SGD_loss_plot(training_set, epoch, learning_rates)

# Q6: (c)
learning_rate = 0.4
weight = SGD_weights_plot(training_set, epoch, learning_rate)
print('Q6 (c) The final weight vector:  ', weight)
Q6_training_set_loss = cal_mean_loss(training_set, weight)
Q6_test_set_loss = cal_mean_loss(test_set, weight)
print('Q6 (c) The loss of training_set: ', Q6_training_set_loss)
print('Q6 (c) The loss of test_set:     ', Q6_test_set_loss)
