# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # INSERT CODE HERE
        self.linear = nn.Linear(in_features=28 * 28, out_features=10, bias=True)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # print('x.type: ', type(x))
        # print('x.shape: ', x.shape)
        # print('x.size: ', x.size())
        reshape_x = x.view([-1, 28 * 28])
        # print('reshape_x.shape: ', reshape_x.shape)
        linear_layer = self.linear(reshape_x)
        log_softmax_output = self.log_softmax(linear_layer)
        return log_softmax_output  # CHANGE CODE HERE


class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE
        num_of_hid_nodes = 10000
        self.linear1 = nn.Linear(in_features=28 * 28, out_features=num_of_hid_nodes, bias=True)
        self.linear2 = nn.Linear(in_features=num_of_hid_nodes, out_features=10, bias=True)
        self.tanh = nn.Tanh()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        reshape_x = x.view([-1, 28 * 28])
        # print('reshape_x.shape: ', reshape_x.shape)
        linear_layer1 = self.linear1(reshape_x)
        hidden = self.tanh(linear_layer1)
        linear_layer2 = self.linear2(hidden)
        log_softmax_output = self.log_softmax(linear_layer2)
        return log_softmax_output  # CHANGE CODE HERE


class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Linear(in_features=64 * 7 * 7, out_features=64, bias=True)
        self.linear2 = nn.Linear(in_features=64, out_features=10, bias=True)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.max_pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # x.shape:  torch.Size([64, 1, 28, 28])

        conv_layer1 = self.conv1(x)
        relu = self.relu(conv_layer1)

        # use max_pool
        pooling = self.max_pool(relu)

        conv_layer2 = self.conv2(pooling)
        relu = self.relu(conv_layer2)

        # use max_pool
        pooling = self.max_pool(relu)

        # print('pooling.shape: ', pooling.shape)
        reshape_input = pooling.view([-1, 64 * 7 * 7])
        # full connected layer
        linear_layer1 = self.linear1(reshape_input)
        relu = self.relu(linear_layer1)

        # output layer
        linear_layer2 = self.linear2(relu)
        log_softmax_output = self.log_softmax(linear_layer2)
        return log_softmax_output  # CHANGE CODE HERE
        # return 0

