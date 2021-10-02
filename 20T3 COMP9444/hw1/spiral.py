# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        # INSERT CODE HERE
        self.linear1 = nn.Linear(in_features=2, out_features=num_hid, bias=True)
        self.linear2 = nn.Linear(in_features=num_hid, out_features=1, bias=True)
        self.tanh = nn.Tanh()
        # self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # print('input.shape', input.shape)
        # print('input: ', input)
        x = input[:, 0]
        y = input[:, 1]
        r = torch.sqrt(x * x + y * y)
        a = torch.atan2(y, x)
        # print('r.shape: ', r.shape)
        # print('a.shape: ', a.shape)
        reshape_r = r.view(-1, 1)
        reshape_a = a.view(-1, 1)
        # print('reshape_r: ', reshape_r)
        # print('reshape_a: ', reshape_a)
        combine_input = torch.cat([reshape_r, reshape_a], dim=1)
        linear_layer1 = self.linear1(combine_input)
        self.hid1 = self.tanh(linear_layer1)
        # self.hid1 = self.relu(linear_layer1)
        linear_layer2 = self.linear2(self.hid1)
        sigmoid_output = self.sigmoid(linear_layer2)  # CHANGE CODE HERE
        return sigmoid_output


class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        # INSERT CODE HERE
        self.linear1 = nn.Linear(in_features=2, out_features=num_hid, bias=True)
        self.linear2 = nn.Linear(in_features=num_hid, out_features=num_hid, bias=True)
        self.linear3 = nn.Linear(in_features=num_hid, out_features=1, bias=True)
        self.tanh = nn.Tanh()
        # self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        linear_layer1 = self.linear1(input)
        self.hid1 = self.tanh(linear_layer1)
        # self.hid1 = self.relu(linear_layer1)
        linear_layer2 = self.linear2(self.hid1)
        self.hid2 = self.tanh(linear_layer2)
        # self.hid2 = self.relu(linear_layer2)
        linear_layer3 = self.linear3(self.hid2)

        sigmoid_output = self.sigmoid(linear_layer3)  # CHANGE CODE HERE
        return sigmoid_output


def graph_hidden(net, layer, node):
    plt.clf()
    # INSERT CODE HERE

    # copy from spiral_main.py
    xrange = torch.arange(start=-7, end=7.1, step=0.01, dtype=torch.float32)
    yrange = torch.arange(start=-6.6, end=6.7, step=0.01, dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1), ycoord.unsqueeze(1)), 1)

    with torch.no_grad():  # suppress updating of gradients
        net.eval()  # toggle batch norm, dropout
        output = net(grid)
        net.train()  # toggle batch norm, dropout back again

        # change here
        ##############################################################
        if layer == 1:  # first linear
            pred = (net.hid1[:, node] >= 0).float()
        elif layer == 2:  # second linear
            pred = (net.hid2[:, node] >= 0).float()
        ##############################################################
        # plot function computed by model
        plt.clf()
        plt.pcolormesh(xrange, yrange, pred.cpu().view(yrange.size()[0], xrange.size()[0]), cmap='Wistia')
