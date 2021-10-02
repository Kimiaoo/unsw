#!/usr/bin/env
# Group ID: g023614
# Group members: Yifan Pang (z5267024)
#                Hongxiao Jin (z5241154)
"""
Answer to the Question:

[Description of our model]
    In the part of building network model, we use LSTM to solve the problem of gradient vanishing and gradient explosion
    in the process of long sequence training.
    Our general model is divided into 2 part, rating and category.
    Rating: Input -> 2-layers LSTM -> Attention Mechanism -> Sigmoid activate Function -> Output
    Since the attitude has only two states,positive and negative,
    which means it is a 2 classification problem, we impose sigmoid to get binary results.
    Loss function we implement BSE Function.

    Category: Input -> 2-layers LSTM ->Attention Mechanism -> Log_softmax -> Output
    Category classification is a multiple classification problem, hence we implement Log_softmax as activate function
    Attention Mechanism is used to record the semantic sequence of the text,
    ensuring that the semantic relationship is not affected by word vectors.
    Loss function we implement NLLLoss Function.

[preprocessing]
    (1) We consider that some junk characters, such as '@', '#', would influence the result, so we import re module and
        use Regular Expressions to remove these characters.
    (2) convert all letters to lowercase
    (3) remove words whose length is too small(<2).
    These three methods are helpful to remove redundancies in texts.

[stopWords & wordVectors]
    (1) We added some high-frequency words without realistic meaning into the stopWords list to reduce noise
        while testing.
    (2) For wordVectors, we have tried several values for dim, finally we find that dim = 300 is suitable for our model.

[training options]
    (1) We set trainValSplit = 0.99 instead of 0.8, since we hope we could make full use of
        existing data sets and get an adequate training model.
    (2) Then we modified default batch_size 32 to 128 to prevent falling into local minimum and reduce running time.
    (3) In order to avoid over-fitting and also ensuring the accuracy, we set epochs equals to 15.
    (4) Finally, we choose Adam as optimizer because compared with SGD, Adam seldom falls into local suboptimal
        solutions or saddle points when learning rate is low.
"""
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the cfg.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
# import numpy as np
# import sklearn
import re

from config import device


################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """

    processed = sample.split()

    return processed


def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    preprocessed = list()
    # turn all letters into lowercase
    words = [w.strip().lower() for w in sample]

    for cw in words:
        # remove junk characters, such as '#' or turn 'you@' to 'you'
        cw = re.sub(r'[^#.!*@_$%a-zA-Z0-9\s]', '', cw)
        # if the length of word is too short(<2), remove it
        if len(cw) >= 2:
            preprocessed.append(cw)

    return preprocessed


def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch


# download from nltk
stopWords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
             'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
             'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
             'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
             'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but',
             'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
             'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
             'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
             'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'own',
             'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', "should've", 'now',
             'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'are', 'could', 'did', 'does', 'had', 'has', 'have', 'is',
             'ma', 'might', 'must', 'need', 'sha', 'should', 'was', 'were', 'won', 'would'}

vectorSize = 300
wordVectors = GloVe(name='6B', dim=vectorSize)


################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """
    rating_output = (ratingOutput > 0.5).long()
    category_output = categoryOutput.argmax(dim=1)
    return rating_output, category_output


################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self):
        super(network, self).__init__()

        self.hidden_unit = 75
        self.dropout_value = 0.4

        self.sigmoid = tnn.Sigmoid()
        self.log_softmax = tnn.LogSoftmax(dim=1)
        self.dropout = tnn.Dropout(self.dropout_value)

        self.lstm_rate = tnn.LSTM(vectorSize, self.hidden_unit, 2, True, True, self.dropout_value, bidirectional=True)
        self.rate_fc_attention = tnn.Linear(self.hidden_unit * 2, self.hidden_unit * 2)
        self.rate_fc_layer = tnn.Linear(self.hidden_unit * 2, 1)

        self.lstm_category = tnn.LSTM(vectorSize, self.hidden_unit, 2, True, True, self.dropout_value,
                                      bidirectional=True)
        self.category_fc_encode = tnn.Linear(self.hidden_unit * 2, 64)
        self.category_fc_attention = tnn.Linear(self.hidden_unit * 2, self.hidden_unit * 2)
        self.category_fc_layer = tnn.Linear(64, 5)

    def rate_forward(self, input, length):
        # h_n stores the output h of the last time step for each layer,
        # here it's the One-way LSTM structure, so we just focus on h_n value.
        _, (rate_h_n, rate_c_n) = self.lstm_rate(input)
        r1, r2 = rate_h_n[-2, :, :], rate_h_n[-1, :, :]
        # concat the output for last two layer
        rate_hidden_final = torch.cat([r1, r2], dim=1)

        # Implement activate function
        rate_attention = self.sigmoid(self.rate_fc_attention(rate_hidden_final))
        rate_hidden_final = rate_hidden_final * rate_attention
        rate_temp = self.rate_fc_layer(rate_hidden_final)
        rate_output = self.sigmoid(rate_temp)
        rate_output = rate_output.squeeze()
        return rate_output

    def category_forward(self, input, length):
        # similar as rate_forward
        # h_n stores the output h of the last time step for each layer,
        # here it's the One-way LSTM structure, so we just focus on h_n value.
        _, (category_h_n, category_c_n) = self.lstm_category(input)
        c1, c2 = category_h_n[-2, :, :], category_h_n[-1, :, :]
        category_hidden_final = torch.cat([c1, c2], dim=1)
        # Implement activate function
        category_attention = self.sigmoid(self.category_fc_attention(category_hidden_final))
        category_hidden_final = category_hidden_final * category_attention
        category_temp = self.category_fc_encode(category_hidden_final)
        category_encode = self.sigmoid(category_temp)
        category_fc_layer_res = self.category_fc_layer(category_encode)
        category_output = self.log_softmax(category_fc_layer_res)
        return category_output

    def forward(self, input, length):
        return self.rate_forward(input, length), self.category_forward(input, length)


class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.rate_loss = tnn.MSELoss()
        # self.rate_loss = tnn.BSELoss()
        self.category_loss = tnn.NLLLoss()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        # ratingTarget = ratingTarget
        res = self.rate_loss(ratingOutput, ratingTarget.float()) + \
              self.category_loss(categoryOutput, categoryTarget)
        return res


net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.9
batchSize = 128
epochs = 15
optimiser = toptim.Adam(net.parameters(), lr=0.001)
