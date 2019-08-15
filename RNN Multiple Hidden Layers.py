# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 18:52:02 2019

@author: dhyla
"""

import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.autograd import Variable as Var
import torch.optim as optim
import pandas as pd
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt


# returns a 20 x 12 one hot encoding of peptide
AA=['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
loc=['N','2','3','4','5','6','7','8','9','10','11','C']
def oneHotter(peptide):
    aminoAcids = list(peptide)  # below: initialize ordered dictionary to hold empty lists of amino acid length
    aaDict = OrderedDict()
    aaDict = {"A": [], "R": [], "N": [], "D": [], "C": [], "Q": [], "E": [], "G": [], "H": [], "I": [],
              "L": [], "K": [], "M": [], "F": [], "P": [], "S": [], "T": [], "W": [], "Y": [], "V": []}
    for index in range(len(aminoAcids)):
        for key, value in aaDict.items():
            if (aminoAcids[index] == key):
                value.append(1)
            else:
                value.append(0)
    result = torch.Tensor(list(aaDict.values())).t()
    #result = np.array(list(aaDict.values()))
    return result


class RNN(nn.Module):
    def __init__(self, n_inputs, n_neurons, n_outputs):
        super(RNN, self).__init__()

        self.weight0 = torch.randn(n_neurons, n_outputs) # 20 x 1
        self.hidden0 = torch.randn(n_inputs, n_neurons) # 12 x 20

    def forward(self, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
        self.h1 = torch.tanh(torch.mm(torch.mm(self.hidden0, self.weight0), x0))
        self.h2 = torch.tanh(torch.mm(torch.mm(self.h1, self.weight0), x1))
        self.h3 = torch.tanh(torch.mm(torch.mm(self.h2, self.weight0), x2))
        self.h4 = torch.tanh(torch.mm(torch.mm(self.h3, self.weight0), x3))
        self.h5 = torch.tanh(torch.mm(torch.mm(self.h4, self.weight0), x4))
        self.h6 = torch.tanh(torch.mm(torch.mm(self.h5, self.weight0), x5))
        self.h7 = torch.tanh(torch.mm(torch.mm(self.h6, self.weight0), x6))
        self.h8 = torch.tanh(torch.mm(torch.mm(self.h7, self.weight0), x7))
        self.h9 = torch.tanh(torch.mm(torch.mm(self.h8, self.weight0), x8))
        self.h10 = torch.tanh(torch.mm(torch.mm(self.h9, self.weight0), x9))
        self.h11 = torch.tanh(torch.mm(torch.mm(self.h10, self.weight0), x10))
        self.h12 = torch.tanh(torch.mm(torch.mm(self.h11, self.weight0), x11))

        return self.h12 # 12 x 20

class RNN2(nn.Module):
    def __init__(self, n_inputs, n_neurons, n_outputs):
        super(RNN2, self).__init__()

        self.weight1 = torch.randn(n_neurons, n_inputs) # 20 x 12
        self.hidden1 = torch.randn(n_inputs, n_neurons) # 12 x 20
        self.weightY = torch.randn(n_neurons, n_outputs) # 20 x 1

    def forward(self, h12):
        self.p1 = torch.tanh(torch.mm(torch.mm(self.hidden1, self.weight1), h12))
        self.p2 = torch.tanh(torch.mm(torch.mm(self.p1, self.weight1), h12))
        self.p3 = torch.tanh(torch.mm(torch.mm(self.p2, self.weight1), h12))
        self.e = torch.tanh(torch.mm(torch.mm(self.p3, self.weight1), h12))

        self.p1 = torch.sum(torch.mm(self.p1, self.weightY))
        self.p2 = torch.sum(torch.mm(self.p2, self.weightY))
        self.p3 = torch.sum(torch.mm(self.p3, self.weightY))
        self.e = torch.sum(torch.mm(self.e, self.weightY))

        return [self.p1, self.p2, self.p3, self.e]


# file input
data = pd.read_csv('cleaned_set1.csv')
data.set_index('AA_seq',inplace=True)

data = data.sample(10)

model = RNN(20, 20, 1)
model2 = RNN2(20,20, 1)
loss_fn = torch.nn.SmoothL1Loss()
parameters = [model.weight0, model.hidden0, model2.weight1, model2.hidden1, model2.weightY]
optimizer = optim.Adam(parameters, lr=1e-5)

num_epochs = 50
numIter = 500
epoch_all = []
# outliers_all = []
for epoch in range(num_epochs):
    model.train()
    model2.train()
    loss_all = []
    # outliers_curr = []
    for i in range(len(data)):
        optimizer.zero_grad()
        pep = data.sample().index[0]
        pephot = oneHotter(pep)
        pepsubs = torch.split(pephot, 1)
        for x in range(12):
            torch.transpose(pepsubs[x], 0, 1)
        hiddenEnd = model(pepsubs[0], pepsubs[1], pepsubs[2], pepsubs[3], pepsubs[4], pepsubs[5], pepsubs[6], pepsubs[7],
                        pepsubs[8], pepsubs[9], pepsubs[10], pepsubs[11])
        yPred = model2(hiddenEnd)
        for j in range(len(yPred)):
            yPred[j] = torch.abs(torch.sum(yPred[j])) #round(yPred[j].data.item(), 7)
        yActual = [data.loc[pep][0], data.loc[pep][1],
                   data.loc[pep][2], data.loc[pep][3]]
        yPredTensor = Var(torch.Tensor(yPred), requires_grad=True)
        yActualTensor = Var(torch.Tensor(yActual), requires_grad=False)
    
        loss = loss_fn(yPredTensor, yActualTensor)
        loss.backward()
        optimizer.step()
    
        if loss.item() < 20:
            loss_all.append(loss.item())
        print("epoch: ", epoch, " iteration: ", i) #, "\n\typred: ", yPred, "\n\tyactual: ", yActual)
    # outliers_all.append(outliers_curr)
    epoch_all.append(np.mean(loss_all))



graphLoss = plt.scatter(x = np.linspace(0, len(epoch_all),num = len(epoch_all)), y = epoch_all)
plt.xlabel(xlabel = 'Epochs (5000 iter)')
plt.ylabel(ylabel = 'Multiple Hidden Loss')
plt.ylim(0, max(epoch_all))
# plt.savefig(dirname + '/Plots RNN/SmoothL1Loss Multiple Hidden.png')
plt.show()
plt.close()





