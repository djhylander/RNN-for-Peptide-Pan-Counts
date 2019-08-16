# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 18:52:02 2019
@author: dhyla

Attempts to learn through an RNN the pan counts of peptides given the peptide one-hot.
Uses PyTorch for the optimizing, and gradients.
"""
# import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'

import torch
import torch.nn as nn
from torch.autograd import Variable as Var
import torch.optim as optim
import pandas as pd
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

# Returns a 20 x 12 one hot encoding of peptide
AA=['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
loc=['N','2','3','4','5','6','7','8','9','10','11','C']
def oneHotter(peptide):
    aminoAcids = list(peptide)  # Below: initialize ordered dictionary to hold empty lists of amino acid length
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
    # result = np.array(list(aaDict.values()))  # in case numpy is needed
    return result

# First RNN to take each column of a peptide one-hot and return the last hidden layer
class NewRNN(nn.Module):
    def __init__(self, batch_size, n_inputs, n_neurons):
        super(NewRNN, self).__init__()
        self.rnn = nn.RNNCell(n_inputs, n_neurons, bias=True, nonlinearity='relu')  # can use tanh instead
        self.hx = torch.randn(batch_size, n_neurons)  # hidden state

    def forward(self, x):
        for i in range(12):  # 12 columns in a peptide one-hot
            self.hx = self.rnn(x[i], self.hx)
        return self.hx

# Second RNN that takes the last hidden layer from first RNN and returns the output vector correlating to peptide pan counts
class NewRNN2(nn.Module):
    def __init__(self, batch_size, n_inputs, n_neurons):
        super(NewRNN2, self).__init__()
        self.rnn = nn.RNNCell(n_inputs, n_neurons, bias=True, nonlinearity='relu')  # can use tanh instead
        self.hx = torch.randn(batch_size, n_neurons)  # hidden state

    def forward(self, x):
        output = []
        for i in range(4):  # 4 pan counts: CP1, CP2, CP3, CE
            self.hx = self.rnn(x, self.hx)
            output.append(self.hx)
        return output

# Reads in data and sets index to peptide sequence
data = pd.read_csv('cleaned_data_avg.csv')
data.set_index('AA_seq',inplace=True)

batch_size = 1
n_inputs = 20
n_neurons = 20

# torch.cuda.set_device(0)  # if gpu is needed

model = NewRNN(batch_size, n_inputs, n_neurons)
model2 = NewRNN2(batch_size, n_inputs, n_neurons)

loss_fn = torch.nn.SmoothL1Loss()
params = list(model.parameters()) + list(model2.parameters())
optimizer = optim.Adam(params, lr=1e-5, weight_decay=.01)

# Loops through the data randomly, inputs one-hot in models and backward propagates to find error
num_epochs = 10
epoch_all = []
# outliers_all = []
for epoch in range(num_epochs):
    model.train()
    model2.train()
    loss_all = []
    # outliers_curr = []
    for i in range(len(data)):
        pep = data.sample().index[0]
        pephot = oneHotter(pep)
        pepsubs = torch.split(pephot, 1)
        for x in range(12):
            torch.transpose(pepsubs[x], 0, 1)
        output_hidden = model(pepsubs)  # 1 x 20 last hidden state
        output_vector = model2(output_hidden)  # prediction of pan counts
        yPred = [0,0,0,0]
        for j in range(4):
            yPred[j] = torch.abs(torch.sum(output_vector[j]))
        yPredTensor = Var(torch.Tensor(yPred), requires_grad=True)
        yActual = [data.loc[pep][0], data.loc[pep][1],  # actual pan counts for the peptide
                   data.loc[pep][2], data.loc[pep][3]]
        yActualTensor = Var(torch.Tensor(yActual), requires_grad=False)
    
        loss = loss_fn(yPredTensor, yActualTensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # if loss.item() > 20:  # roughly 2% of the results
            # outliers_curr.append(loss.item())
        if loss.item() < 20:  
            loss_all.append(loss.item())
        print("epoch: ", epoch, " iteration: ", i, ' loss: ', loss.item())

    # outliers_all.append(outliers_curr)
    epoch_all.append(np.mean(loss_all))

# If need to graph error within an epoch
'''
loss_allGraph = plt.scatter(x = np.linspace(0, len(loss_all),num = len(loss_all)), y = loss_all)
plt.title("Loss All Trend")
plt.xlabel(xlabel = 'Iter')
plt.ylabel(ylabel = 'Loss')
plt.ylim(0, max(loss_all))
# plt.savefig(dirname + '/Plots RNN/SmoothL1Loss RNN Cell.png')
plt.show()
plt.close()
'''
# Prints and saves a figure of epoch error
graphLoss = plt.scatter(x = np.linspace(0, len(epoch_all),num = len(epoch_all)), y = epoch_all)
plt.title("Epoch Loss Trend")
plt.xlabel(xlabel = 'Epochs (5000 iter)')
plt.ylabel(ylabel = 'RNN Cell Loss')
plt.ylim(0, max(epoch_all))
#plt.savefig('/Plots RNN/SmoothL1Loss RNN Cell.png')
plt.show()
plt.close()








