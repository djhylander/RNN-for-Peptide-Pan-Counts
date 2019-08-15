# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:33:03 2019

@author: dhyla
"""

'''
start with peptide one hot, split into 12 vectors
randomly initialized hidden state 12 x 20 and weight 20 x 20
h^i+1 = Tanh((h^i * w) * x^i)
get h^12 and do it again with k
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as Var
import numpy as np
import pandas as pd
from collections import OrderedDict

#device = torch.device("cuda")

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
    #result = torch.Tensor(list(aaDict.values())).t()
    result = np.array(list(aaDict.values()))
    return result

# file input
dirname = 'C:/Users/dhyla/OneDrive/Desktop/Research/'
data = pd.read_csv(dirname + 'All_peptides_Set1.csv')
data.set_index('AA_seq',inplace=True)
data['Total']=data.CP1+data.CP2+data.CP3+data.CE
data = data[data.Total>=5]

# hyperparameters
hidden0 = Var(torch.randn(12,20), requires_grad=True)
weight0 = Var(torch.randn(20,1), requires_grad=True)
hidden1 = Var(torch.randn(12,20), requires_grad=True)
weight1 = Var(torch.randn(20,12), requires_grad=True)
weightOutput = Var(torch.randn(20,1), requires_grad=True)
lr = 1e-5

m = nn.Tanh()
optimizer = optim.Adam([hidden0,weight0,hidden1,weight1,weightOutput],lr=lr)
loss_fn = nn.MSELoss()
epoch=0
epochloss=[]
while epoch <=10:
    iteration = 0
    iterationloss=[]
    while iteration <=500:
        pep = data.sample().index[0]
        pephot = oneHotter(pep)
        pepsubs = np.split(pephot, 12, 1)
        for x in pepsubs:
            x = torch.Tensor(np.transpose(x))
            hidden0 = m(torch.mm(torch.mm(hidden0, weight0), x))
            
        preP1 = m(torch.mm(torch.mm(hidden1, weight1), hidden0))
        preP2 = m(torch.mm(torch.mm(preP1, weight1), hidden0))
        preP3 = m(torch.mm(torch.mm(preP2, weight1), hidden0))
        PreE = m(torch.mm(torch.mm(preP3, weight1), hidden0))
        
        pan1 = torch.sum(torch.mm(preP1, weightOutput))
        pan2 = torch.sum(torch.mm(preP2, weightOutput))
        pan3 = torch.sum(torch.mm(preP3, weightOutput))
        end = torch.sum(torch.mm(PreE, weightOutput))
        
        yPred = Var(torch.Tensor([pan1.item(), pan2.item(), pan3.item(), end.item()]), requires_grad=True)
        yActual = Var(torch.Tensor([data.loc[pep][7], data.loc[pep][6], 
                   data.loc[pep][5], data.loc[pep][4]]), requires_grad=False)
    #    yActual = Var(torch.Tensor([data['CP1'][data.index == pep], data['CP2'][data.index == pep], 
    #               data['CP3'][data.index == pep], data['CE'][data.index == pep]]), requires_grad=True)
        loss = loss_fn(yPred, yActual)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("iteration:", iteration, " loss:", round(loss.item(), 3))
        iterationloss.append(loss.item())
        iteration +=1
    epochloss.append(np.mean(iterationloss))
    print('epoch', epoch, 'completed')
    epoch +=1

import matplotlib.pyplot as plt
plt.scatter(np.linspace(0,len(epochloss),num=len(epochloss)),epochloss)

















'''
class Net(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):

        #Initializing hidden state for first input using method defined below
        hidden = self.init_hidden()

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, self.hidden_dim)
         # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden
    
    
model = Net()

loss_fn = torch.nn.SmoothL1Loss(size_average=False)

optimizer = optim.Adam(model.parameters(), lr=1e-5)

# file input
dirname = 'C:/Users/dhyla/OneDrive/Desktop/Research/'
data = pd.read_csv(dirname + 'All_peptides_Set1.csv')
data.set_index('AA_seq',inplace=True)

model.train()
'''



