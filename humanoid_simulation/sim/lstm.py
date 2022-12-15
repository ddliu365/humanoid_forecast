import torch
import torch.nn as nn
import torch.optim as optim 
import math
import numpy as np
import csv
import json



class Predictor(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim):
        super(Predictor, self).__init__()

        self.weights=nn.Sequential(
            nn.BatchNorm1d(outputDim),
            nn.Linear(outputDim, 1),
            nn.Sigmoid()
        )
        self.rnn = nn.LSTM(input_size = inputDim,
                            hidden_size = hiddenDim,
			    num_layers=2,
                            batch_first = True)
        self.output_layer =nn.Linear(hiddenDim, outputDim)
    
    def forward(self, inputs, hidden0=None):
        output, (hidden, cell) = self.rnn(inputs, hidden0)
        out1 = (self.output_layer(output[:, -1, :]))
        out2=self.weights(out1)
        return out2