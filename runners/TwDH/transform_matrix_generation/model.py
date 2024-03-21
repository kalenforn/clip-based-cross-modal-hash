import torch
import torch.nn as nn
import numpy as np


class Matrix(nn.Module):

    def __init__(self, input_dim=512, output_dim=16, select=True):

        super(Matrix, self).__init__()
        self.select = select

        matrix = torch.from_numpy(np.random.uniform(low=-1, high=1, size=(input_dim * 2, output_dim * 2))).float() if select else torch.from_numpy(np.random.uniform(low=-1, high=1, size=(input_dim, output_dim))).float()
        self.matrix = nn.Parameter(matrix)
    
    def forward(self, x):
        
        if self.select:
            output = x.mm(self.matrix).view(x.shape[0], -1, 2)
            output = torch.softmax(output, dim=-1).view(output.shape[0], -1)
        else:
            print(x.shape, self.matrix.shape)
            output = x.mm(self.matrix)
        return output
