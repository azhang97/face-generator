'''
model_util.py

Utility functions for the models
'''
import torch.nn as nn

# Layer to reshape the input for Sequential() mode
# https://discuss.pytorch.org/t/equivalent-of-np-reshape-in-pytorch/144/4
class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)
