import pandas
import numpy
import torch as torch

def TrainNeuralNetwork():
    t1 = torch.tensor([1,2,3])
    t2 = torch.tensor([1,2,3])

    print(torch.dot(t1,t2))