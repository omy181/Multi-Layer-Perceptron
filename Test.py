import pandas as pandas
from NeuralNet import * 


Pandas_Data = pandas.read_csv("midtermProject-part1-TRAIN.csv")

# insert layer counts array without the input layer
TrainNeuralNetwork(Pandas_Data,[3,1],1)
