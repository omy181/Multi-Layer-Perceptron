from pandas import read_csv
from NeuralNet import * 

TrainData = read_csv("midtermProject-part1-TRAIN.csv")
TestData = read_csv("midtermProject-part1-TEST.csv")

model = TrainNeuralNetwork(TrainData,[3,1],TruthLabel="ANGLE-ACC-ARM",learning_rate=0.01,epoch=1000)

TestNeuralNetwork(model,TestData,TruthLabel="ANGLE-ACC-ARM")

# printe weights and biases
for name,val in model.named_parameters():
    print("Node:",name)
    print("Values:",val)
