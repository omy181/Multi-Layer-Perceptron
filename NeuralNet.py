import pandas
import numpy
import torch as torch
import torch.nn as nn
import torch.nn.functional as f
from torcheval.metrics import R2Score

def MeanAbsoluteError(prediction_list,truth_list):
    return (sum(abs(prediction_list-truth_list))/len(prediction_list)).detach().numpy()[0]

def RootMeanSquaredError(prediction_list,truth_list):
    diff = prediction_list-truth_list
    return torch.sqrt(sum(diff*diff)/len(prediction_list)).detach().numpy()[0]

def R2(prediction_list,truth_list):
    metric = R2Score()
    metric.update(prediction_list,truth_list)
    return metric.compute()

def MeanSquaredError(prediction_list,truth_list):
    criterion = nn.MSELoss()
    return criterion(prediction_list,truth_list)

class Neural_Network(nn.Module):

    def __init__ (self,I,layers):
        super().__init__()
        
        self.l1 = nn.Linear(I,layers[0])
        self.l2 = nn.Linear(layers[0],layers[1])

        """self.layerslist = []

        self.layerslist.append(nn.Linear(I,layers[0]))
        for layerindex,_ in enumerate(layers,1):   
            if layerindex == len(layers): break         
            self.layerslist.append(nn.Linear(layers[layerindex-1],layers[layerindex]))"""


    def forward_propogation(self,x):

        x = f.relu(self.l1(x))
        x = self.l2(x)

        """ for layerindex,layer in enumerate(layerslist): 
            if layerindex == len(layerslist)-1: break   
            x = f.relu(self.layer(x))
        
        x = self.layerslist[-1](x)"""
        return x



def TrainNeuralNetwork(pandas_data:pandas.DataFrame,layers:list,TruthLabel,learning_rate,epoch):
    
    # data preperation
    Train_data = pandas_data.drop(TruthLabel,axis=1)
    Train_labels = pandas_data[TruthLabel]

    input_node_count = Train_data.values.shape[1]

    Train_data = torch.FloatTensor(Train_data.values)
    Train_labels = torch.FloatTensor(Train_labels.values).unsqueeze(1)

    # initialize model
    model = Neural_Network(input_node_count,layers)

   
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),learning_rate)

    # training
    for i in range(epoch):
        prediction_list = model.forward_propogation(Train_data)

        # calculate error
        error = criterion(prediction_list,Train_labels)

        
        if i % (epoch/10) == 0:
            print(f"Progress: {(i/epoch)*100}% Error: {error}")


        # back propogation
        optimizer.zero_grad()
        error.backward()
        optimizer.step()

    print(f"\nTrain set MSE: {MeanSquaredError(prediction_list,Train_labels)}")
    print(f"Train set MAE: {MeanAbsoluteError(prediction_list,Train_labels)}")
    print(f"Train set RMSE: {RootMeanSquaredError(prediction_list,Train_labels)}")
    print(f"Train set R2: {R2(prediction_list,Train_labels)}")
    return model


def TestNeuralNetwork(model:Neural_Network,pandas_data:pandas.DataFrame,TruthLabel):

    # data preperation
    Test_data = pandas_data.drop(TruthLabel,axis=1)
    Test_labels = pandas_data[TruthLabel]

    Test_data = torch.FloatTensor(Test_data.values)
    Test_labels = torch.FloatTensor(Test_labels.values).unsqueeze(1)


    prediction_list = model.forward_propogation(Test_data)

    # test
    criterion = nn.MSELoss()
    error = criterion(prediction_list,Test_labels)

    print(f"\nTest set MSE: {MeanSquaredError(prediction_list,Test_labels)}")
    print(f"Test set MAE: {MeanAbsoluteError(prediction_list,Test_labels)}")
    print(f"Test set RMSE: {RootMeanSquaredError(prediction_list,Test_labels)}")
    print(f"Test set R2: {R2(prediction_list,Test_labels)}")
    

    """for prediction,TruthLabel in zip(prediction_list.detach().numpy(),Test_labels.detach().numpy()):
        print(f"P/T: {TruthLabel -prediction} ")"""
        
