import pandas
import numpy
import torch as torch
import torch.nn as nn
import torch.nn.functional as f

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

    
    return model


def TestNeuralNetwork(model:Neural_Network,pandas_data:pandas.DataFrame,TruthLabel):

    

    return 


