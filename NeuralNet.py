import pandas
import numpy
import random 
import math

def TrainNeuralNetwork(pandas_data:pandas.DataFrame,layerCounts:list,epochcount:int):

    numpy_data = numpy.array(pandas_data)
    True_Results = numpy_data[:,-1]

    inputlayerlength = len(numpy_data[0])-1
    layerCounts.insert(0,inputlayerlength)


    allweights,allbiases =CreateRandomWeightsandBiases(layerCounts)
  
    for epoch in range(epochcount):

        allerrors = []
        for index,inputdata in enumerate(numpy_data[:,0:-1],start=0):
            output = CalculateEpoch(inputdata,allweights,allbiases)
            trueanswer = numpy_data[index,-1]
            allerrors.append(abs( CalculateError(output,trueanswer)))

        meanerror = CalculateMeanError(allerrors)

        allweights,allbiases = BackPropagation(inputdata,allweights,allbiases,meanerror)

    
    return allweights,allbiases

def CalculateError(guessanswer,trueanswer):
    return trueanswer-guessanswer

def CalculateMeanError(Allerrors):
    return sum(Allerrors)/len(Allerrors)

def BackPropagation(inputs,allweights,allbiases,error):
    print(error)


    

    return allweights,allbiases

def CreateRandomWeightsandBiases(layerCounts:list):

    allweights = []
    allbiases = []
    for layerindex in range(len(layerCounts)-1):   
        allweights.append([])  
        allbiases.append([]) 
        for i in range(layerCounts[layerindex+1]):
            allweights[layerindex].append([])
            allbiases[layerindex].append([])
            for x in range(layerCounts[layerindex]):
                allweights[layerindex][i].append(random.random()*2-1)
                allbiases[layerindex][i].append(random.random()*2-1)
       
    return allweights,allbiases


def CalculateEpoch(initialinputs,allweights,allbiases,activation = True):
    inputs = initialinputs
    index = 0
    for weights,biases in zip(allweights,allbiases):
        if index == len(allweights)-1: break

        inputs = CalculateLayer(inputs,weights,biases,activation)
        index += 1

    inputs = CalculateLayer(inputs,weights,biases,False)
    return inputs[0]


def CalculateLayer(inputs,weights,biases,activation = True):
    output =[]
    for wlayer,blayer in zip(weights,biases):
        sum = 0
        for i,w,b in zip(inputs,wlayer,blayer):
           sum += i*w+b
        
        if activation:
            sum = ReLU(sum)

        output.append(sum)
    return output

def ReLU(i):
    return max(0,i)

def Sigmoid(i):
    return 1/(1+numpy.exp(-i))
   