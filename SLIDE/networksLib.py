#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  It includes data scaling, data arrangement as per SLIDE, network model
#           and evaluation of trained networks.
# Author:   Qasim Khadim, Peter Manzl, Johannes Gerstmayr
# Date:     2025-06-23
#
# Copyright:This file is part of Exudyn. Exudyn is free software.
# You can redistribute it and/or modify it under the terms of the Exudyn license. 
# See 'LICENSE.txt' for more details.
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import sys
import numpy as np
# #from math import sin, cos, sqrt,pi
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import random
import time
from timeit import default_timer as timer
from SLIDE.fnnModels import NNtestModel
from models.data_source import SLIDEModel

from exudyn.plot import PlotSensor, listMarkerStyles

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec


from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from matplotlib.patches import FancyArrowPatch

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++

def ScaleZeroToOne(x, xmin, xmax):
    LowRange  = 0.01
    HighRange = 0.99
    scaled_x = LowRange + ((x - xmin) * (HighRange - LowRange)) / (xmax - xmin)
    return scaled_x       

def ScaleMinusOneToOne(x, xmin, xmax):
     LowRange  = -0.99
     HighRange =  0.99
     
     scaled_x = LowRange + ((x - xmin) * (HighRange - LowRange)) / (xmax - xmin)
     return scaled_x
 
#ScaleBack
def ScaleBackZeroToOne(scaled_x,xmin,xmax ):
   LowRange  = 0.01
   HighRange = 0.99   
   x   = xmin + ((scaled_x - LowRange) * (xmax - xmin)) / (HighRange - LowRange)
   return x   
  
 
def ScaleBackMinusOneToOne(scaled_x,xmin,xmax):
     LowRange  = -0.99
     HighRange =  0.99
     x   = xmin + ((scaled_x - LowRange) * (xmax - xmin)) / (HighRange - LowRange)
     return x
 
    
def SLIDE_Data(nStepsTotal, nSamples=None, td=None, input_data=None,target_data=None,
              StepPredictor=None, inputLayer=None, outputLayer=None, typeName=None,
              Evaluate=None):
    
    # td : SLIDE window
    
    X = []
    Y = []
    
    if not Evaluate:
        slices = int((nStepsTotal - StepPredictor) // td)

        
        for i in range(nSamples):
            inputVec0  = []
            outputVec0 = []
        
            for j in range(slices):
                startIndex0   = j * td
                startIndex1   = (j + 1) * td-1
                endIndex      = startIndex1 + StepPredictor
                  
                inputLayers = np.hstack([input_data[key1][i][startIndex0:endIndex] for key1 in inputLayer])
                if j == 0:
                    inputVec0 = inputLayers[np.newaxis, :]
                else:
                    inputVec0 = np.vstack((inputVec0, inputLayers))
                    
                if outputLayer is not None and len(outputLayer) > 0:
                    outputLayers = np.hstack([target_data[key2][i][startIndex1:endIndex] for key2 in outputLayer])
                    if j == 0:
                        outputVec0 = outputLayers[np.newaxis, :]
                    else:
                        outputVec0 = np.vstack((outputVec0, outputLayers))
        
            if i == 0:
                X  = inputVec0.reshape(inputVec0.shape[0], 1, inputVec0.shape[1])
                Y  = outputVec0
            else: 
                inputVec0       = inputVec0.reshape(inputVec0.shape[0], 1, inputVec0.shape[1])
                X               = np.vstack((X, inputVec0))
                Y               = np.vstack((Y, outputVec0))
    else:
        
      inputVec0  = []
      
      for j in range(0, nStepsTotal- td):
          startIndex0 = j
          startIndex1 = td+StepPredictor+j-1
  
          inputLayers = np.hstack([input_data[key1][0, startIndex0:startIndex1] for key1 in inputLayer])
          if j == 0:
              inputVec0 = inputLayers[np.newaxis, :]
          else:
              inputVec0 = np.vstack((inputVec0, inputLayers))
             
      Y = []
      X  = inputVec0
    
    return [X, Y]

#####################################################
def NeuralNetworkStructureTypes():
    return [' ','L', 'LL', 'RL', 'LR','R', 'RR', 'RRR','TLTLT','EEE', 'SL', 'LS' , 'LSL', #12
            'SLSLS', 'SLS', 'TSTST','TSLTSLT','TLSLT','LTTL','TLT', 'LLR', #18, TLSLTLSLT
            'LLRL', 'LLLRL', 'LLRLRL', 'LLLL', 'LRdL','RLR','RSR','LRL', #25
                  'RdR', 'LRdLRdL', 'LLLRLRL','TdTdTdTdTdT'] #29


def BasicNeuralNetwork(typeName):
    if typeName == 'RNN':
        return nn.RNN
    elif typeName == 'FFN':
        return nn.Linear
    elif typeName == 'LSTM': #not ready; needs extension for (h0,c0) hidden state
        return nn.LSTM
    raise ValueError('BasicNeuralNetwork: invalid type: ', typeName)
    return None

def BasicNeuralNetworkNames():
    return ['RNN', 'FFN', 'LSTM']


variableShortForms = {'maxEpochs':'EP', 'learningRate':'LR', 'nTraining':'NT',
                      'batchSize':'BS', 'hiddenLayerSize':'HL', 'hiddenLayerStructureType':'NN'}

def VariableShortForm(var):
    if var in variableShortForms:
        return variableShortForms[var]
    else:
        return var
    

#%%+++++++++++++++++++++++++++
def ExtendResultsFile(
        resultsFile, infoDict, parameterDict={}):
    with open(resultsFile, "r") as f: 
        contents = f.readlines()

    contentsNew = contents[0:3]
    contentsNew += ['#info:'+str(infoDict).replace(' ','').replace('\n','\\n')+'\n']
    contentsNew += ['#params:'+str(parameterDict).replace(' ','').replace('\n','\\n')+'\n']
    contentsNew += contents[4:]

    with open(resultsFile, "w") as f:
        for line in contentsNew:
            f.write(line)
    
#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++
#NeuralNetwork helper class
class MyNeuralNetwork(nn.Module):
    #computeDevice =['cpu' | 'cuda']
    def __init__(self, inputSize, outputSize, 
                 hiddenLayerSize, hiddenLayerStructure,
                 neuralNetworkTypeName = 'RNN', computeDevice = 'cpu', 
                 rnnNumberOfLayers = 1,
                 rnnNonlinearity = 'tanh', #'relu', None; 'tanh' is the default
                 ):
        super().__init__()
        self.computeDevice = torch.device(computeDevice)
        # print('torch NN device=',computeDevice)
        self.hiddenLayerSize = hiddenLayerSize
        self.hiddenLayerStructure = hiddenLayerStructure
        self.rnnNumberOfLayers = rnnNumberOfLayers
        self.initializeHiddenStates = []
        self.neuralNetworkTypeName = neuralNetworkTypeName
        self.neuralNetworkBase = BasicNeuralNetwork(neuralNetworkTypeName)
        self.rnnNonlinearity = rnnNonlinearity
        
        self.hiddenLayersList = nn.ModuleList([]) #this correctly registers list of modules!!!

        if self.neuralNetworkTypeName == 'RNN' or self.neuralNetworkTypeName == 'LSTM':
            self.rnn = self.neuralNetworkBase(inputSize, hiddenLayerSize, 
                                              batch_first=True,#batch_first=True means that batch dimension is first one in input/output vectors
                                              num_layers=self.rnnNumberOfLayers,
                                              nonlinearity=self.rnnNonlinearity)#.to(self.computeDevice) #take output of first layer as input of second RNN
        elif self.neuralNetworkTypeName == 'FFN':
            self.rnn = self.neuralNetworkBase(inputSize, hiddenLayerSize)
            #self.rnn = self.rnn.to(self.computeDevice)
        else:
            raise ValueError('MyNeuralNetwork: invalid neuralNetworkTypeName:', self.neuralNetworkTypeName)

        for c in self.hiddenLayerStructure:
            if c.upper() == 'L':
                self.hiddenLayersList.append(nn.Linear(hiddenLayerSize, hiddenLayerSize))
                #self.hiddenLayersList.append(nn.BatchNorm1d(hiddenLayerSize)) 
            elif c.upper() == 'R':
                self.hiddenLayersList.append(nn.ReLU())
            elif c.upper() == 'E':
                self.hiddenLayersList.append(nn.ELU())
            
            elif c.upper() == 'T':
                self.hiddenLayersList.append(nn.Tanh())    
                    
            elif c.upper() == 'S':
                self.hiddenLayersList.append(nn.Sigmoid())
            elif c == 'D':
                self.hiddenLayersList.append(nn.Dropout(0.5))
            elif c == 'd':
                self.hiddenLayersList.append(nn.Dropout(0.1))
            elif c == ' ':
               continue
            else:
                raise ValueError('MyNeuralNetwork: invalid layer type: '+c)
                

        self.hiddenLayersList.append(nn.Linear(hiddenLayerSize, outputSize))

    def SetInitialHiddenStates(self, hiddenStatesBatch):
        if self.neuralNetworkTypeName == 'RNN':
            self.initializeHiddenStates = hiddenStatesBatch


    def forward(self, x):
        if self.neuralNetworkTypeName == 'RNN' or self.neuralNetworkTypeName == 'LSTM':
            # print('x=', x.size())
            batchSize = x.size(0)
            hidden = self.initialize_hidden_state(batchSize).to(self.computeDevice)
            #hidden = self.initialize_hidden_state(batchSize).to(x.get_device())
            out, _ = self.rnn(x, hidden)
            out = out.contiguous().view(-1, self.hiddenLayerSize)
        elif self.neuralNetworkTypeName == 'FFN':
            out = self.rnn(x)
            
        for item in self.hiddenLayersList:
            out = item(out)

        #out = self.lastLayer(out)
        return out

    # def forward(self, x, hidden):
    #     out, _ = self.rnn(x, hidden)
    #     out = out.contiguous().view(-1, self.hiddenLayerSize)

    #     for item in self.hiddenLayersList:
    #         out = item(out)

    #     out = self.lastLayer(out)
    #     return out

    def initialize_hidden_state(self, batchSize):
        hs = torch.zeros((self.rnnNumberOfLayers, batchSize, self.hiddenLayerSize)).to(self.computeDevice)
        if self.neuralNetworkTypeName == 'RNN' and self.initializeHiddenStates.size(1) != 0:
            nInitAvailable = self.initializeHiddenStates.size(1)
            hs[0,:,:nInitAvailable] = self.initializeHiddenStates

        # for i, hsInit in enumerate(self.initializeHiddenStates):
        #     hs[0,:,i] = hsInit
        return hs

moduleNntc = None #must be set in calling module

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++
#parameter function for training with exudyn ParameterVariation(...)
def ParameterFunctionTraining(parameterSet):
    global moduleNntc
        
    #++++++++++++++++++++++++++++++++++++++++++++++
    #++++++++++++++++++++++++++++++++++++++++++++++
    #store default parameters in structure (all these parameters can be varied!)
    class P: pass #create emtpy structure for parameters; simplifies way to update parameters

    #default values
    P.maxEpochs=1000 #sufficient
    P.learningRate=0.001
    P.lossThreshold=1e-8
    P.batchSize=64
    P.neuralNetworkType = 0 #[0=RNN, 1=FFN]
    P.hiddenLayerSize=128
    P.hiddenLayerStructureType=0 #'L'
    P.nTraining = 512
    P.nTest = 20
    P.testEvaluationInterval = 0 #number of epochs after which tests are evaluated; 0=no evaluation
    P.lossLogInterval = 20
    P.computationIndex = None
    P.storeModelName = ''
    P.dataFile = None
    P.rnnNonlinearity = 'tanh'
    P.dataLoaderShuffle = False
    P.case = 0 #
    P.sensorType = -1 
    P.InputLayer = ['U1', 'U2', 's1', 's2']
    P.stepPredictor = 1
    P.LyaerUnits = 200
    P.system = 'system'
    P.SLIDESteps= 29
    P.nStepsTotal = 200
    P.OutputLayer = ['deltaY']
    
    
    # #now update parameters with parameterSet (will work with any parameters in structure P)
    for key,value in parameterSet.items():
        setattr(P,key,value)
    
    #functionData are some values that are not parameter-varied but may be changed for different runs!
    if 'functionData' in parameterSet:
        for key,value in P.functionData.items():
            if key in parameterSet:
                print('ERROR: duplication of parameters: "'+key+'" is set BOTH in functionData AND in parameterSet and would be overwritten by functionData; Computation will be stopped')
                raise ValueError('duplication of parameters')
            setattr(P,key,value)

    hiddenLayerStructure = NeuralNetworkStructureTypes()[int(P.hiddenLayerStructureType)] #'L'
    neuralNetworkTypeName = BasicNeuralNetworkNames()[P.neuralNetworkType]
  
    #++++++++++++++++++++++++++++++++++++++++++++++
    #++++++++++++++++++++++++++++++++++++++++++++++
    if P.dataFile == None:
        moduleNntc.CreateTrainingAndTestData(nTraining=P.nTraining, nTest=P.nTest,
                                       #parameterFunction=PVCreateData, #for multiprocessing
                                       )
    else:
        #print(P.hiddenLayerSize)system=P.system
        moduleNntc.LoadTrainingAndTestsData(P.dataFile, nTraining=P.nTraining, nTest=P.nTest,
                                            nStepsTotal= P.nStepsTotal, 
                                            InputLayer= P.InputLayer, OutputLayer=P.OutputLayer,
                                            StepPredictor=P.stepPredictor, #Units=P.LyaerUnits,
                                            SLIDESteps=P.SLIDESteps)
        
            
        
    moduleNntc.TrainModel(maxEpochs=P.maxEpochs, learningRate=P.learningRate, 
                    lossThreshold=P.lossThreshold, batchSize=P.batchSize,
                    neuralNetworkTypeName = neuralNetworkTypeName,
                    hiddenLayerSize=P.hiddenLayerSize, 
                    hiddenLayerStructure=hiddenLayerStructure,
                    testEvaluationInterval=P.testEvaluationInterval,
                    lossLogInterval=P.lossLogInterval,
                    rnnNonlinearity=P.rnnNonlinearity,
                    dataLoaderShuffle=P.dataLoaderShuffle,
                    seed = P.case)

    rv = moduleNntc.EvaluateModel()
    rv['testResultsEpoch'] = moduleNntc.TestResults()[0]
    rv['testResults'] = moduleNntc.TestResults()[1]
    rv['testResultsMin'] = moduleNntc.TestResults()[2]
    rv['testResultsMean'] = moduleNntc.TestResults()[3]
    rv['testResultsMax'] = moduleNntc.TestResults()[4]
    rv['testResultsMae'] = moduleNntc.TestResults()[5]
    
    
    rv['trainResultsEpoch'] = moduleNntc.TrainResults()[0]
    rv['trainResults'] = moduleNntc.TrainResults()[1]
    rv['trainResultsMin'] = moduleNntc.TrainResults()[2]
    rv['trainResultsMean'] = moduleNntc.TrainResults()[3]
    rv['trainResultsMax'] = moduleNntc.TrainResults()[4]
    rv['trainResultsMae'] = moduleNntc.TrainResults()[5]
    
    if P.storeModelName!='':
        moduleNntc.SaveNNModel(P.storeModelName+str(P.computationIndex))
    return rv #dictionary


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, predicted, target):
        return torch.sqrt(self.mse(predicted, target))
    
    
class network_training_center():
    #create class with model to be trained
    #createModelDict is used to initialize CreateModel function with specific parameters
    #initialize seed
    def __init__(self,nnModel=NNtestModel(), computeDevice = 'cpu', verboseMode=1):
        
        self.nnModel = nnModel
        self.verboseMode = verboseMode


        #initialize for pure loading of model:
        self.computeDevice = computeDevice
        self.lossFunction = nn.MSELoss() #mean square error, equals: np.linalg.norm(x-y)**2/len(x)
        
        #self.lossFunction = nn.L1Loss()
        # self.lossFunction = RMSELoss() 
         
        self.inputsTraining = []
        self.targetsTraining = []
        self.inputsTest = []
        self.targetsTest = []

        #initialization of hidden layers in RNN (optional)
        self.hiddenInitTraining = []
        self.hiddenInitTest = []

        self.floatType = torch.float #16, 32, 64
        
    
    #get stored NN-mbs model
    def GetNNModel(self):
        return self.nnModel
    
    def ScaleTrainingAndTestsData(self, fileName,Data=None,InputLayer=None, OutputLayer=None,                                   #system=None, #nStepsTotal=None,
                                       Evaluate=None, file = None ):
        
        Xtrain,Ytrain,Xtest,Ytest      = {},{},{},{}
        XtrainS,YtrainS,XtestS,YtestS  = {},{},{},{}

        #1. Loading the data
        if not Evaluate:
            fileExtension = ''
            if len(fileName) < 4 or fileName[-4:]!='.npy':
                fileExtension = '.npy'
            with open(fileName+fileExtension, 'rb') as f:
                dataDict = np.load(f, allow_pickle=True).item() 
        else:
            dataDict = Data
        
        if not Evaluate:
            nTraining  = dataDict['nTraining']
            nTest      =  dataDict['nTest']
            
        #2. Finding  Xtrain,Ytrain,Xtest,Ytest from the data.
        for key in InputLayer:    
            if not Evaluate:
               Xtrain[key]     = np.stack([dataDict['inputsTraining'][i][key] for i in range(nTraining)])
               Xtest[key]      = np.stack([dataDict['inputsTest'][i][key] for i in range(nTest)])
            else:
               Xtrain[key]     = np.stack([dataDict[0][key]])
               Xtest[key]      = []
        
        for key in OutputLayer:
            if not Evaluate:
                Ytrain[key]     = np.stack([dataDict['targetsTraining'][i][key] for i in range(nTraining)]) 
                Ytest[key]     = np.stack([dataDict['targetsTest'][i][key] for i in range(nTest)])
            else:
                Ytrain[key]     = np.stack([dataDict[1][key]]) 
                Ytest[key]      = []
    
        #3. Calculating  scaling factors
        if not Evaluate:
            InputScalingFactors  = {}
            for key in InputLayer:
                min_val = Xtrain[key].min()
                max_val = Xtrain[key].max()
                InputScalingFactors[key] = {"min": min_val - 0.02 * abs(min_val),"max": max_val + 0.02 * abs(max_val)}
      
            np.save(fileName + 'InputScalingFactors.npy', InputScalingFactors)
        else:
            InputScalingFactors = np.load(file + 'InputScalingFactors.npy', allow_pickle=True).item()
            
        if not Evaluate:
            OutputScalingFactors = {}
            for key in OutputLayer:
                min_val = Ytrain[key].min()
                max_val = Ytrain[key].max()
                OutputScalingFactors[key] = {"min": min_val - 0.02 * abs(min_val),"max": max_val + 0.02 * abs(max_val)}
           
            np.save(fileName + 'OutputScalingFactors.npy', OutputScalingFactors)
        else:
            OutputScalingFactors = np.load(file + 'OutputScalingFactors.npy', allow_pickle=True).item()

        
        #4. Scaling the data
        for key in InputLayer:    
            if key in ['s1', 's2', 'p1', 'p2', 'p3', 'p4']:
                XtrainS[key] = ScaleZeroToOne(Xtrain[key], InputScalingFactors[key]['min'], InputScalingFactors[key]['max'])
                XtestS[key]  = ScaleZeroToOne(Xtest[key], InputScalingFactors[key]['min'], InputScalingFactors[key]['max'])
            else:
                if key in ['U1', 'U2']:
                    XtrainS[key] = Xtrain[key]
                    XtestS[key]  = Xtest[key]
                else:
                    XtrainS[key] = ScaleMinusOneToOne(Xtrain[key], InputScalingFactors[key]['min'], InputScalingFactors[key]['max'])
                    XtestS[key] = ScaleMinusOneToOne(Xtest[key], InputScalingFactors[key]['min'], InputScalingFactors[key]['max'])

        for key in OutputLayer:
              YtrainS[key]         = ScaleMinusOneToOne(Ytrain[key],OutputScalingFactors[key]['min'],OutputScalingFactors[key]['max'] ) 
              YtestS[key]          = ScaleMinusOneToOne(Ytest[key],OutputScalingFactors[key]['min'],OutputScalingFactors[key]['max'] )  
        
        return XtrainS,YtrainS, XtestS, YtestS 
        


    #load data from .npy file
    #allows loading data with less training or test sets
    # fileName,Data=None,system=None, nTraining=None, nTest=None, nStepsTotal=None,Input=[], Output=[], StepPredictor=None, Units=None,
                                # SLIDESteps=None)
    def LoadTrainingAndTestsData(self, fileName,Data=None, nTraining=None, nTest=None,nStepsTotal=None, #system=None, #nStepsTotal=None,
                                 InputLayer=[], OutputLayer=[], StepPredictor=None, Units=None,  SLIDESteps=None, 
                                 Evaluate=None, file=None):
        
        #SLIDESteps:User defined steps.
        #fileName: data file name
        #nTraining: Number of training set
        #nTest: Number of validation set
        #system: LiftBoom/Patu
        #InputLayer: Measurements from the system
        #OutputLayer: Output sensors
        #StepPredictor: Steps to be predicted.
        
        self.inputsTraining     = []
        self.targetsTraining    = []
        self.hiddenInitTraining = []
        self.inputsTest         = []
        self.targetsTest        = []
        self.hiddenInitTest     = []
        
        if not Evaluate: 
            fileExtension = ''
            if len(fileName) < 4 or fileName[-4:]!='.npy':
                fileExtension = '.npy'
                
            with open(fileName+fileExtension, 'rb') as f:
                dataDict = np.load(f, allow_pickle=True).item()   #allow_pickle=True for lists or dictionaries; .all() for dictionaries
                
            if dataDict['version'] >= 1:
                if nTraining == None:
                    nTraining = dataDict['nTraining']
                if nTest == None:
                    nTest = dataDict['nTest']
                
                if StepPredictor == None:
                     nPredictor   = Data['parameters']['stepPredictor']
                     nPredictor   = int(nPredictor[0])
                     StepPredictor = nPredictor 
                
                if dataDict['nTraining'] < nTraining:
                    raise ValueError('NeuralNetworkTrainingCenter.LoadTrainingAndTestsData: available training sets ('+
                                     str(dataDict['nTraining'])+') are less than requested: '+str(nTraining))
                if dataDict['nTest'] < nTest:
                    raise ValueError('NeuralNetworkTrainingCenter.LoadTrainingAndTestsData: available test sets ('+
                                     str(dataDict['nTest'])+') are less than requested: '+str(nTest))
    
                self.nTraining = nTraining
                self.nTest = nTest
                self.nStepsTotal =nStepsTotal
                
                XtrainS,YtrainS, XtestS, YtestS  = self.ScaleTrainingAndTestsData(fileName,Data, InputLayer, OutputLayer,
                                                                                          Evaluate=Evaluate, file=file)
                    
        else:
                self.nTraining   = 1
                self.nTest       = None
                self.nStepsTotal =nStepsTotal
                
                XtrainS,YtrainS, XtestS, YtestS  = self.ScaleTrainingAndTestsData(fileName,Data, InputLayer, OutputLayer,
                                                                                          Evaluate=Evaluate, file=file)
        
        # Steps from damped oscillations
        self.SLIDESteps = SLIDESteps
        print(f'Data arrangement using SLIDE window={self.SLIDESteps}')
          #Data points devision division based SLIDESteps
          
        self.inputsTraining, self.targetsTraining = SLIDE_Data(self.nStepsTotal, self.nTraining, self.SLIDESteps, 
                                                                  XtrainS,YtrainS,StepPredictor,InputLayer,
                                                                  OutputLayer,  Evaluate=Evaluate)  
        if not Evaluate:
            self.inputsTest, self.targetsTest     = SLIDE_Data(self.nStepsTotal, self.nTest, self.SLIDESteps, 
                                                                 XtestS, YtestS,StepPredictor,InputLayer,OutputLayer)
            
            self.hiddenInitTraining = np.zeros((self.inputsTraining.shape[0], 1))
            self.hiddenInitTest     = np.zeros((self.targetsTest.shape[0], 1))
            
        else:
            self.inputsTest, self.targetsTest    =  [],[]
        
        return [self.inputsTraining,self.targetsTraining, self.inputsTest, self.targetsTest] 
            # # Training data
            # for i in range(self.nTraining): 
                   
                        

        # print('inputs shape2=', self.inputsTraining[0].shape)

    def AdjustInputsToNN(self, neuralNetworkTypeName):
        #for FFN, we need flat input vectors:
        if neuralNetworkTypeName == 'FFN':
            #print('adjust inputs to FFN')
            dataFFN = []
            for v in self.inputsTraining:
                dataFFN += [v.flatten()]

            self.inputsTraining = np.stack(dataFFN, axis=0)

            dataFFN = []
            for v in self.inputsTest:
                dataFFN += [v.flatten()]

            self.inputsTest = np.stack(dataFFN, axis=0)


    #create training data
    def TrainModel(self, maxEpochs = 1000, lossThreshold=1e-7,
                   learningRate = 0.001, batchSize = 32,
                   hiddenLayerSize = 64, 
                   hiddenLayerStructure = 'L',
                   neuralNetworkTypeName = 'RNN',
                   testEvaluationInterval = 0,
                   seed = 0,
                   lossLogInterval = 50,
                   epochPrintInterval = 500,
                   rnnNumberOfLayers = 1,
                   rnnNonlinearity = 'tanh',
                   dataLoaderShuffle = False,
                   batchSizeIncrease = 1,           #can be used to increase batch size after initial slower training
                   batchSizeIncreaseLoss = None,    #loss threshold, at which we switch to (larger) batch size
                   reloadTrainedModel = '',         #load (pre-)trained model after setting up optimizer
                   ):

        #%%++++++++++++++++++++++++++++++++++++++++
        self.maxEpochs = int(maxEpochs)
        self.lossThreshold = lossThreshold
        self.learningRate = learningRate
        self.batchSize = int(batchSize)

        self.neuralNetworkTypeName = neuralNetworkTypeName
        
        self.hiddenLayerSize = int(hiddenLayerSize)
        self.hiddenLayerStructure = hiddenLayerStructure
        self.rnnNumberOfLayers = rnnNumberOfLayers
        self.rnnNonlinearity = rnnNonlinearity
        self.testEvaluationInterval = int(testEvaluationInterval)
        self.dataLoaderShuffle = dataLoaderShuffle

        self.lossLogInterval = int(lossLogInterval)
        self.seed = int(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(self.computeDevice != 'cuda')
        np.random.seed(seed)
        random.seed(seed)
        
        self.floatType = torch.float #16, 32, 64
        
        
        #%%++++++++++++++++++++++++++++++++++++++++
        #[inputSize, outputSize] = self.nnModel.GetInputOutputSizeNN()
        
        inputSize =self.inputsTraining.shape[2]
        outputSize=self.targetsTraining.shape[1]
        
        self.rnn = MyNeuralNetwork(inputSize = inputSize, 
                                   outputSize = outputSize, 
                                   neuralNetworkTypeName = self.neuralNetworkTypeName,
                                   hiddenLayerSize = self.hiddenLayerSize,
                                   hiddenLayerStructure = self.hiddenLayerStructure,
                                   computeDevice=self.computeDevice, 
                                   rnnNumberOfLayers=self.rnnNumberOfLayers,
                                   rnnNonlinearity=self.rnnNonlinearity)
        
        if self.floatType == torch.float64:
            self.rnn = self.rnn.double()
        if self.floatType == torch.float16:
            self.rnn = self.rnn.half() #does not work in linear/friction
        
        self.rnn = self.rnn.to(self.computeDevice)
        #adjust for FFN
        self.AdjustInputsToNN(neuralNetworkTypeName)
        
        #Training data 
        inputs = torch.tensor(self.inputsTraining, dtype=self.floatType, requires_grad=True).to(self.computeDevice)#,non_blocking=True)        
        targets = torch.tensor(self.targetsTraining, dtype=self.floatType, requires_grad=True).to(self.computeDevice)#,non_blocking=True)
        hiddenInit = torch.tensor(self.hiddenInitTraining, dtype=self.floatType, requires_grad=True).to(self.computeDevice)#,non_blocking=True)
        
        #Validation data 
        inputsTest = torch.tensor(self.inputsTest, dtype=self.floatType).to(self.computeDevice)
        targetsTest = torch.tensor(self.targetsTest, dtype=self.floatType).to(self.computeDevice)
        hiddenInitTest = torch.tensor(self.hiddenInitTest, dtype=self.floatType,requires_grad=True).to(self.computeDevice)
        
        # Convert your data to PyTorch tensors and create a DataLoader
        dataset = TensorDataset(inputs.requires_grad_(), targets.requires_grad_(), hiddenInit.requires_grad_())
        dataloader = DataLoader(dataset, batch_size=batchSize, 
                                shuffle=self.dataLoaderShuffle)
        if batchSizeIncrease != 1:
            dataloader2 = DataLoader(dataset, batch_size=batchSize*batchSizeIncrease,
                                    shuffle=self.dataLoaderShuffle)

        datasetTest = TensorDataset(inputsTest, targetsTest, hiddenInitTest)
        dataloaderTest = DataLoader(datasetTest, batch_size=batchSize, shuffle=self.dataLoaderShuffle)

        
        # Define a loss function and an optimizer
        optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.learningRate)        
        self.lossLog = []
        self.trainResults = [[], [], [], [], [],[]] #epoch, tests, min, mean, max, mae
        self.testResults = [[], [], [], [], [],[]] #epoch, tests, min, mean, max, mae
                
        if reloadTrainedModel!='': #this may cause problems with cuda!
            #self.rnn.rnn.load_state_dict(torch.load(reloadTrainedModelDict)) #this does not work
            self.LoadNNModel(reloadTrainedModel)
            self.rnn = self.rnn.to(self.computeDevice)

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # MAIN TRAINING LOOP
        tStart = time.time()
        trainresults = []
        trainMAE = []
        for epoch in range(self.maxEpochs):  
            self.rnn.train()
            for i, (inputs, targets, initial_hidden_states) in enumerate(dataloader):
                # Forward pass
                optimizer.zero_grad()
                
                outputs = self.rnn(inputs)
                loss = self.lossFunction(outputs, targets)
                                
                maeTrain = torch.mean(torch.abs(outputs - targets)).item()
                maeTrain  = float(maeTrain)
        
                # Backward pass and optimization
                optimizer.zero_grad() #(set_to_none=True)
                loss.backward()
                optimizer.step()

            currentLoss = loss.item()
            trainresults += [float(loss.detach())]

            #switch to other dataloader:
            if batchSizeIncreaseLoss != None and currentLoss < batchSizeIncreaseLoss:
                dataloader = dataloader2

            #log loss at interval
            if (epoch % self.lossLogInterval == 0) or (epoch == self.maxEpochs-1):
                self.lossLog += [[epoch, currentLoss]]
            
            self.trainResults[0] += [epoch]
            self.trainResults[1] += [trainresults]
            self.trainResults[2] += [min(trainresults)]
            self.trainResults[3] += [currentLoss]
            self.trainResults[4] += [max(trainresults)]
            self.trainResults[5] += [maeTrain]
            
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #perform tests after interval
            if testEvaluationInterval > 0 and (epoch > 0 
                                               and epoch%testEvaluationInterval == 0
                                               or epoch == self.maxEpochs-1
                                               or currentLoss < self.lossThreshold):
                self.rnn.eval() #switch network to evaluation mode
                testresults = []
                with torch.no_grad(): #avoid calculation of gradients:
                    for inputs, targets, initial_hidden_states in dataloaderTest:
                        # Forward pass
                        self.rnn.SetInitialHiddenStates(initial_hidden_states)
                        outputs = self.rnn(inputs)
                        mse =  self.lossFunction(outputs, targets)
                        maeTest = torch.mean(torch.abs(outputs - targets))
                       
                        
                        testresults += [float(mse.detach())] #np.linalg.norm(y-yRef)**2/len(y)]
                        maeTest     = float(maeTest.detach())
                        
                    
                        #resultsMAE += []
                        
                self.testResults[0] += [epoch]
                self.testResults[1] += [testresults]
                self.testResults[2] += [min(testresults)]
                self.testResults[3] += [np.mean(testresults)]
                self.testResults[4] += [max(testresults)]
                self.testResults[5] += [maeTest]
                
                #switch back to training!
                self.rnn.train()
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            
            #print(epoch)
            #printing
            if (epoch+1) % epochPrintInterval == 0:
                tNow = time.time()
                #tLast = time.time()

                
                lossStr     = "{:.3e}".format(currentLoss)
                maeTrainStr = "{:.3e}".format(maeTrain)
                mseStr     = "{:.3e}".format(np.mean(testresults))
                testMAEStr = "{:.3e}".format(maeTest)
                
                
                if self.verboseMode > 0:
                    print(f'Epoch {epoch+1}/{self.maxEpochs}, Loss: {lossStr}, MAE:{maeTrainStr}, Val loss: {mseStr}, Val MAE:{testMAEStr} ',end='')
                    print('; t:',round(tNow-tStart,1),'/',round((tNow-tStart)/epoch*self.maxEpochs,0))

            if currentLoss < self.lossThreshold:
                if self.verboseMode > 0:
                    print('iteration stopped at', epoch,'due to tolerance; Loss:', currentLoss)
                break
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        self.trainingTime = time.time()-tStart 
        if self.verboseMode > 0:
            print('training time=',self.trainingTime)

    #return loss recorded by TrainModel
    def LossLog(self):
        return np.array(self.lossLog)

    #return test results recorded by TrainModel
    def TestResults(self):
        return self.testResults

    def TrainResults(self):
        return self.trainResults

    #save trained model, which can be loaded including the structure (hidden layers, etc.)
    def SaveNNModel(self, fileName):
        os.makedirs(os.path.dirname(fileName), exist_ok=True)
        torch.save(self.rnn, fileName) #save pytorch file

    #load trained model, including the structure (hidden layers, etc.)
    def LoadNNModel(self, fileName):
        self.rnn = torch.load(fileName, map_location=self.computeDevice,weights_only=False)
        self.rnn.eval()
        return self.rnn 

    #plot loss and tests over epochs, using stored data from .npy file
    #dataFileTT is the training and test data file name
    def PlotTrainingResults(self, resultsFileNpy=None, dataFileName=None, plotLoss=True, plotTests=True, 
                            plotTestsMaxError=True, plotTestsMeanError=True,
                            sizeInches=[12.8,12.8], Patu=True, nTrain=None,
                            Sensors=None ,Liftload = None, string=None,  closeAll=True, 
                            ):
      if resultsFileNpy:
            #load and visualize results:
            with open(resultsFileNpy, 'rb') as f:
                dataDict = np.load(f, allow_pickle=True).item()   #allow_pickle=True for lists or dictionaries; .all() for dictionaries
    
            #self.LoadTrainingAndTestsData(dataFileName, dataDict)
    
            # if closeAll:
            #     PlotSensor(None,closeAll=True)
    
            values = dataDict['values']
            parameters = dataDict['parameters']
            parameterDict = dataDict['parameterDict']
            #plot losses:
            labelsList = []
            
            markerStyles = []
            markerSizes = [6]
            
            cntColor = -1
            lastCase = 0
            
            import matplotlib.pyplot as plt
            a40_width_inches, a40_height_inches = 8.3 / 2, 11.7 / 4  
            fig, ax = plt.subplots(figsize=(a40_width_inches, a40_height_inches))
            
            for i, v in enumerate(values):
                #colors are identical for cases
                if 'case' in parameters:
                    case = parameterDict['case'][i]
                    markerStyles=[listMarkerStyles[case]]
                    
                    cntColor += 1
                    if case < lastCase or cntColor < 0:
                        cntColor += 1
                    lastCase = case
                else:
                    cntColor += 1
                 
                colors = ['black', 'red', 'green', 'blue', 'cyan']
                sLabel='var'+str(i)
                sep = ''
                
                #for j,par in enumerate(list(parameters)):
                   
                     #sLabel+=sep+VariableShortForm(par)+str(SmartRound2String(parameterDict[par][i],1))
                Label1= NeuralNetworkStructureTypes()[parameters['hiddenLayerStructureType'][i]]
                Label2 = parameters['hiddenLayerSize'][0]
                
                # for j in range(len(parameters['hiddenLayerSize'])):
                #         Label2 = parameters['hiddenLayerSize'][j]
             
                sLabel= f"Hidden layer:{Label1}, Units:{Label2}"
                
                        #sLabel= f"Units:{Label2}"
                
                sep=','
                
                #labelsList += [sLabel]
                dataTest = np.vstack((v['testResultsEpoch'],v['testResultsMean'])).T
                dataTrain = np.vstack((v['trainResultsEpoch'],v['trainResultsMean'])).T
                
                if plotLoss:
                            # PlotSensor(None, [dataTrain], xLabel='Number of epochs', yLabel='Loss (Log Scale)', labels=[sLabel],
                            #            newFigure=i==0, colorCodeOffset=cntColor, colors = [],
                            #            markerStyles=markerStyles, markerSizes=markerSizes, 
                            #            sizeInches=sizeInches,fileName = 'solution/Trainingloss_plot.PNG', logScaleY=True)
                        
                    
                            ax.plot(dataTrain[:, 0], dataTrain[:, 1], label=sLabel, color=colors[i])
                            ax.set_xlabel(r'Number of epochs',fontdict={'family': 'Times New Roman', 'size': 11})
                            ax.set_ylabel(r'Loss',fontdict={'family': 'Times New Roman', 'size': 11})
                            x_min, x_max = 0, 1000 
                            padding = x_max/20  
                            ax.set_xlim(x_min - padding, x_max + padding)
                    
                            # Set y-axis to logarithmic scale
                            ax.set_yscale('log')
                            ax.set_ylim(10**-8, 10**0)
                            # Define x-ticks and y-ticks
                            x_ticks = [0,      2*(x_max/4) ,x_max] #1*(x_max/4),3*(x_max/4)
                            y_ticks = [10**-8,  10**-4,     10**0]
                    
                            ax.set_xticks(x_ticks)
                            ax.set_yticks(y_ticks)
                    
                            # Set custom tick labels
                            ax.set_xticklabels([r'$0$', rf'${2*(x_max/4):.0f}$'
                                        , rf'${4*(x_max/4):.0f}$']) #rf'${x_max/4:.0f}$',rf'${3*(x_max/4):.0f}$'
                            # ax.set_yticklabels([r'$10^{-6}$',r'$10^{-4}$', r'$10^{-2}$',
                            #                     r'$10^{0}$'])
                    
                            ax.set_yticklabels([r'$10^{-8}$',r'$10^{-4}$', r'$10^{0}$'])
                    
                            # Customize tick parameters
                            ax.tick_params(axis='both', labelsize=11)
                            ax.grid(True)
                            ax.legend(fontsize=6, loc='upper right')
                            plt.tight_layout()
                            if Patu:
                                plt.savefig(f'solution/Figures/Results/Patu/Result_{Sensors}sensors_1280Samples_{Liftload}kg.svg', 
                                            format='png', dpi=300)
                            else:
                                plt.savefig(f'solution/Figures/Results/LiftBoom/{Liftload} kg/Result_{nTrain}trainingsamples_{string}.pdf',
                                                format='pdf',bbox_inches='tight')
                                
                        
                            plt.show()
                
      else:
        import matplotlib.pyplot as plt
        a40_width_inches, a40_height_inches = 8.3 / 2, 11.7 / 4  
        fig, ax = plt.subplots(figsize=(a40_width_inches, a40_height_inches))
        
        ax.plot(self.trainResults[0], self.trainResults[3], label="Training")
        ax.plot(self.testResults[0], self.testResults[3], label="Valicdation")

        ax.set_xlabel(r'Number of epochs',fontdict={'family': 'Times New Roman', 'size': 11})
        ax.set_ylabel(r'Loss',fontdict={'family': 'Times New Roman', 'size': 11})
        x_min, x_max = 0, 1000 
        padding = x_max/20  
        ax.set_xlim(x_min - padding, x_max + padding)

        # Set y-axis to logarithmic scale
        ax.set_yscale('log')
        ax.set_ylim(10**-7, 10**0)
        # Define x-ticks and y-ticks
        x_ticks = [0,      2*(x_max/4) ,x_max] #1*(x_max/4),3*(x_max/4)
        y_ticks = [10**-7,  10**-4,     10**0]

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        # Set custom tick labels
        ax.set_xticklabels([r'$0$', rf'${2*(x_max/4):.0f}$'
                    , rf'${4*(x_max/4):.0f}$']) #rf'${x_max/4:.0f}$',rf'${3*(x_max/4):.0f}$'
        # ax.set_yticklabels([r'$10^{-6}$',r'$10^{-4}$', r'$10^{-2}$',
        #                     r'$10^{0}$'])

        ax.set_yticklabels([r'$10^{-7}$',r'$10^{-4}$', r'$10^{0}$'])

        # Customize tick parameters
        ax.tick_params(axis='both', labelsize=11)
        ax.grid(True)
        ax.legend(fontsize=6, loc='upper right')
        plt.tight_layout()
        if Patu:
            plt.savefig(f'solution/Figures/Results/Patu/Result_{Sensors}sensors_1280Samples_{Liftload}kg.svg', 
                        format='png', dpi=300)
        else:
            plt.savefig(f'solution/Figures/Results/LiftBoom/{Liftload} kg/Result_{nTrain}trainingsamples_{string}.pdf',
                            format='pdf',bbox_inches='tight')
            
    
        plt.show()              
                
               
    #evaluate all training and test data; plot some tests
    #nTrainingMSE avoids large number of training evaluations for large data sets
    def EvaluateModel(self,plotTests=[], plotTrainings=[], plotVars=['time','ODE2'], plotInputs=[],
                      closeAllFigures=False, nTrainingMSE=64, measureTime=False,
                      saveFiguresPath='', figureEnding='.pdf'):
        
       self.rnn.eval() #switch network to evaluation mode
       mbs = self.nnModel.mbs
       
       if len(plotTests)+len(plotTrainings) and closeAllFigures:
           mbs.PlotSensor(closeAll=True)

       #[inputSize, outputSize] = self.nnModel.GetInputOutputSizeNN()
       inputSize =self.inputsTraining.shape[1]
       outputSize=self.targetsTraining.shape[1]
       # outputScaling = self.nnModel.GetOutputScaling()[0:outputSize]
       # inputScaling  = self.nnModel.GetInputScaling()[0:outputSize]
       
       #outputScaling = self.nnModel.GetOutputScaling().reshape(outputSize)
       #inputScaling  = self.nnModel.GetInputScaling().reshape(inputSize)
       nStepsTotal              = self.nnModel.nStepsTotal

       inputs = torch.tensor(self.inputsTraining, dtype=self.floatType).to(self.computeDevice)
       targets = torch.tensor(self.targetsTraining, dtype=self.floatType).to(self.computeDevice)
       inputsTest = torch.tensor(self.inputsTest, dtype=self.floatType).to(self.computeDevice)
       targetsTest = torch.tensor(self.targetsTest, dtype=self.floatType).to(self.computeDevice)

       hiddenInit = torch.tensor(self.hiddenInitTraining, dtype=self.floatType).to(self.computeDevice)
       hiddenInitTest = torch.tensor(self.hiddenInitTest, dtype=self.floatType).to(self.computeDevice)


       dataset = TensorDataset(inputs, targets, hiddenInit)
       dataloader = DataLoader(dataset, batch_size=1)
       datasetTest = TensorDataset(inputsTest, targetsTest, hiddenInitTest)
       dataloaderTest = DataLoader(datasetTest, batch_size=1)

       trainingMSE = []
       testMSE = []

       listDataInputs = {}
       saveFigures = (saveFiguresPath != '')
       nMeasurements = 0
       timeElapsed = 0.
       self.rnn.eval() #switch network to evaluation mode
       with torch.no_grad(): #avoid calculation of gradients:
           
           newFigure=True
           i = -1
           for inputs, targets, initial_hidden_states in dataloader:
               i+=1
               if i < nTrainingMSE or (i in plotTrainings):
                   self.rnn.SetInitialHiddenStates(initial_hidden_states)
                   if measureTime:
                       timeElapsed -= timer()
                       outputs = self.rnn(inputs)
                       timeElapsed += timer()
                       nMeasurements += 1
                   else:
                       outputs = self.rnn(inputs)
                       
                   # outputs = outputs.view(-1, *outputSize)  # Reshape the outputs to match the target shape
                   #outputs = outputs.view(-1, 63, 2)  # Reshape the outputs to match the target shape
                   trainingMSE += [float(self.lossFunction(outputs, targets))]
                   
               if i in plotTrainings:
                   y = np.array(outputs.tolist()[0])
                   yRef = self.targetsTraining[i:i+1][0]

                   y = y*outputScaling
                   yRef = yRef*outputScaling
       
                   data = self.nnModel.OutputData2PlotData(y)
                   dataRef = self.nnModel.OutputData2PlotData(yRef)
   
                   comp = [self.nnModel.PlotDataColumns()[plotVars[1]] - 1]#time=-1
                   compX = [self.nnModel.PlotDataColumns()[plotVars[0]] - 1]#time=-1

                   if plotInputs != []:
                       for inputSel in plotInputs:
                           if inputSel not in listDataInputs:
                               listDataInputs[inputSel] = {}
                               # print('inputSel created:',inputSel)
                           yiRef = self.inputsTraining[i:i+1][0]
                           
                           yiData = self.nnModel.SplitInputData(yiRef)[inputSel]
                           dataInput = np.vstack((data[:,0].T,yiData)).T

                           listDataInputs[inputSel][i] = [dataInput, 'input train'+str(i), plotVars[0]]

                       
                   fileName = ''
                   #print('plotTrainings',plotTrainings,i,saveFigures)
                   if len(plotTrainings) > 0 and i==plotTrainings[-1] and saveFigures:
                       fileName = saveFiguresPath+plotVars[1]+'Training'+figureEnding

                   mbs.PlotSensor(data, components=comp, componentsX=compX, newFigure=newFigure,
                                   labels='NN train'+str(i), xLabel = plotVars[0], yLabel=plotVars[1],
                                   colorCodeOffset=i)
                   mbs.PlotSensor(dataRef, components=comp, componentsX=compX, newFigure=False,
                                   labels='Ref train'+str(i), xLabel = plotVars[0], yLabel=plotVars[1],
                                   colorCodeOffset=i,lineStyles=[':'],
                                   fileName=fileName)

                   newFigure = False

           # print(listDataInputs), 
           for keySel, listInputSels in listDataInputs.items():
               newFigure = True
               print('figure for ',keySel)
               cci=0
               for key, value in listInputSels.items():
                   
                   plotDataX   = value[0][:,0]
                   plotDataY   = value[0][:,1]*inputScaling[key*nStepsTotal:(key+1)*nStepsTotal]
                   plotData=value
                   datas= np.column_stack((plotDataX, plotDataY))
                   
                   # listDataInputs[i][inputSel] = [dataInput, 'input train'+str(i), plotVars[0]]
                   mbs.PlotSensor(plotData[0], components=[0], newFigure=newFigure,
                           labels=plotData[1], xLabel = plotData[2], yLabel=keySel,
                           colorCodeOffset=cci)
                   newFigure = False
                   cci+=1
       
           newFigure=True
   
           i = -1
           for inputs, targets, initial_hidden_states in dataloaderTest:
               i+=1
               self.rnn.SetInitialHiddenStates(initial_hidden_states)
               outputs = self.rnn(inputs)
               #outputs = outputs.view(-1, *outputSize)  # Reshape the outputs to match the target shape
               testMSE += [float(self.lossFunction(outputs, targets))]
                   
               if i in plotTests:
                   y = np.array(outputs.tolist()[0])
                   yRef = self.targetsTest[i:i+1][0]
                   
                   y = y*outputScaling
                   yRef = yRef*outputScaling
       
                   data = self.nnModel.OutputData2PlotData(y)
                   dataRef = self.nnModel.OutputData2PlotData(yRef)
   
                   # print('data=', data)
                   # print('dataRef=', dataRef)
                   comp = [self.nnModel.PlotDataColumns()[plotVars[1]] - 1]#time=-1
                   compX = [self.nnModel.PlotDataColumns()[plotVars[0]] - 1]#time=-1
                       
                   fileName = ''
                   #print('plotTests',plotTests,i,saveFigures)
                   if len(plotTests) > 0 and i==plotTests[-1] and saveFigures:
                       fileName = saveFiguresPath+plotVars[1]+'Test'+figureEnding

                   mbs.PlotSensor(data, components=comp, componentsX=compX, newFigure=newFigure,
                                   labels='NN test'+str(i), xLabel = plotVars[0], yLabel=plotVars[1],
                                   colorCodeOffset=i)
                   mbs.PlotSensor(dataRef, components=comp, componentsX=compX, newFigure=False,
                                   labels='Ref test'+str(i), xLabel = plotVars[0], yLabel=plotVars[1],
                                   colorCodeOffset=i,lineStyles=[':'],
                                   fileName=fileName)
                   newFigure = False

       if self.verboseMode > 0:
           print('max/mean test MSE=', max(testMSE), np.mean(testMSE))
           print('max training MSE=', max(trainingMSE))
       
       if measureTime and nMeasurements>0:
           print('forward evaluation total CPU time:', SmartRound2String(timeElapsed))
           print('Avg. CPU time for 1 evaluation:', SmartRound2String(timeElapsed/nMeasurements))

       return {'testMSE':testMSE, 'maxTrainingMSE':max(trainingMSE)}



    def Plotting(self,ns= None, SLIDESteps=None, Steps=None, Y_Estimation= None, data=None, InputLayer=None, 
                 OutputLayer=None,string=None,file=None):
        
        a4_width_inches, a4_height_inches = 11.7, 8.3 / 2  # Horizontal layout
        timeVecOut          = data[0]['t']
        fontSize1           = 12
        fontSize2           = 12
        tEnd                = timeVecOut[-1]
            
        InputScalingFactors = np.load(file + 'InputScalingFactors.npy', allow_pickle=True).item()
        OutputScalingFactors = np.load(file + 'OutputScalingFactors.npy', allow_pickle=True).item()
        # Figure dimensions
        
        Y_Scale = {}
        Y = {}
        timeVecOut = data[0]['t']
        
        for key in OutputLayer:
            # Scale back
            Y_Scale[key] = ScaleBackMinusOneToOne(Y_Estimation[key],
                                                 OutputScalingFactors[key]['min'],
                                                 OutputScalingFactors[key]['max'])
            
            fig, ax_main = plt.subplots(figsize=(8,6)) 
            
            # Set labels based on output type
            if key == 'deltaY':
                data1 = data[1][key] * 1000  # mm
                Y[key] = Y_Scale[key] * 1000
                ylabel_main = r'$\delta_\mathrm{y}$, mm'
                ylabel_error = 'Error, mm'
                ylabel_zoom = r'$\delta_\mathrm{y},\ \mathrm{mm}$'
                
                y_min  = OutputScalingFactors[key]['min']*1000
                y_max  = OutputScalingFactors[key]['max']*1000
                
            elif key == 'sig1':
                data1 = data[1][key]/1e6  # MPa
                Y[key] = Y_Scale[key]/1e6
                ylabel_main = r'$\sigma_{\mathrm{xx}}$, MPa'
                ylabel_error = 'Error, MPa'
                ylabel_zoom = r'$\sigma_{\mathrm{xx}}$, MPa'
                
                y_min  = OutputScalingFactors[key]['min']/1e6
                y_max  = OutputScalingFactors[key]['max']/1e6
                
            elif key == 'sig2':
                data1 = data[1][key]/1e6
                Y[key] = Y_Scale[key]/1e6
                ylabel_main = r'$\sigma_{\mathrm{xx}}$, MPa'
                ylabel_error = 'Error, MPa'
                ylabel_zoom = r'$\sigma_{\mathrm{xx}}$, MPa'
                y_min  = OutputScalingFactors[key]['min']/1e6
                y_max  = OutputScalingFactors[key]['max']/1e6
                
            elif key == 'eps1':
                data1 = data[1][key]*1e6  # micro-strain
                Y[key] = Y_Scale[key]*1e6
                ylabel_main = r'$\epsilon_{\mathrm{xy}}, \mu$'
                ylabel_error = r'Error, \mu'
                ylabel_zoom = r'$\epsilon_\mathrm{xy}, \mu$'
                
                y_min  = OutputScalingFactors[key]['min']*1e6
                y_max  = OutputScalingFactors[key]['max']*1e6
                
            elif key == 'eps2':
                data1 = data[1][key]*1e6
                Y[key] = Y_Scale[key]*1e6
                ylabel_main = r'$\epsilon_{\mathrm{xy}}, \mu$'
                ylabel_error = r'Error, \mu'
                ylabel_zoom = r'$\epsilon_\mathrm{xy}, \mu$'
                y_min  = OutputScalingFactors[key]['min']*1e6
                y_max  = OutputScalingFactors[key]['max']*1e6
                
            else:
                data1 = data[1][key]
                Y[key] = Y_Scale[key][0]
                ylabel_main =  fr'${key}$'
                ylabel_error = r'Error, \mu'
                ylabel_zoom = r'$\epsilon_\mathrm{xy}, \mu$'
                y_min  = OutputScalingFactors[key]['min']
                y_max  = OutputScalingFactors[key]['max']
                
                # print(f'Output "{key}" not added yet!')
                # continue
            
            mape = mean_absolute_percentage_error(data1[SLIDESteps+Steps:ns], Y[key].flatten()[SLIDESteps+Steps:ns])
            ax_main.plot(timeVecOut, data1, color='red', linestyle='-', label='Reference solution')
            ax_main.plot(timeVecOut, Y[key].flatten(), color='blue', linestyle=':', label='SLIDE estimations')
            ax_main.set_xlim(0, tEnd)  # keep full data range
            #ax_main.set_ylim(y_min, y_max)  # keep full data range
            ax_main.set_xticks([0, tEnd/2, tEnd])  # show only 0, 10, 20 on x-axis
            y_ticks = np.linspace(y_min, y_max, 5)
            #ax_main.set_yticks(y_ticks)
            ax_main.set_xticklabels([f"{0:.0f}", f"{tEnd/2:.0f}", f"{tEnd:.0f}"])     
            #ax_main.set_yticklabels([f"{val:.0f}" for val in y_ticks]) 
            ax_main.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': fontSize1})
            ax_main.set_ylabel(ylabel_main, fontdict={'family': 'Times New Roman', 'size': fontSize1})
            ax_main.grid(color='lightgray', linestyle='--', linewidth=0.5)
            ax_main.tick_params(axis='both', labelsize=fontSize2)
            ax_main.set_facecolor('#f0f8ff')
            ax_main.legend(fontsize=fontSize2)
            # ax_main.annotate(f'MAPE: {mape:.2f}%', xy=(0.7, y_ticks[2]+0.8*y_ticks[2]), fontsize=10, backgroundcolor='lightgrey')
            ax_main.text(0.8, 0.1, f'MAPE: {mape:.2f}%',transform=ax_main.transAxes,fontdict={'family': 'Times New Roman', 'size': fontSize1},backgroundcolor='lightgrey')
            ax_main.axvline(x=timeVecOut[SLIDESteps - 1], color='gray', linestyle='--', linewidth=1)
            # ax_main.text(timeVecOut[SLIDESteps - 1]+0.08, y_ticks[2]-0.8*y_ticks[2], r'$t_d$', fontdict={'family': 'Times New Roman', 'size': fontSize1})
            ax_main.text(0.03, 0.6, r'$t_d$',transform=ax_main.transAxes,fontdict={'family': 'Times New Roman', 'size': fontSize1},va='top', ha='center')
            fig.savefig(f"{string}_estimate{key}.pdf", format='pdf', bbox_inches='tight')
            fig.show()


       
        return         
#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__': #include this to enable parallel processing

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #put this part out of if __name__ ... if you like to use multiprocessing
    endTime=0.25
    
    model       = SLIDEModel(nStepsTotal=200, endTime=1)
 
    #MyNeuralNetwork()
    nntc = network_training_center(nnModel=model, computeDevice='cpu')
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    nntc.CreateTrainingAndTestData(nTraining=64, nTest=10,
                                   #parameterFunction=PVCreateData, #for multiprocessing
                                   )
    
    nntc.TrainModel(maxEpochs=100)
    
    nntc.EvaluateModel(plotTests=[0,1,2,3,4], plotTrainings=[0,1,2,3,4], plotVars=['time','ODE2'])





