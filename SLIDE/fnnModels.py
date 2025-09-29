import exudyn as exu
from exudyn.utilities import *
from exudyn.signalProcessing import GetInterpolatedSignalValue
from exudyn.physics import StribeckFunction

import sys
import numpy as np
from math import sin, cos, pi, tan, exp, sqrt, atan2

from enum import Enum #for data types

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++
#create an Exudyn model
class NNtestModel():

    #initialize class 
    def __init__(self):
        
        self.SC = None
        self.mbs = None
        self.modelName = 'None'
        self.modelNameShort = 'None'
        self.inputScaling = np.array([])
        self.outputScaling = np.array([])
        self.inputScalingFactor = 1. #additional scaling factor (hyper parameter)
        self.outputScalingFactor = 1.#additional scaling factor (hyper parameter)
        self.nOutputDataVectors = 1  #number of output vectors (e.g. x,y,z)
        self.nnType = 'RNN'
        
        self.nStepsTotal = None
        self.computationType = None

        self.nODE2 = None               #number of ODE2 states
        self.useVelocities = None
        self.useInitialValues = None
        
        self.simulationSettings = None
    
    #return NN-Type; ['RNN', 'FFN', 'LSTM']
    def NNtype(self):
        return self.nnType

    def IsRNN(self):
        return self.nnType == 'RNN'
    
    def IsFFN(self):
        return self.nnType == 'FFN'
    
    #create a model and interfaces
    def CreateModel(self):
        pass

    #get model name
    def GetModelName(self):
        return self.modelName

    #get short model name
    def GetModelNameShort(self):
        return self.modelNameShort

    #get number of simulation steps
    def GetNSimulationSteps(self):
        return self.nStepsTotal
    
    #get number of sensors
    def GetNumberofSensors(self):
        return self.numSensors
    
    #get historical window
    def GetHistoricalWindow(self):
        return self.histwindow    
    
    #get historical window
    def SLIDEWindow(self):
        return self.n_td    


    #return a numpy array with additional scaling for inputs when applied to mbs (NN gets scaled data!)
    #also used to determine input dimensions
    def GetInputScaling(self):
        return self.inputScalingFactor*self.inputScaling
    
    #return a numpy array with scaling factors for output data
    #also used to determine output dimensions
    def GetOutputScaling(self):
        return self.outputScalingFactor*self.outputScaling

    #return input/output dimensions for torch depending on FFN and RNN
    #returns [size of input, shape of output]
    def GetInputOutputSizeNN(self):
        if self.nnType == 'FFN':
            #self.numSensors = 5
            #self.histwindow = 199
            return [self.inputScaling.size, (self.outputScaling.size,)]
        else:
            return [self.inputScaling.shape[-1], self.outputScaling.shape]

    #get time vector according to output data
    def GetOutputXAxisVector(self):
        return np.array([])

    #create a randomized input vector
    #relCnt can be used to create different kinds of input vectors (sinoid, noise, ...)
    #isTest is True in case of test data creation
    def CreateInputVector(self, relCnt = 0, isTest=False):
        return np.array([])

    #create initialization of (couple of first) hidden states
    def CreateHiddenInit(self, isTest):
        return np.array([])
    
    #split input data into initial values, forces or other inputs
    #return dict with 'data' and possibly 'initialODE2' and 'initialODE2_t'
    def SplitInputData(self, inputData, hiddenData=None):
        return {'data':None}
    
    #split output data to get ODE2 values (and possibly other data, such as ODE2)
    #return dict {'ODE2':[], 'ODE2_t':[]}
    def SplitOutputData(self, outputData):
        return {'ODE2':[]}

    #convert all output vectors into plottable data (e.g. [time, x, y])
    #the size of data allows to decide how many columns exist
    def OutputData2PlotData(self, outputData, forSolutionViewer=False):
        return np.array([])

    #return dict of names to columns for plotdata        
    def PlotDataColumns(self):
        return {}

    #get compute model with given input data and return output data
    def ComputeModel(self, inputData, hiddenData=None, verboseMode = 0, solutionViewer = False):
        return np.array([])
    
    #visualize results based on given outputData
    #outputDataColumns is a list of mappings of outputData into appropriate column(s), not counting time as a column
    #  ==> column 0 is first data column
    def SolutionViewer(self, outputData, outputDataColumns = [0]):
        nColumns = self.nODE2
        data = self.OutputData2PlotData(outputData, forSolutionViewer=True)
        
        # columnsExported = dict({'nODE2':self.nODE2, 
        #                         'nVel2':0, 'nAcc2':0, 'nODE1':0, 'nVel1':0, 'nAlgebraic':0, 'nData':0})
        columnsExported = [nColumns, 0, 0, 0, 0, 0, 0] #nODE2 without time
        if data.shape[1]-1 != nColumns:
            raise ValueError('NNtestModel.SolutionViewer: problem with shape of data: '+
                             str(nColumns)+','+str(data.shape))

        nRows = data.shape[0]
        
        
        sol = dict({'data': data, 'columnsExported': columnsExported,'nColumns': nColumns,'nRows': nRows})

        self.mbs.SolutionViewer(sol,runOnStart=True)
    

#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__': #include this to enable parallel processing

    #model = NonlinearOscillator(nStepsTotal=100)
    model = DoublePendulum(nStepsTotal=500, endTime=5, nnType='FFN')
    #model.CreateModel()
    
    # inputData = [1.6,1.6,0.000,0.300,1.010,2.130] #case1
    inputData = [1.6,2.2,0.030,0.330,1.500,2.41] #case2
    output = model.ComputeModel(inputData, 
                                #hiddenData=[1.6,1.6,0,0],  #for RNN
                                verboseMode=True, solutionViewer=False)
    # model.mbs.PlotSensor([model.sAngles]*4,components=[0,1,2,3], closeAll=True,labels=['phi0','phi1','phi0_t','phi1_t'])
    model.mbs.PlotSensor([model.sAngles]*2,components=[0,2], closeAll=True, labels=['phi0','phi0_t'])
    model.mbs.PlotSensor([model.sPos0,model.sPos1],components=[1,1],componentsX=[0,0], newFigure=True,
                         labels=['m0','m1'])
    print('mbs last step=\n',
          model.mbs.GetSensorStoredData(model.sAngles)[-1,1:])
    
   