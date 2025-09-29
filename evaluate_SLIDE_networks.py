# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 11:47:30 2025

@author: qkhadim22
"""
from models.data_source import SLIDEModel
from generate_training_data import *
from SLIDE.networksLib import  *
from train_SLIDE_networks import *

Y_Estimation ={}   
angleInit1   = 0 #0.780053102206466+5
angleInit2   = -45.4078798791467
endTime      = 10              #total simulation time of training/test data, change to 2 for Patu
nStepsTotal  = int(200*endTime)  #number of steps used for training/test data

model        = SLIDEModel(nStepsTotal=nStepsTotal, endTime=endTime, nnType='FFN', 
                                    system=Case, nModes= 3, 
                                   loadFromSavedNPY=True, mL = LiftLoad,
                                   material=Material,verboseMode=1)

inputVec     = model.CreateInputVector(theta1=angleInit1, theta2=angleInit2, Evaluate=True ) 
data         = model.ComputeModel(inputVec,  solutionViewer = False)
nntc         = network_training_center(computeDevice='cuda' if useCUDA else 'cpu', verboseMode=0)

X,Y,X0,Y0    = nntc.LoadTrainingAndTestsData(fileName=None, Data=data, nStepsTotal= nStepsTotal, 
                                              InputLayer=inputConfig, OutputLayer=outConfig,
                                              StepPredictor=Steps,  SLIDESteps=SLIDESteps, 
                                              Evaluate=True,file=dataFile)
                
input_tensor = torch.tensor(X, dtype=nntc.floatType).to(nntc.computeDevice)
 
nModels = 5 if parameterVariation else 1

for i in range(nModels):
    myModel = torch.load(storeModelName + str(i) + '.pth', weights_only=False)
    identifyString = storeModelName + str(i)
    model_device = next(myModel.rnn.parameters()).device
    output_tensor = myModel.rnn(input_tensor)
    
    for j, key in enumerate(outConfig):
        Y_Estimation[key] = np.zeros((1, nStepsTotal))
        Y_Estimation[key][0, SLIDESteps:nStepsTotal] = output_tensor[:, j].detach().cpu().numpy()
    
    nntc.Plotting(ns=nStepsTotal,SLIDESteps=SLIDESteps,Steps=Steps, Y_Estimation=Y_Estimation,
                  data=data,InputLayer=inputConfig,OutputLayer=outConfig,string=identifyString,
                  file=dataFile)

  

       