#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Evaluate trained SLIDE networks.
#
# Author:   Johannes Gerstmayr
# Date:     2025-06-23
#
# Copyright:This file is part of Exudyn. Exudyn is free software. You can redistribute it and/or modify it under the terms of the Exudyn license. See 'LICENSE.txt' for more details.
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#from generate_training_data import *
#from SLIDE.networksLib import  *
from train_SLIDE_networks import *

Y_Estimation ={}   
angleInit1   = 18.237484289236395 #18.237484289236395(LiftBoom)
angleInit2   = -30                   # -45.4078798791467
T            = 10              #total simulation time of training/test data, change to 2 for Patu
nSteps      = int(200*T)  #number of steps used for training/test data

model        = SLIDEModel(nStepsTotal=nSteps, endTime=T, nnType='FFN', 
                                    system=Case, nModes= nModes, 
                                   loadFromSavedNPY=True, mL = LiftLoad,
                                   material=Material,verboseMode=1)

inputVec     = model.CreateInputVector(theta1=angleInit1, theta2=angleInit2, Evaluate=True ) 
data         = model.ComputeModel(inputVec,  solutionViewer = False)
nntc         = network_training_center(computeDevice='cuda' if useCUDA else 'cpu', verboseMode=0)

X,Y,X0,Y0    = nntc.LoadTrainingAndTestsData(fileName=None, Data=data, nStepsTotal= nSteps, 
                                              InputLayer=inputConfig, OutputLayer=outConfig,
                                              StepPredictor=Steps,  SLIDESteps=SLIDESteps, 
                                              Evaluate=True,file=dataFile)
                
input_tensor = torch.tensor(X, dtype=nntc.floatType).to(nntc.computeDevice)
 
nModels = 5 if parameterVariation else 1

for i in range(nModels):
    
    if nModels==1:
        myModel = nntc.LoadNNModel(storeModelName)
        identifyString = storeModelName
    else:
        myModel = nntc.LoadNNModel(storeModelName + str(i))
        identifyString = storeModelName + str(i)
        
    output_tensor = myModel(input_tensor)
    
    for j, key in enumerate(outConfig):
        Y_Estimation[key] = np.zeros((1, nSteps))
        Y_Estimation[key][0, SLIDESteps:nSteps] = output_tensor[:, j].detach().cpu().numpy()
    
    nntc.Plotting(ns=nSteps,SLIDESteps=SLIDESteps,Steps=Steps, Y_Estimation=Y_Estimation,
                  data=data,InputLayer=inputConfig,OutputLayer=outConfig,string=identifyString,
                  file=dataFile)

  

       