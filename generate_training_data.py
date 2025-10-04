#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Generate training data for hydraulically actuated 3D FFRF reduced order model with 2 flexible bodies.
#           It includes lift boom (1 DOF) and Patu crane (2-DOF) systems.
#
# Author:   Qasim Khadim,Johannes Gerstmayr
# Date:     2025-06-23
#
# Copyright:This file is part of Exudyn. Exudyn is free software. You can redistribute 
#it and/or modify it under the terms of the Exudyn license. See 'LICENSE.txt' for more details.
#+++++++++++++++++++++++++++0++++++++++++++++++++++++++++++++++++++++++++++++++
from models.data_source import SLIDEModel
from models.dataset_manager import  *
import models.dataset_manager

import torch
import multiprocessing as mp 
import time, sys
# %%Simulation parameters 
Case                = 'Patu'            #Patu else 'LiftBoom'
endTime             = 2                 #total simulation time of training/test data, change to 2 for Patu
nStepsTotal         = int(200*endTime)  #number of steps used for training/test data
LiftLoad            = 0
Material            = 'Steel'           #Steel, Aluminium, Titanium and Composites (Only Steel in MSSP)
nModes              = 10                 # 2(8) modes (MSSP:Lift Boom), 10(16) modes (MSSP:PATU)
model               = SLIDEModel(nStepsTotal=nStepsTotal, endTime=endTime, nnType='FFN', system=Case, nModes= nModes, 
                                   loadFromSavedNPY=True, mL = LiftLoad,
                                   material=Material,verboseMode=1)

#Training data parameters: 80 % data is training data and 20 % test data
nTrain              = 256
nTraining           = nTrain*8                  #16*8
nTest               = nTrain*2                  #16*2
dataFile            = f"solution/data/{Case}/data{Material}_{endTime}T_{nTraining}nTrainings_{nTest}nTests_{nStepsTotal}steps_Load{LiftLoad}_Modes{nModes}"
          

# %% Parallel computing
torch.set_num_threads(1)
useCUDA             = torch.cuda.is_available()
useCUDA             = True              #CUDA support helps for fully connected networks > 256
runParallel         = True              #parallel data set creation (loads your computer heavily!)
parameterVariation  = True              #True, we can perform parameter variation for hyperparameters

if __name__ == '__main__':              #include this to enable parallel processing
    print('pytorch cuda=',useCUDA)

#put this part out of if __name__ ... if you like to use multiprocessing
nntc                              = training_data_center(nnModel=model, computeDevice='cuda' if useCUDA else 'cpu', verboseMode=0)
models.dataset_manager.moduleNntc = nntc #this informs the module about the NNTC (multiprocessing)

if __name__ == '__main__': #include this to enable parallel processing
         
        import os
        #do not overload the computer during data creation (eigenmode computation on many threads) ...
        os.environ["MKL_NUM_THREADS"] = "1" 
        os.environ["NUMEXPR_NUM_THREADS"] = "1" 
        os.environ["OMP_NUM_THREADS"] = "1" 
        
        nntc.verboseMode = 1
        parameterFunction=None
        
        start_time      = time.time()

        if runParallel:
            parameterFunction=PVCreateData
        nntc.CreateTrainingAndTestData(nTraining=nTraining, nTest=nTest,nStepsTotal=nStepsTotal,flattenData=False,
                                       parameterFunction=PVCreateData, #for multiprocessing, uses all threads available
                                       # showTests=[0], #run SolutionViewer for this test
                                       )
        nntc.SaveTrainingAndTestsData(dataFile)
        cpuTime = time.time() - start_time
        print("--- data generation took: %.5f seconds ---" % (cpuTime))

        #nntc.SaveTrainingAndTestsData('testFileDataCreation.txt')
        sys.exit()

