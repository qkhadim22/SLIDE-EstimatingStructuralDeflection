#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Test model for 3D FFRF reduced order model with 2 flexible bodies
#
# Author:   Johannes Gerstmayr
# Date:     2023-06-23
#
# Copyright:This file is part of Exudyn. Exudyn is free software. You can redistribute it and/or modify it under the terms of the Exudyn license. See 'LICENSE.txt' for more details.
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from models.data_source import SLIDEModel
from generate_training_data import *
from SLIDE.networksLib import  *
import SLIDE.networksLib


#Training phase
nnType='FFN'
data           = np.load(dataFile + ".npy", allow_pickle=True).item()
# SLIDESteps     =int(np.mean(data['SLIDESteps'][0][0])) # Statistics-based
SLIDESteps     = int(np.mean(data['SLIDESteps'][0][1])) # EOM-based

if not Case == 'Patu':
    #These input-output sensor configurations are in MSSP paper for 1-DOF system.
    #inputConfig     = ['U1','s1', 'ds1', 'p1', 'p2']
    #outConfig       = ['deltaY','eps1', 'sig1']
    
    inputConfig      = ['U1','s1', 'ds1', 'p1', 'p2'] # Minimum sensor configuration
    outConfig        = ['deltaY','eps1', 'sig1']
    Steps            = 1 # 1 (single step)/more (multisteps)    
else: 
    #These input-output sensor configurations are in MSSP paper for 2-DOF system.
    #inputConfig     = ['U1', 'U2','s1', 's2', 'ds1', 'ds2', 'p1', 'p2', 'p3', 'p4']
    #outConfig       = ['deltaY','eps1', 'sig1', 'eps2', 'sig2']
    
    inputConfig      = ['U1', 'U2','s1', 's2', 'ds1', 'ds2', 'p1', 'p2', 'p3', 'p4']
    outConfig        = ['deltaY','eps1', 'sig1', 'eps2', 'sig2']
    Steps            = 1 # 1 (single step)/more (multisteps)
    
nntc                            = network_training_center(computeDevice='cuda' if useCUDA else 'cpu', verboseMode=0)
SLIDE.networksLib.moduleNntc    = nntc #this informs the module about the NNTC (multiprocessing)
inputStr       = ",".join(inputConfig)
outputStr      = ",".join(outConfig)
storeModelName = f"solution/model/{Case}/{model.GetModelNameShort()}_{nnType}{nTraining}_{nTest}_{Material}_{endTime}t{nStepsTotal}{LiftLoad}_{inputStr}_{outputStr}" 
resultsFile    = storeModelName


def main():    
    if not parameterVariation:
        # Load and scale data
        nntc.LoadTrainingAndTestsData(dataFile,nTraining=nTraining, nTest=nTest, nStepsTotal=nStepsTotal, #system=Case,
                                      InputLayer=inputConfig, OutputLayer=outConfig,StepPredictor=Steps)
        
        # Train Network
        nntc.verboseMode=1
        hiddenLayerStructure=''
        nntc.TrainModel(neuralNetworkTypeName=nnType, maxEpochs=1000, learningRate= 1e-3, 
                        lossThreshold=1e-6, #5E-6 was used in MSSP paper 
                        batchSize=int(nTraining/8),hiddenLayerSize=SLIDESteps*len(inputConfig), hiddenLayerStructure=hiddenLayerStructure,
                        #hiddenLayerSize=200, hiddenLayerStructure='LLSLSL', #slower, more accurate
                        lossLogInterval=1, epochPrintInterval=50, testEvaluationInterval=1,
                        dataLoaderShuffle=True  )
        
        nntc.SaveNNModel(storeModelName)
        
        
        nntc.PlotTrainingResults()


    else:
        
        if nnType == 'FFN':
            functionData = {'maxEpochs':1000, #for linear 1mass, 5mass, friction: 2500 is alredy very good
                            'nTest':nTest,
                            'lossThreshold':1e-10,
                            'lossLogInterval':20,
                            'testEvaluationInterval':1,
                            'nTraining':nTraining,
                            'neuralNetworkType':int(nnType=='FFN'), #0=RNN, 1=FFN
                            'rnnNonlinearity': 'tanh',
                            'batchSize':int(nTraining/8),
                            'dataLoaderShuffle':True,
                            'storeModelName':storeModelName,
                            'modelName':model.GetModelName(),
                            'dataFile':dataFile,
                            'InputLayer': inputConfig, 
                             'OutputLayer':outConfig
                            # 'system': Patu
                            } #additional parameters
            parameters = {
                          'learningRate':[1e-4], # best one[ 0, 11, 12, 17, 25]. 14 (4 sensors) and 1022 (all sensors)[0, 11, 12, 17, 25]
                          'hiddenLayerStructureType':[ 1, 19,26,8,17], #1, 19,26,8,17
                          'hiddenLayerSize':[SLIDESteps], #larger is better; 4: does not work at all; 'RL' with 8 works a bit
                          'case':[1], #randomizer
                          'stepPredictor': [Steps] , #[0, 49, 14,19, 24]
                          
                          # 'sensorType': [2 ], #[30, 14]
                          }
        else: #RNN
            functionData = {'maxEpochs':500*4, #for linear 1mass, 5mass, friction: 2500 is alredy very good
                            #'nTraining':64,
                            'nTest':20,
                            'lossThreshold':1e-10,
                            'lossLogInterval':25,
                            'testEvaluationInterval':25,
                            'nTraining':256,
                            'neuralNetworkType':int(nnType=='FFN'), #0=RNN, 1=FFN
                            # 'hiddenLayerSize':128,
                            # 'hiddenLayerStructureType': 0,
                            'rnnNonlinearity': 'tanh',
                            'batchSize':32,
                            'dataLoaderShuffle':True,
                            'storeModelName':storeModelName,
                            'modelName':model.GetModelName(),
                            'dataFile':dataFile,
                            } #additional parameters
            parameters = {
                          'hiddenLayerSize':[32], #larger is better; 4: does not work at all; 'RL' with 8 works a bit
                          }
        tStart = time.time()
        #mp.set_start_method('spawn')
	    #torch.set_num_threads(1)
        print('start variation')
        
        [parameterDict, values] = ParameterVariation(parameterFunction=ParameterFunctionTraining, 
                                                     parameters=parameters,
                                                     useMultiProcessing=True,
                                                     #numberOfThreads=6,
                                                     resultsFile=resultsFile+'.txt', 
                                                     addComputationIndex=True, # necessary for seed
                                                      # useMPI = True, 
                                                     parameterFunctionData=functionData)
        CPUtime=time.time()-tStart
        print('training variation took:',round(CPUtime,2),'s')
        # print('values=', values)
        functionData['CPUtime']=CPUtime
        
        #++++++++++++++++++++++++++++++++++++++
        #store data in readable format
        ExtendResultsFile(resultsFile+'.txt', functionData, parameterDict)
        
        dataDict = {'version':1}
        dataDict['parameters'] = parameters
        dataDict['parameterDict'] = parameterDict
        dataDict['values'] = values
        dataDict['functionData'] = functionData
        

        with open(resultsFile+'.npy', 'wb') as f:
            np.save(f, dataDict, allow_pickle=True) #allow_pickle=True for lists or dictionaries
         
        resultsFileNpy = resultsFile+'.npy'
        nntc.PlotTrainingResults(resultsFileNpy, dataFile)  
        
if __name__ == "__main__":
    
    print('pytorch cuda=',useCUDA)
    import os
    #do not overload the computer during data creation (eigenmode computation on many threads) ...
    os.environ["MKL_NUM_THREADS"] = "1" 
    os.environ["NUMEXPR_NUM_THREADS"] = "1" 
    os.environ["OMP_NUM_THREADS"] = "1" 
    
    
    main()