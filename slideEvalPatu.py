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
import exudyn as exu
from exudyn.utilities import *
# from exudyn.signalProcessing import GetInterpolatedSignalValue
from exudyn.processing import ParameterVariation
from exudyn.plot import PlotSensorDefaults

from Models.ModelNN import NNHydraulics
from SLIDE.fnnLib import * 
import SLIDE.fnnLib

import multiprocessing as mp
import sys
import numpy as np
import time
# #from math import sin, cos, sqrt,pi
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import matplotlib.pyplot as plt

import torch
from torch import nn
# %%
from torch.utils.data import TensorDataset, DataLoader


torch.set_num_threads(1)

useCUDA = torch.cuda.is_available()
useCUDA = True #CUDA support helps for fully connected networks > 256

if __name__ == '__main__': #include this to enable parallel processing
    print('pytorch cuda=',useCUDA)

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



#for creating data, set createData = True
createData      =   False       #activate this to create new training and test data sets
runParallel     =   True       #parallel data set creation (loads your computer heavily!)
Patu            =   False       #Cases TwoArms (Patu), OneArm (LiftBoom) set Patu = True for TwoArms
parameterVariation  = True      #in case of True, we can perform parameter variation for hyperparameters (untested for hydraulics)


#80 % data is provided for training and 20 % data is provided for validation (nTest)
nTraining           =128*8 #number of training samples to be used, 256*8
nTest               =128*2  #number of test samples for evaluation--validation
sensorsConfig       = [0,2] # 
# runParallel         = False
#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
listNLOSC=['hydraulics']#, '5mass']
nnCase = 0
caseStr = listNLOSC[nnCase] 
nnType = 'RNN' if 'RNN' in caseStr else 'FFN'

if __name__ == '__main__': 
    print('NN CASE='+caseStr+', nnType='+nnType)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#put this part out of if __name__ ... if you like to use multiprocessing

endTime=1           #total simulation time of training/test data
nStepsTotal=200*endTime     #number of steps used for training/test data
ReD   = 3.35e-3 #.45, 2
LiftLoad = 100


#endTime=1*2*2           #total simulation time of training/test data
#nStepsTotal=500*2*2     #number of steps used for training/test data


model = NNHydraulics(nStepsTotal=nStepsTotal, endTime=endTime, 
                     nnType=nnType, 
                     ReD   = ReD,
                     mL    = LiftLoad,
                     loadFromSavedNPY=True, #if data is not available, set this to false for first (serial run)    #activate this to find history window for damped oscillationns
                     visualization = False,
                     system=Patu,
                     verboseMode=1)

nntc = NeuralNetworkTrainingCenter(nnModel=model, computeDevice='cuda' if useCUDA else 'cpu',
                                   verboseMode=0)
SLIDE.fnnLib.moduleNntc = nntc #this informs the module about the NNTC (multiprocessing)

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#in this section we create or load data
if  Patu:
    dataFile = 'data/PATU/'+nnType+'T'+str(nTraining)+'-'+str(nTest)+'s'+str(nStepsTotal)+\
                't'+str(endTime)+'ReD'+str(ReD)+'Load'+str(LiftLoad)
else: 
    dataFile = 'data/LiftBoom/'+nnType+'T'+str(nTraining)+'-'+str(nTest)+'s'+str(nStepsTotal)+\
                   't'+str(endTime)+'ReD'+str(ReD)+'Load'+str(LiftLoad)

                

if __name__ == '__main__': #include this to enable parallel processing
    if createData:
        import os
        #do not overload the computer during data creation (eigenmode computation on many threads) ...
        os.environ["MKL_NUM_THREADS"] = "1" 
        os.environ["NUMEXPR_NUM_THREADS"] = "1" 
        os.environ["OMP_NUM_THREADS"] = "1" 
        
        nntc.verboseMode = 1
        parameterFunction=None
        if runParallel:
            parameterFunction=PVCreateData
        nntc.CreateTrainingAndTestData(nTraining=nTraining, nTest=nTest,
                                       parameterFunction=PVCreateData, #for multiprocessing, uses all threads available
                                       system=Patu, 
                                       #showTests=[0], #run SolutionViewer for this test
                                       )
        nntc.SaveTrainingAndTestsData(dataFile)
        #nntc.SaveTrainingAndTestsData('testFileDataCreation.txt')
        sys.exit()

if not createData and not parameterVariation:
# %%
    nntc.LoadTrainingAndTestsData(dataFile, 
                                  nTraining=nTraining, nTest=nTest, 
                                  sensorConfiguration=sensorsConfig, 
                                  StepPredictor=1,
                                  #these values must be in the range of available data!
                                  )

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__': #include this to enable parallel processing

    identifierString = 'B_'+nnType
    formatted_ReD = "{:.2e}".format(ReD)
    storeModelName = 'model/'+model.GetModelName()+identifierString+'ReD'+str(formatted_ReD)
    resultsFile = 'solution/'+model.GetModelNameShort()+'Res'+identifierString+'ReD'+str(formatted_ReD)

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if parameterVariation:
        
        # Liftboom: Sensor combination ([2]= [0, 1]) indicates U and s.
        # Liftboom: Sensor combination ([30]= [0, 1]) indicates U, s, dots, p1 and p2.
        
        # PATU Crane: Sensor combination ([14]= [0, 1, 2, 3]) indicates U1,U2, s1 and s2.
        # PATU Crane:: Sensor combination ([1022]= [0, 1, 2, 3, 4,5, 6, 7, 8,9]) 
        #indicates U1,U2, s1, s2, dots1, dots2, p1, p2, p3 and p4.
        
        if nnType == 'FFN':
            functionData = {'maxEpochs':1000, #for linear 1mass, 5mass, friction: 2500 is alredy very good
                            'nTest':nTest,
                            'lossThreshold':2.5e-5,
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
                            'system': Patu
                            } #additional parameters
            parameters = {
                          'learningRate':[1e-3], # best one[ 0, 11, 12, 17, 25]. 14 (4 sensors) and 1022 (all sensors)[0, 11, 12, 17, 25]
                          'hiddenLayerStructureType':[0], #0,7,8,9,13,14, 15, 31, 0,14, 15, 16,17
                          'hiddenLayerSize':[29*2], #larger is better; 4: does not work at all; 'RL' with 8 works a bit
                          'case':[1], #randomizer
                          'stepPredictor': [1] , #[0, 49, 14,19, 24]
                          'sensorType': [2 ], #[30, 14]
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

        #%%++++++++++++++++++++++++++++++++++++++
        #show loss, test error over time, ...
        resultsFileNpy = resultsFile+'.npy'
        nntc.PlotTrainingResults(resultsFileNpy, dataFile)
        


    
        #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #check specific solution
        
        
        # nntc.LoadNNModel(storeModelName+str(3)+'.pth') #best model:
        # # model.mbs.PlotSensor(closeAll=True)
        # #ll=nntc.EvaluateModel(plotTests=[0,1,2,3,4], plotTrainings=[0,1,2,3,4], plotVars=['time','phi0'])
        # trainings = np.array([0,1,2,3,4,5])
        # tests = np.array([0,1,2,3])
        # compShow = []
        # # compShow += ['velGroundX','velGroundY','velGroundZ',]
        # compShow += ['posArmY']
        # for comp in compShow:
        #     nntc.EvaluateModel(plotTests=tests, plotTrainings=trainings, plotVars=['time',comp])
    
    
        #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    
    else: #single run
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        nntc.verboseMode=1
        hiddenLayerStructure=''
        
        Training = True

        #FFN:
        if Training:
            nntc.TrainModel(neuralNetworkTypeName=nnType, maxEpochs=2000, learningRate= 5e-3, 
                        lossThreshold=1e-8, batchSize=nTraining, 
                        hiddenLayerSize=415, hiddenLayerStructure=hiddenLayerStructure,
                        #hiddenLayerSize=200, hiddenLayerStructure='LLSLSL', #slower, more accurate
                        lossLogInterval=1, epochPrintInterval=50, testEvaluationInterval=1,
                        dataLoaderShuffle=True
                        )
        else:
            
            # Step 01: Load FFN model
           
        
            # Step 02: Design a simulation setup to test FFN. Start from a random angle, initial pressures and input signal.
            T      = 10
            ns     = T*nStepsTotal
            p1Init = 3e6
            p2Init = 3e6
            angleMinDeg = -10
            angleMaxDeg = 50
            angleInit = np.random.rand()*(angleMaxDeg-angleMinDeg)+angleMinDeg
            
            rv = {}
            inputVecLong = np.ones((5*ns))
            rv['U'] = inputVecLong[0:ns]        
            rv['p0'] = inputVecLong[1*ns:2*ns]
            rv['s'] = inputVecLong[2*ns:3*ns]
            rv['p1'] = inputVecLong[3*ns:4*ns]
            rv['s_t'] = inputVecLong[4*ns:5*ns]

            #only prior to simulation:        
            rv['sInit'] = inputVecLong[2*ns]
            rv['s_tInit'] = inputVecLong[4*ns]
            
            #Step 02: Design a simulation setup
            model12     = NNHydraulics(nStepsTotal=ns, endTime=T, 
                          nnType=nnType, 
                          ReD   = ReD,
                          loadFromSavedNPY=True, #if data is not available, set this to false for first (serial run)    #activate this to find history window for damped oscillationns
                          verboseMode=1) 
            model12.CreateModel()
            
            # Step 02.1: Define sequence of control signal. Change it as you want.
            ten_percent  = int(0.1 * ns)
            five_percent = int(0.05 * ns)
            two_percent  = int(0.02 * ns)
            one_percent  = int(0.01 * ns)
            
            segment1     = np.zeros(ten_percent)        # 2% steps at zero.
            segment2     = np.ones(ten_percent)        # 5% steps at 1.
            segment3     = np.zeros(ten_percent)       # 5% steps at zero.
            segment4     = -1*np.ones(ten_percent+one_percent)     # 5% steps at -1.
            
            ramp_up_1    = np.linspace(0, 0.6, five_percent)  # 5% ramp between 0 and 0.6.
            ramp_down_1  = np.linspace(0.6, 0, five_percent)  # 5% ramp between 0.6 and 0.
            static1      = 0.6*np.ones(2*two_percent)         # 4% steps at 0.6.
            segment5     = np.concatenate((ramp_up_1,static1, ramp_down_1)) 
            
            segment6 = np.random.uniform(low=-0.8, high=0.8, size=two_percent)
            segment6 = np.repeat(segment6, 5)
            
# %%
            #segment6 = segment6[(segment6 != -1) & (segment6 != 0) & (segment6 != 1)]

            
            ramp_up_2    = np.linspace(0, -1, five_percent) # 5% ramp between 0 and -0.8.
            static2      = -1*np.ones(2*two_percent)        # 4% steps at 0.6.
            ramp_down_2  = np.linspace(-1, 0, five_percent) # 5% ramp between -0.8 and 0.
            segment7     = np.concatenate((ramp_up_2,static2, ramp_down_2)) 
            
# %%
            segment8     = np.zeros(ten_percent)    # 10% steps at zero.

            segment9     = np.ones(five_percent)     # 10% steps at 1.
            segment10    = -1*np.ones(ten_percent)  # 10% steps at -1.
            segment11    = np.zeros(five_percent+one_percent)    # 10% steps at zero.
            
            segments    = [segment3,-segment9, segment11, segment9,segment11,-segment9,
                           segment1,segment9, segment3, segment7, segment8, -segment7]
            ControlSig  = np.concatenate(segments)
            rv['U']     = ControlSig[0:ns]
            inputData   = np.concatenate([rv['U'], rv['p0'], rv['s'], rv['p1'], rv['s_t']])
            
            timeVecLong    = np.zeros(ns)
            dataTest       = model12.ComputeModel(inputData, solutionViewer = False, dampedwindow=False)
            dataTestPath = 'data/hyd'+nnType+'T'+'s'+str(ns)+\
                            't'+str(T)+'ReD'+str(ReD)
                            
            dampedSteps  = 62
            StepPredictor=1
            slices       = int(ns- dampedSteps)
            
            outputVecLong      = np.zeros((ns,1))
            Yprediction        = np.zeros((ns,1))
            Inputs             = dataTest[0]
            outputVecLong      = dataTest[1]
            
            inputVecLong           = []
                
            for k in range(slices): 
                        startIndex0 = k
                        startIndex1 = k+dampedSteps
                        endIndex    = startIndex1+StepPredictor
                    
                        #Sensor layer for each layer
                        sensorsLayers = [] 
                        for n, j in enumerate(sensorsConfig): 
                            sensLayer            =  Inputs[j*ns:(j+1)*ns] # Extract sensor with user configuration
                            dampedSensor         = sensLayer[startIndex0:startIndex1] # This sensor layer is built on dampedSteps
                            sensorsLayers        = np.hstack((sensorsLayers, dampedSensor))
                                            
                        if k == 0:
                            inputVecLong = np.vstack(( sensorsLayers)).T  
                        else: 
                            inputVecLong = np.vstack((inputVecLong, sensorsLayers))
                     
            nntc.LoadNNModel(storeModelName+str(0)+'.pth')    
            model_device = next(nntc.rnn.parameters()).device
            input_tensor = torch.tensor(inputVecLong, dtype=nntc.floatType)
            input_tensor = input_tensor.to(model_device)
            output_tensor = nntc.rnn(input_tensor)
            # Move output_tensor to CPU before converting to numpy
            Yprediction[dampedSteps:ns] = output_tensor[:, -1].detach().cpu().numpy().reshape(-1, 1)
            
            for i in range(ns):
                 timeVecLong[i] = i* T/ns
                 
            import matplotlib.gridspec as gridspec
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
            from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
            from matplotlib.patches import FancyArrowPatch

            
            # Fig01: Control signal
            a40_width_inches, a40_height_inches = 8.3 / 2, 11.7 / 2  

            fig, ax = plt.subplots(figsize=(a40_width_inches, a40_height_inches))

            # Plot control signal
            ax.plot(timeVecLong, ControlSig, color='k')
            ax.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': 10})
            ax.set_ylabel(r'Control signal, V', fontdict={'family': 'Times New Roman', 'size': 10})
            ax.set_xlim(0, 10)
            # Set ticks
            ax.set_xticks(np.linspace(0, 10, 6))
            ax.set_yticks(np.linspace(-1.25, 1.25, 6))
            ax.tick_params(axis='both', labelsize=10)  # Change 8 to the desired font size
            # Enable grid and legend
            ax.grid(True)
            # Adjust layout and save the figure
            plt.tight_layout()
            plt.savefig('solution/Evaluation_controlsignal.png', format='png', dpi=300)
            plt.show()

            # Fig02: Deflection estimation
            a4_width_inches, a4_height_inches = 8.3, 11.7/2  
            fig = plt.figure(figsize=(a4_width_inches, a4_height_inches))
            gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1])

            # Main plot
            ax_main = fig.add_subplot(gs[0, :])  # Span all columns for the first row
            # Plot data
            ax_main.plot(timeVecLong, outputVecLong, color='r', linestyle='-', label='Reference solution')
            ax_main.plot(timeVecLong, Yprediction, color='b', linestyle=':', label='DNN estimations')

            # Set legend with specific font size
            ax_main.set_xlabel('Time, s',fontdict={'family': 'Times New Roman', 'size': 10})
            ax_main.set_ylabel(r'$\delta_y$, m (Normalized)', 
                                   fontdict={'family': 'Times New Roman', 'size': 10})  
            ax_main.set_xlim(0, 10)
            ax_main.set_xticks(np.linspace(0, 10, 6))  
            ax_main.set_yticks(np.linspace(-1, 1, 6))
            mape = mean_absolute_percentage_error(outputVecLong[dampedSteps:ns,0], Yprediction[dampedSteps:ns,0]) 
            ax_main.annotate(f'MAPE: {mape:.2f}%', xy=(0.5, 0.8), xycoords='data', fontsize=10, backgroundcolor='lightgrey')
            
            t_d = timeVecLong[dampedSteps - 1] 
            ax_main.axvline(x=t_d, color='gray', linestyle='--', linewidth=1)  # Add a vertical dashed line at t_d

            # arrow_left = FancyArrowPatch((-0.01, -0.15), (t_d+0.02, -0.15),
            #                  arrowstyle="<->",
            #                  mutation_scale=10, color="black", linewidth=2)
            # ax_main.add_patch(arrow_left)
            # Add label below the arrow
            ax_main.text(t_d+0.1, 0.05, r'$t_d$', fontdict={'family': 'Times New Roman', 'size': 12})
            # Enable grid and legend
            ax_main.tick_params(axis='both', labelsize=10)  # Change 8 to the desired font size

            ax_main.grid(True)
            ax_main.legend()

            # Error subplot
            ax_error = fig.add_subplot(gs[1, 0])  
            error_percentage = np.array([
                mean_absolute_error([true_val], [pred_val]) 
                for true_val, pred_val in zip(outputVecLong, Yprediction)])
            ax_error.plot(timeVecLong, error_percentage, color='g')
            ax_error.set_xlabel('Time, s')
            ax_error.set_ylabel('MAE')
            
            ax_error.set_xlim(0, 10)
            ax_error.set_xticks(np.linspace(0, 10, 6))
            ax_error.set_yticks(np.linspace(0, 0.1, 6))  

            # Set y-axis limit if desired
            ax_error.set_ylim(0, 0.1)
            ax_error.tick_params(axis='both', labelsize=10)  # Change 8 to the desired font size

            ax_error.grid(True)

            # Zoomed-in subplot
            ax_zoom = fig.add_subplot(gs[1, 1])  # First row, second column
            ax_zoom.plot(timeVecLong, outputVecLong, color='r', linestyle='-')
            ax_zoom.plot(timeVecLong, Yprediction, color='b', linestyle=':')
            
            ax_zoom.set_xlim(8, 10)  # Zoom between 6 and 8 on the x-axis
            
            ax_zoom.set_xticks(np.linspace(8, 10, 5))
            ax_zoom.set_yticks(np.linspace(-0.5, 0.5, 5))
            ax_zoom.set_ylim(-0.5, 0.5)
            
            ax_zoom.tick_params(axis='both', labelsize=10)  # Change 8 to the desired font size


            ax_zoom.grid(True)
            ax_zoom.set_xlabel('Time, s',fontdict={'family': 'Times New Roman', 'size': 10})
            ax_zoom.set_ylabel(r'$\delta_y$, m (Normalized)', 
                                   fontdict={'family': 'Times New Roman', 'size': 10}) 
            
            # Adjust the layout
            plt.tight_layout()
            plt.savefig('solution/Evaluation_2sensors_1280Samples_50kg.png', format='png', dpi=300)
            plt.show()
