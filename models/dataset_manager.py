
import numpy as np
import torch

from models.data_source import SLIDEModel
from exudyn.processing import ParameterVariation
import os
import matplotlib.pyplot as plt
#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++
#create training and test data
#parameter function, allowing to run parallel

def PVCreateData(parameterFunction):
    global moduleNntc
    
    nnModel         = moduleNntc.GetNNModel()
    
    isTest          = parameterFunction['functionData']['isTest']
    nSamples        = parameterFunction['functionData']['nSamples']
    cnt             = parameterFunction['cnt'] #usually not needed
    flattenData     = parameterFunction['functionData']['flattenData']
    SLIDEwindow     = parameterFunction['functionData']['SLIDEwindow']
    seed            = int(cnt)
   
    if isTest:
        seed = 2**31 - seed #2**32-1 is max value
        #print('seed:', seed)
    np.random.seed(seed)
    
    hiddenVec       = nnModel.CreateHiddenInit(isTest)
    inputVec        = nnModel.CreateInputVector(relCnt = cnt/nSamples, isTest = isTest, SLIDEwindow=SLIDEwindow)
  
    [inputVec, outputVec, SLIDESteps] = nnModel.ComputeModel(inputVec, hiddenVec,SLIDEwindow)

    # if flattenData:
        
    #     return [np.array(inputVec.flatten()), np.array(outputVec.flatten()),hiddenVec.flatten(), SLIDESteps.flatten()]
    # else:
    return [inputVec, outputVec, hiddenVec, SLIDESteps]


    
class training_data_center():
    #create class with model to be trained
    #createModelDict is used to initialize CreateModel function with specific parameters
    #initialize seed
    def __init__(self, nnModel=SLIDEModel(), createModelDict={}, computeDevice = 'cpu', verboseMode=1):
        
        self.nnModel       = nnModel
        self.verboseMode   = verboseMode
        #initialize for pure loading of model:
        self.computeDevice = computeDevice
        
        self.inputsTraining = []
        self.targetsTraining = []
        self.inputsTest = []
        self.targetsTest = []

        #initialization of hidden layers in RNN (optional)
        self.hiddenInitTraining = []
        self.hiddenInitTest = []

        self.floatType = torch.float #16, 32, 64
        
        self.nnModel.CreateModel(**createModelDict)

    
    #get stored NN-mbs model
    def GetNNModel(self):
        return self.nnModel

    #serial version to create data
    def CreateData(self,parameterFunction):
        nnModel = self.GetNNModel()
        isTest = parameterFunction['functionData']['isTest']
        nSamples = parameterFunction['functionData']['nSamples']
        showTests = parameterFunction['functionData']['showTests']
        cnt = parameterFunction['cnt'] #usually not needed
        flattenData = parameterFunction['functionData']['flattenData']
        SLIDEwindow = parameterFunction['functionData']['SLIDEwindow']
        
        inputVec        = nnModel.CreateInputVector(relCnt = cnt/nSamples, isTest = isTest, SLIDEwindow=SLIDEwindow)
        hiddenVec       = nnModel.CreateHiddenInit(isTest)
        [inputVec, outputVec,SLIDESteps ] = nnModel.ComputeModel(inputVec, 
                                         hiddenData=hiddenVec,SLIDEwindow=SLIDEwindow,
                                         solutionViewer = isTest and (cnt in showTests))
    
        if flattenData:
            return [[inputVec.flatten()], outputVec.flatten(), hiddenVec.flatten(),SLIDESteps]
        else:
            
            
            #print('inputs shape=', inputVec.shape)
            return [inputVec, outputVec, hiddenVec,SLIDESteps]
    
    #create input data from model function
    #if plotData > 0, the first data is plotted
    #showTests is a list of tests which are shown with solution viewer (not parallel)
    def CreateTrainingAndTestData(self, nTraining, nTest,nStepsTotal,flattenData=True,
                                  parameterFunction=None, showTests=[]):
        
        useMultiProcessing=True
        if parameterFunction is None:
            parameterFunction = self.CreateData
            useMultiProcessing=False
        else:
            parameterFunction = PVCreateData                     
            if showTests != []:
                print('CreateTrainingAndTestData: showTests does not work with multiprocessing')
                
            
        self.inputsTraining         = []
        self.targetsTraining        = []
        self.inputsTest             = []
        self.targetsTest            = []
        self.SLIDESteps            = []

        #initialization of hidden layers in RNN (optional)
        self.hiddenInitTraining     = []
        self.hiddenInitTest         = []

        self.nTraining              = nTraining
        self.nTest                  = nTest
        
        flattenData                 = self.nnModel.IsFFN()
        
        modes = ['SLIDEwindow','training', 'test']
        for mode, modeStr in enumerate(modes):
            if self.verboseMode>0:
                    print('create '+modeStr+' data ...')
                    
            if modeStr == 'SLIDEwindow': 
                nData = nTraining
            elif modeStr == 'training':
                 nData = nTraining 
            else: nData = nTest
                        
            #nData = nTraining if mode==0 elseif mode==1 else nTest

            #+++++++++++++++++++++++++++++++++++++++++++++
            #create training data
            [parameterDict, values] = ParameterVariation(parameterFunction, parameters={'cnt':(0,nData-1,nData)},
                                                         useMultiProcessing=useMultiProcessing and nData>2,
                                                         showProgress=self.verboseMode>0,
                                                         parameterFunctionData={ 'SLIDEwindow':modeStr== 'SLIDEwindow',
                                                                                'isTest':mode==2, 
                                                                                'nSamples':nData,
                                                                                'showTests':showTests, 
                                                                                'flattenData':flattenData})
            for item in values:
                if modeStr== 'SLIDEwindow':
                    self.SLIDESteps += [item[3]]
                    
                elif modeStr== 'training':
                    self.inputsTraining += [item[0]]
                    self.targetsTraining += [item[1]]
                    self.hiddenInitTraining += [item[2]]
                    
                else:
                    self.inputsTest += [item[0]]
                    self.targetsTest += [item[1]]
                    self.hiddenInitTest += [item[2]]


        #convert such that torch does not complain about initialization with lists:
        self.inputsTraining = np.stack(self.inputsTraining, axis=0)
        self.targetsTraining = np.stack(self.targetsTraining, axis=0)
        self.inputsTest = np.stack(self.inputsTest, axis=0)
        self.targetsTest = np.stack(self.targetsTest, axis=0)
        self.SLIDESteps = np.stack(self.SLIDESteps, axis=0)

        self.hiddenInitTraining = np.stack(self.hiddenInitTraining, axis=0)
        self.hiddenInitTest = np.stack(self.hiddenInitTest, axis=0)
        
        


    #save data to .npy file
    def SaveTrainingAndTestsData(self, fileName):
        fileExtension = ''
        if len(fileName) < 4 or fileName[-4:]!='.npy':
            fileExtension = '.npy'
        
        os.makedirs(os.path.dirname(fileName+fileExtension), exist_ok=True)
        
        dataDict = {}
        
        dataDict['version'] = 1 #to check correct version
        dataDict['modelName'] = self.GetNNModel().GetModelName() #to check if correct name
        dataDict['inputShape'] = self.GetNNModel().GetInputScaling().shape #to check if correct size
        dataDict['outputShape'] = self.GetNNModel().GetOutputScaling().shape #to check if correct size
        dataDict['nTraining'] = self.nTraining
        dataDict['nTest'] = self.nTest
        dataDict['inputsTraining'] = self.inputsTraining
        dataDict['targetsTraining'] = self.targetsTraining
        dataDict['inputsTest'] = self.inputsTest
        dataDict['targetsTest'] = self.targetsTest
        dataDict['SLIDESteps'] = self.SLIDESteps
        #dataDict['StepPredictor'] = self.StepPredictor

        #initialization of hidden layers in RNN (optional)
        dataDict['hiddenInitTraining'] = self.hiddenInitTraining
        dataDict['hiddenInitTest'] = self.hiddenInitTest

        dataDict['floatType'] = self.floatType
        
        #version 2 from here

        with open(fileName+fileExtension, 'wb') as f:
            np.save(f, dataDict, allow_pickle=True) #allow_pickle=True for lists or dictionaries
    
   

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


   
#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__': #include this to enable parallel processing
      #put this part out of if __name__ ... if you like to use multiprocessing
    endTime=0.25
    model = SLIDEModel(nStepsTotal=40, endTime=endTime, )
    nntc = training_data_center(nnModel=model, computeDevice='cpu')
    nntc.CreateTrainingAndTestData(nTraining=64, nTest=10,parameterFunction=PVCreateData, #for multiprocessing
                                   )
    
    





