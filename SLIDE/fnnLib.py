
from Models.Container import *


#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++

def NeuralNetworkStructureTypes():
    return [' ','L', 'LL', 'RL', 'LR','R', 'RR', 'RRR','TLTLT','EEE', 'SL', 'LS' , 'LSL', #12
            'SLSLS', 'SLS', 'TSTST','TSLTSLT','TLSLT','LTTL','TLT', 'LLR', #18, TLSLTLSLT
            'LLRL', 'LLLRL', 'LLRLRL', 'LLLL', 'LRdL','RLR','RSR','LRL', #25
                  'RdR', 'LRdLRdL', 'LLLRLRL','TdTdTdTdTdT'] #29
def GetInputFromSensorType(system): 
    if system:
        N = 10
    else:
        N = 5
        
    inputVector = []
    for i in range(1, 2**N): 
        inputVector += [[]]
        binStr = format(i, '#0{}b'.format(N+2))
        for j in range(0, N): 
            if binStr[::-1][:-2][j] == '1': 
                inputVector[-1] += [j]
    return inputVector # add all combinations you want


def GetOutputFromSensorType(system): 
    #if system:
     #   N = 10
    #else:
    N = 3
        
    outputVector = []
    for i in range(1, 2**N): 
        outputVector += [[]]
        binStr = format(i, '#0{}b'.format(N+2))
        for j in range(0, N): 
            if binStr[::-1][:-2][j] == '1': 
                outputVector[-1] += [j]
    return outputVector # add all combinations you want


# def GetStepPredictor(): 
#     N               = 25
#     StepPredictor   = []
#     for i in range(1, N + 1): 
#         StepPredictor.append([i])
#     return StepPredictor


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
def SLIDEdATA(ns, td, dataTest,StepPredictor,InputVec, Patu=None):
    
   # StepPredictor       = 1
    slices              = int(ns- td-StepPredictor)
    
    
    Inputs             = dataTest[0]
    
    inputConfig        = GetInputFromSensorType(system=Patu)[int(InputVec)]
    inputVecLong       = []
    
    for k in range(slices): 
                startIndex0 = k
                startIndex1 = k+td+StepPredictor
                endIndex    = startIndex1+StepPredictor
            
                #Sensor layer for each layer
                sensorsLayers = [] 
                for n, j in enumerate(inputConfig): 
                    sensLayer            =  Inputs[j*ns:(j+1)*ns] # Extract sensor with user configuration
                    SLIDESensor         = sensLayer[startIndex0:startIndex1] # This sensor layer is built on dampedSteps
                    sensorsLayers        = np.hstack((sensorsLayers, SLIDESensor))
                                    
                if k == 0:
                    inputVecLong = np.vstack(( sensorsLayers)).T  
                else: 
                    inputVecLong = np.vstack((inputVecLong, sensorsLayers))
    return inputVecLong

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
    P.inputVec = [0,1,2,3,4]
    P.stepPredictor = 1
    P.LyaerUnits = 200
    P.system = 'system'
    P.SLIDESteps= False
    P.nStepsTotal = 200
    P.outputVec = [0,1,2]
    
    
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
    
    # if P.sensorType != -1: 
    #     #inputVec = GetInputFromSensorType(P.system)[P.sensorType]
    #     #outputVec = GetOutputFromSensorType(P.system)[P.outputVec]
    #     # nPredictor  = GetStepPredictor()[P.nPredictor]
    #     # nPredictor  = int(nPredictor[0])

    # # if P.hiddenLayerSize != -1: 
    # #     Units = [P.hiddenLayerSize]
    
    #print(f"inputVec: {inputVec}, steppPredictor: {P.stepPredictor} ")
    
    #print(neuralNetworkTypeName )
    #++++++++++++++++++++++++++++++++++++++++++++++
    #++++++++++++++++++++++++++++++++++++++++++++++

    if P.dataFile == None:
        moduleNntc.CreateTrainingAndTestData(nTraining=P.nTraining, nTest=P.nTest,
                                       #parameterFunction=PVCreateData, #for multiprocessing
                                       )
    else:
        #print(P.hiddenLayerSize)
        moduleNntc.LoadTrainingAndTestsData(P.dataFile, system=P.system, nTraining=P.nTraining, nTest=P.nTest,nStepsTotal= P.nStepsTotal, 
                                            Input= P.sensorType, Output=P.outputVec,
                                            StepPredictor=P.stepPredictor, Units=P.LyaerUnits,
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
    #rv['lossEpoch'] = list(moduleNntc.lossLog()[:,0])
    #rv['loss'] = list(moduleNntc.lossLog()[:,1])

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
        moduleNntc.SaveNNModel(P.storeModelName+str(P.computationIndex)+'.pth')
    
    return rv #dictionary



class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, predicted, target):
        return torch.sqrt(self.mse(predicted, target))
    
    
class NeuralNetworkTrainingCenter():
    #create class with model to be trained
    #createModelDict is used to initialize CreateModel function with specific parameters
    #initialize seed
    def __init__(self, nnModel=NNtestModel(), createModelDict={}, 
                 computeDevice = 'cpu', verboseMode=1):
        
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
        
        self.nnModel.CreateModel(**createModelDict)

    
    #get stored NN-mbs model
    def GetNNModel(self):
        return self.nnModel


    #load data from .npy file
    #allows loading data with less training or test sets
    def LoadTrainingAndTestsData(self, fileName,Data=None,system=None, nTraining=None, nTest=None, nStepsTotal=None,
                                 Input=[], Output=[], 
                                 StepPredictor=None, Units=None,
                                 SLIDESteps=None):
        fileExtension = ''
        if len(fileName) < 4 or fileName[-4:]!='.npy':
            fileExtension = '.npy'
            
        with open(fileName+fileExtension, 'rb') as f:
            dataDict = np.load(f, allow_pickle=True).item()   #allow_pickle=True for lists or dictionaries; .all() for dictionaries
            
        if dataDict['version'] >= 1:
            if dataDict['modelName'] != self.GetNNModel().GetModelName(): #to check if correct name
                raise ValueError('NeuralNetworkTrainingCenter.LoadTrainingAndTestsData: model name does not match current model')
            if dataDict['inputShape'] != self.GetNNModel().GetInputScaling().shape:
                raise ValueError('NeuralNetworkTrainingCenter.LoadTrainingAndTestsData: inputShape does match current model inputSize')
            if dataDict['outputShape'] != self.GetNNModel().GetOutputScaling().shape:
                raise ValueError('NeuralNetworkTrainingCenter.LoadTrainingAndTestsData: outputShape does match current model outputShape')

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
                        
            print(system)
            inputConfiguration      = GetInputFromSensorType(system)[int(Input)]
            outputConfiguration     = GetOutputFromSensorType(system)[int(Output)]
            # Steps from damped oscillations
            if SLIDESteps: 
                
                # if Units:
                #     if isinstance(Units, list):
                #         self.SLIDESteps = int(Units[0])  #
                #     else:
                #         self.SLIDESteps = int(Units)

                # else:  if Patu: 
                if system:
                     self.SLIDESteps              = int(np.mean(dataDict['SLIDESteps'][0][1])) # EOM-based

                else:    
                     self.SLIDESteps=int(np.mean(dataDict['SLIDESteps'][0][0]))
                     #etd              = int(np.mean(dataDict['SLIDESteps'][0][1]))
                    
                
                
                
                print(f'Data arrangement using SLIDE window={self.SLIDESteps}')
                
                self.inputsTraining     = []
                self.targetsTraining    = []
                self.hiddenInitTraining = []
                self.inputsTest         = []
                self.targetsTest        = []
                self.hiddenInitTest     = []
                
                inputsTraining0  = dataDict['inputsTraining']
                targetsTraining0 = dataDict['targetsTraining']
                inputsTest0      = dataDict['inputsTest']
                targetsTest0     = dataDict['targetsTest']
    

                dataPoints       = nStepsTotal            #Per-step approach used in saving data
                slices           = int((dataPoints-StepPredictor) // self.SLIDESteps)  #Data points devision division based SLIDESteps
                
                # Create data artificially.
                #slices           = int(dataPoints- self.SLIDESteps)
                
                inputsTraining   = []
                targetsTraining  = []
                inputsTest       = []
                targetsTest      = []
                #hiddenInitTraining = []
                #hiddenInitTest  = []
                
                # Training data
                for i in range(nTraining): 
                    # Extract data from each batch step
                    batchInputTraining0  = inputsTraining0[i, 0, :]
                    batchTargetTraining0 = targetsTraining0[i,:] 
                    
                    InputDataTargets  = []
                    InputDataTraining = []
                    # Extract each slice based on SLIDESteps
                    for k in range(slices): 
                        startIndex0 = (k) * self.SLIDESteps
                        startIndex1 = (k +1) * self.SLIDESteps
                        endIndex    = startIndex1+StepPredictor
                        
                        #startIndex0 = k
                        #startIndex1 = self.SLIDESteps+k
                        
                        #print(startIndex0, startIndex1)
                        #endIndex    = startIndex1+StepPredictor
                        
                        # Extract target data for eacch slide window
                        inputLayers = []
                        
                        for n, j in enumerate(outputConfiguration):
                            TragetLayer      = batchTargetTraining0[j*dataPoints:(j+1)*dataPoints]
                            SLIDETarget      = TragetLayer[startIndex1:endIndex]
                            inputLayers      = np.hstack((inputLayers, SLIDETarget))

                        #Sensor layer for each layer
                        outputLayers = []
                        
                        # Extract sensor data for each batch 
                        for n, j in enumerate(inputConfiguration):
                            
                            sensLayer      =  batchInputTraining0[j*dataPoints:(j+1)*dataPoints] # Extract sensor with user configuration
                            SLIDEInput     = sensLayer[startIndex0:endIndex] # This sensor layer is built on SLIDESteps
                            outputLayers   = np.hstack((outputLayers, SLIDEInput))
                                                    
                        if k == 0:
                            InputDataTraining = np.vstack(( outputLayers)).T 
                            InputDataTargets  =  np.vstack((inputLayers)).T
                            
                            # InputDataTraining = sensorsLayers
                            # InputDataTargets  = sensTarget
                        else: 
                            InputDataTraining = np.vstack((InputDataTraining, outputLayers))
                            InputDataTargets  = np.vstack((InputDataTargets, inputLayers))

                            
                    if i == 0:
                        #self.inputsTraining  = InputDataTraining.reshape(InputDataTraining.shape[0], 1, InputDataTraining.shape[1])
                        self.inputsTraining  = InputDataTraining.reshape(InputDataTraining.shape[0], 1, InputDataTraining.shape[1])
                        self.targetsTraining = InputDataTargets
                    else: 
                        InputDataTraining       = InputDataTraining.reshape(InputDataTraining.shape[0], 1, InputDataTraining.shape[1])
                        self.inputsTraining     = np.vstack((self.inputsTraining, InputDataTraining))
                        self.targetsTraining    = np.vstack((self.targetsTraining, InputDataTargets))
                                    
                # Test data
                for i in range(nTest): 
                    # Extract data from each batch step
                    batchInputTest0  = inputsTest0[i, 0, :]
                    batchTargetTest0 = targetsTest0[i,:] 
                    
                    InputDataTargets  = []
                    InputDataTraining = []
                    # Extract each slice based on SLIDESteps
                    for k in range(slices): 
                        startIndex0 = (k) * self.SLIDESteps
                        startIndex1 = (k + 1) * self.SLIDESteps
                        endIndex    = startIndex1+StepPredictor
                        
                        #startIndex0 = k
                        #startIndex1 = self.SLIDESteps+k
                        #endIndex    = startIndex1+StepPredictor
                        
                        # Extract target data for eacch slide window
                        inputLayers = []
                        
                        for n, j in enumerate(outputConfiguration):
                            TragetLayer      = batchTargetTraining0[j*dataPoints:(j+1)*dataPoints]
                            SLIDETarget      = TragetLayer[startIndex1:endIndex]
                            inputLayers      = np.hstack((inputLayers, SLIDETarget))

                        #Sensor layer for each layer
                        outputLayers = []
                        
                        # Extract sensor data for each batch 
                        for n, j in enumerate(inputConfiguration):
                            
                            sensLayer      =  batchInputTraining0[j*dataPoints:(j+1)*dataPoints] # Extract sensor with user configuration
                            SLIDEInput     = sensLayer[startIndex0:endIndex] # This sensor layer is built on SLIDESteps
                            outputLayers   = np.hstack((outputLayers, SLIDEInput))
                       
                        if k == 0:
                            InputDataTraining = np.vstack((outputLayers)).T 
                            InputDataTargets  = np.vstack((inputLayers)).T
                        else: 
                            InputDataTraining = np.vstack((InputDataTraining, outputLayers))
                            InputDataTargets  = np.vstack((InputDataTargets, inputLayers))

                            
                    if i == 0:
                        self.inputsTest  = InputDataTraining.reshape(InputDataTraining.shape[0], 1, InputDataTraining.shape[1])
                        self.targetsTest = InputDataTargets
                    else: 
                        InputDataTraining    = InputDataTraining.reshape(InputDataTraining.shape[0], 1, InputDataTraining.shape[1])
                        self.inputsTest  = np.vstack((self.inputsTest, InputDataTraining))
                        self.targetsTest = np.vstack((self.targetsTest, InputDataTargets))
                        
 
                self.hiddenInitTraining = np.zeros((self.inputsTraining.shape[0], 1))
                self.hiddenInitTest     = np.zeros((self.targetsTest.shape[0], 1))

    
            else:
                
                self.inputsTraining = dataDict['inputsTraining'][:nTraining]
                self.targetsTraining = dataDict['targetsTraining'][:nTraining]
                self.hiddenInitTraining = dataDict['hiddenInitTraining'][:nTraining]
                self.inputsTest = dataDict['inputsTest'][:nTest]
                self.targetsTest = dataDict['targetsTest'][:nTest]
                self.hiddenInitTest = dataDict['hiddenInitTest'][:nTest]

                self.floatType = torch.float
                if 'floatType' in dataDict:
                    self.floatType = dataDict['floatType']
        

        
        # print('inputs shape=', self.inputsTraining[0].shape)

        #convert such that torch does not complain about initialization with lists:
                self.inputsTraining = np.stack(self.inputsTraining, axis=0)
                self.targetsTraining = np.stack(self.targetsTraining, axis=0)
                self.inputsTest = np.stack(self.inputsTest, axis=0)
                self.targetsTest = np.stack(self.targetsTest, axis=0)
                self.hiddenInitTraining = np.stack(self.hiddenInitTraining, axis=0)
                self.hiddenInitTest = np.stack(self.hiddenInitTest, axis=0)

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
        [inputSize, outputSize] = self.nnModel.GetInputOutputSizeNN()
        
        inputSize =self.inputsTraining.shape[2]
        outputSize=self.targetsTraining.shape[1]
        
        #print('size=',[inputSize, outputSize])
        # inputSize = self.nnModel.GetInputScaling().shape[self.nnModel.GetInputScaling().ndim-1]  #OLD: len(...)
        # outputSize = self.nnModel.GetOutputScaling().shape#[self.nnModel.GetOutputScaling().ndim-1]

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
        
        # Convert your data to PyTorch tensors and create a DataLoader
        inputs = torch.tensor(self.inputsTraining, dtype=self.floatType, requires_grad=True).to(self.computeDevice)#,non_blocking=True)
        #inputs = inputs[:, :-5]
        
        #print(inputs.shape)
        targets = torch.tensor(self.targetsTraining, dtype=self.floatType, requires_grad=True).to(self.computeDevice)#,non_blocking=True)
        
        #print(targets.shape)
        #targets = targets[:, -1:]
        #self.hiddenInitTraining=torch.zeros((inputs.shape[0], 0))
        
        #hiddenInit=torch.zeros((inputs.shape[0], 1), requires_grad=True)
        
        hiddenInit = torch.tensor(self.hiddenInitTraining, dtype=self.floatType, requires_grad=True).to(self.computeDevice)#,non_blocking=True)
        
        
        inputsTest = torch.tensor(self.inputsTest, dtype=self.floatType).to(self.computeDevice)
        #inputsTest = inputsTest[:, :-5]
        targetsTest = torch.tensor(self.targetsTest, dtype=self.floatType).to(self.computeDevice)
        #targetsTest = targetsTest[:, -1:]
        hiddenInitTest = torch.tensor(self.hiddenInitTest, dtype=self.floatType,requires_grad=True).to(self.computeDevice)
        
        # hiddenInitTest=torch.zeros((inputsTest.shape[0], 0))

        # print('torch NNTC device=',self.computeDevice)

        # print('inputSize = ', inputSize)
        # print('inputs[-1] = ', inputs.size(-1))
        # print('outputSize = ', outputSize)
        # print('targets[-1] = ', targets.size(-1))

        dataset = TensorDataset(inputs.requires_grad_(), targets.requires_grad_(), hiddenInit.requires_grad_())
        dataloader = DataLoader(dataset, batch_size=batchSize, 
                                shuffle=self.dataLoaderShuffle)
        if batchSizeIncrease != 1:
            dataloader2 = DataLoader(dataset, batch_size=batchSize*batchSizeIncrease,
                                    shuffle=self.dataLoaderShuffle)

        datasetTest = TensorDataset(inputsTest, targetsTest, hiddenInitTest)
        dataloaderTest = DataLoader(datasetTest, batch_size=batchSize, shuffle=self.dataLoaderShuffle)

        
        # Define a loss function and an optimizer
        
        #optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.learningRate,weight_decay=2e-6) #5e-6
        optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.learningRate)
        
        self.lossLog = []
        self.trainResults = [[], [], [], [], [],[]] #epoch, tests, min, mean, max, mae
        self.testResults = [[], [], [], [], [],[]] #epoch, tests, min, mean, max, mae

        l1_lambda =5e-9#Optimal values: 1e-5, 5e-9, 16,2.0e-9
        max_grad_norm = 1
        
        #accumulation_steps= 1
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
                # self.rnn.SetInitialHiddenStates(initial_hidden_states)
                # inputs.requires_grad_() # = True
                # targets.requires_grad_()
                
                # optimizer.zero_grad() if i % accumulation_steps == 0 else None
                outputs = self.rnn(inputs)
                if not self.nnModel.IsFFN():
                     outputs = outputs.view(-1, *outputSize)  # Reshape the outputs to match the target shape
                
                loss = self.lossFunction(outputs, targets)
                #maeTrain  =  torch.mean(torch.abs(outputs - targets))
                #maeTrain  = float(maeTrain)
                #optimizer.zero_grad() #(set_to_none=True)
                #loss.backward()
                #optimizer.step()
                
                
                ########################################
                #loss = self.lossFunction(outputs, targets)/ accumulation_steps
                
                # #L1 regularization
                #l1_norm = sum(p.abs().sum() for p in self.rnn.parameters())
                #loss = loss + (l1_lambda) * l1_norm 
                ##############################################
                maeTrain  =  torch.mean(torch.abs(outputs - targets))
                maeTrain  = float(maeTrain)
                
                ##############################################
                         
                # # Backward pass and optimization
                optimizer.zero_grad() #(set_to_none=True)
                loss.backward() 
                clip_grad_norm_(self.rnn.parameters(), max_grad_norm)
                optimizer.step()
                # if (i + 1) % accumulation_steps == 0:
                #      optimizer.step()
                
                ##############################################

            currentLoss = loss.item()
            trainresults += [float(loss)]

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
                        # outputs = outputs.view(-1, *outputSize)  # Reshape the outputs to match the target shape
                        mse =  self.lossFunction(outputs, targets)
                        maeTest = torch.mean(torch.abs(outputs - targets))
                        
                        #print(outputs.view(-1, *outputSize))

                        # for i in range(self.nTest):
    
                        #     inputVec = self.inputsTest[i:i+1]
                            
                        #     x = torch.tensor(inputVec, dtype=self.floatType).to(self.computeDevice)
                        #     y = np.array(self.rnn(x).to(self.computeDevice).tolist()[0]) #convert output to list
                        #     yRef = self.targetsTest[i:i+1][0]
                            
                        #     print('x.size=',x.size())
                        #     print('y.size=',y.shape)
                        #     print('yRef.size=',yRef.shape)
    
                        #     #this operation fully runs on CPU:
                        #     mse = self.lossFunction(torch.tensor(y, dtype=self.floatType), 
                        #                             torch.tensor(yRef, dtype=self.floatType))
                        
                        testresults += [float(mse)] #np.linalg.norm(y-yRef)**2/len(y)]
                        maeTest     = float(maeTest)
                        
                    
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
                mseStr     = "{:.3e}".format(mse)
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
        self.rnn = self.rnn.load_state_dict(torch.load(fileName))

    #plot loss and tests over epochs, using stored data from .npy file
    #dataFileTT is the training and test data file name
    def PlotTrainingResults(self, resultsFileNpy, dataFileName, plotLoss=True, plotTests=True, 
                            plotTestsMaxError=True, plotTestsMeanError=True,
                            sizeInches=[12.8,12.8], Patu=True, nTrain=None,
                            Sensors=None ,Liftload = None, string=None,  closeAll=True, 
                            ):
        #load and visualize results:
        with open(resultsFileNpy, 'rb') as f:
            dataDict = np.load(f, allow_pickle=True).item()   #allow_pickle=True for lists or dictionaries; .all() for dictionaries

        #self.LoadTrainingAndTestsData(dataFileName, dataDict)

        if closeAll:
            PlotSensor(None,closeAll=True)

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
                        ax.set_ylim(10**-6, 10**0)
                        # Define x-ticks and y-ticks
                        x_ticks = [0,      2*(x_max/4) ,x_max] #1*(x_max/4),3*(x_max/4)
                        y_ticks = [10**-6,  10**-3,     10**0]
                
                        ax.set_xticks(x_ticks)
                        ax.set_yticks(y_ticks)
                
                        # Set custom tick labels
                        ax.set_xticklabels([r'$0$', rf'${2*(x_max/4):.0f}$'
                                    , rf'${4*(x_max/4):.0f}$']) #rf'${x_max/4:.0f}$',rf'${3*(x_max/4):.0f}$'
                        # ax.set_yticklabels([r'$10^{-6}$',r'$10^{-4}$', r'$10^{-2}$',
                        #                     r'$10^{0}$'])
                
                        ax.set_yticklabels([r'$10^{-6}$',r'$10^{-3}$', r'$10^{0}$'])
                
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
                
                
                
                # PlotSensor(None, [dataTest], xLabel='Number of epochs', yLabel='Loss (Log Scale)', labels=[sLabel],
                #             newFigure=i==0, colorCodeOffset=cntColor, 
                #             markerStyles=markerStyles, markerSizes=markerSizes, 
                #             sizeInches=sizeInches,fileName = 'solution/Trainingloss_plot.pdf', logScaleY=True)
                

        # cntColor = -1
        # lastCase = 0
        # for i, v in enumerate(values):
        #     if 'case' in parameters:
        #         case = parameterDict['case'][i]
        #         markerStyles=[listMarkerStyles[case]]
        #         if case < lastCase or cntColor < 0:
        #             cntColor += 1
        #         lastCase = case
        #     else:
        #         cntColor += 1

        #     sLabel = labelsList[i]
        #     if plotTests:
        #         dataTest = np.vstack((v['testResultsEpoch'],v['testResultsMean'] )).T
        #         dataTestMax = np.vstack((v['testResultsEpoch'],v['testResultsMax'] )).T
                
        #         if plotTestsMaxError:
        #             PlotSensor(None, [dataTestMax], xLabel='Number of epochs', yLabel='Loss (Log Scale)', labels=[sLabel+'(max)'],
        #                        newFigure=i==0, colorCodeOffset=cntColor, lineWidths=[1], 
        #                        markerStyles=markerStyles, markerSizes=markerSizes, 
        #                        sizeInches=sizeInches,fileName = 'solution/TestMax_error.pdf', logScaleY=True)

        #         if plotTestsMeanError:
        #             PlotSensor(None, [dataTest], xLabel='Number of epochs', yLabel='test mean square error', labels=[sLabel+'(mean)'],
        #                        newFigure=(not plotTestsMaxError and i==0), colorCodeOffset=cntColor, lineWidths=[2], 
        #                        markerStyles=markerStyles, markerSizes=markerSizes, 
        #                        sizeInches=sizeInches, fileName = 'solution/TestMean_error.pdf', logScaleY=True)


    #evaluate all training and test data; plot some tests
    #nTrainingMSE avoids large number of training evaluations for large data sets
    def EvaluateModel(self, plotTests=[], plotTrainings=[], plotVars=['time','ODE2'], 
                      plotInputs=[],
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
 


#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__': #include this to enable parallel processing

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #put this part out of if __name__ ... if you like to use multiprocessing
    endTime=0.25
    model = NonlinearOscillator(useVelocities=True, useInitialValues=True, 
                                nStepsTotal=40, endTime=endTime, )
    #MyNeuralNetwork()
    nntc = NeuralNetworkTrainingCenter(nnModel=model, computeDevice='cpu')
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    nntc.CreateTrainingAndTestData(nTraining=64, nTest=10,
                                   #parameterFunction=PVCreateData, #for multiprocessing
                                   )
    
    nntc.TrainModel(maxEpochs=100)
    
    nntc.EvaluateModel(plotTests=[0,1,2,3,4], plotTrainings=[0,1,2,3,4], plotVars=['time','ODE2'])





