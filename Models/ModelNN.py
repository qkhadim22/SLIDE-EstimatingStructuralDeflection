#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

#This files defines the simulation of hydraulically actuated flexible structure.

# Author        : Qasim Khadim
# Contact       : qasim.khadim@outlook.com,qkhadim22 (Github)
# Dated         : 02-05-2023
# Organization  : University of Oulu in the collaboration of LUT University and University of Innsbruck.

# Copyright     :
#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

# import sys
# sys.exudynFast = True #this variable is used to signal to load the fast exudyn module

import sys
sys.exudynFast = True #this variable is used to signal to load the fast exudyn module

import exudyn as exu
from exudyn.itemInterface import *
from exudyn.utilities import *
from exudyn.FEM import *

import time

import matplotlib.pyplot as plt
from exudyn.plot import PlotSensor, listMarkerStyles
from exudyn.signalProcessing import GetInterpolatedSignalValue
from exudyn.physics import StribeckFunction



import scipy.io 
from scipy.optimize import fsolve, newton

import os
import numpy as np

import math as mt
from math import sin, cos, sqrt, pi, tanh, atan2,degrees


from fnnModels import ModelComputationType, NNtestModel

import os, sys
#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

# physical parameters
g           = [0, -9.8066, 0]  # Gravity
tEnd        = 10  # simulation time
h           = 25e-3  # step size
nSteps      = int(tEnd/h)+2
colLift     = color4blue

L1              = 0.365    # Length in x-direction
H1              = 1.4769      # Height in y-direction
W1              = 0.25    # Width in z-direction

bodyDim1        = [L1, H1, W1]  # body dimensions
pMid1           = np.array([-0.017403, 0.577291, 0])  # center of mass, body0
PillarP         = np.array([0, 0, 0])
LiftP           = np.array([-0.09, 1.4261, 0])
TiltL           = LiftP + np.array([2.879420180699481, -0.040690041435711005, 0])

# Second Body: LiftBoom
L2              = 3.01055           # Length in x-direction
H2              = 0.45574           # Height in y-direction
W2              = 0.263342          # Width in z-direction


L3              = 2.580         # Length in x-direction
H3              = 0.419         # Height in y-direction
W3              = 0.220         # Width in z-direction

L4              = 0.557227    # Length in x-direction
H4              = 0.1425      # Height in y-direction
W4              = 0.15        # Width in z-direction

L5              = 0.569009       # Length in x-direction
H5              = 0.078827       # Height in y-direction
W5              = 0.15           # Width in z-direction
        
pS              = 140e5
pT              = 1e5                           # Tank pressure
Qn1             = (18/60000)/((9.9)*sqrt(35e5))                      # Nominal flow rate of valve at 18 l/min under
Qn2             = (35/60000)/((9.9)*sqrt(35e5))                      # Nominal flow rate of valve at 18 l/min under
Qn               = 1.667*10*2.1597e-08                     # Nominal flow rate of valve at 18 l/min under

# Cylinder and piston parameters
L_Cyl1          = 820e-3                            # Cylinder length
D_Cyl1          = 100e-3                            # Cylinder dia
A_1             = (pi/4)*(D_Cyl1)**2                # Area of cylinder side
L_Pis1          = 535e-3                            # Piston length, also equals to stroke length
d_pis1          = 56e-3                             # Piston dia
A_2             = A_1-(pi/4)*(d_pis1)**2            # Area on piston-rod side
L_Cyl2          = 1050e-3                           # Cylinder length
L_Pis2          = 780e-3                            # Piston length, also equals to stroke length
d_1             = 12.7e-3                         # Dia of volume 1
V1              = (pi/4)*(d_1)**2*1.5             # Volume V1 = V0
V2              = (pi/4)*(d_1)**2*1.5             # Volume V2 = V1
A               = [A_1, A_2]
Bh              = 700e6
Bc              = 2.1000e+11
Bo              = 1650e6
Fc              = 210
Fs              = 300
sig2            = 330
vs              = 5e-3


# Loading Graphics of bodies
fileName1       = 'LowQualityPATU/Pillar.stl'
fileName2       = 'LowQualityPATU/LiftBoom.stl'
fileName3       = 'LowQualityPATU/TiltBoom.stl'
fileName4       = 'LowQualityPATU/Bracket1.stl'
fileName5       = 'LowQualityPATU/Bracket2.stl'
fileName6       = 'LowQualityPATU/ExtensionBoom.stl'


#fileNameT       = 'TiltBoomANSYS/TiltBoom' #for load/save of FEM data

feL             = FEMinterface()
feT             = FEMinterface()



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class NNHydraulics():

    #initialize class 
    def __init__(self, nStepsTotal=100, endTime=0.5, ReD = 3.35e-3, mL= 50, nnType='FFN',
                 nModes = 2, 
                 loadFromSavedNPY=True,
                 visualization = False,
                 system = True,
                 useFriction=True,
                 verboseMode = 0):

        NNtestModel.__init__(self)

        self.nStepsTotal = nStepsTotal
        self.nnType = nnType
        self.endTime = endTime
        self.useFriction = useFriction
        
        #+++++++++++++++++++++++++++++
        #from hydraulics:
        self.nModes = nModes
        self.loadFromSavedNPY = loadFromSavedNPY
        self.StaticCase = False
        self.Hydraulics = True
        self.Visualization = visualization
        self.Plotting = False
        self.FineMesh = True
        self.TrainingData= False

        #+++++++++++++++++++++++++++++
        
        self.ReD    = ReD
        self.mL     = mL

        self.p1     = 1.231689452311020e+07
        self.p2     = 3.161621091660600e+06
        self.p3     = 4.064941403563620e+06
        self.p4     = 1.063232421172350e+07
        
        self.p1Init = 7e6
        self.p2Init = 7e6
        
        self.angleMinDeg1 = -10
        self.angleMaxDeg1 = 50
        
        self.angleMinDeg2 = -65
        self.angleMaxDeg2 = 0
        
        self.nOutputs = 1    #tip deflection; number of outputs to be predicted
        #self.nInputs = 5    #U, p0, p1, s, s_t
        
        self.nInputs = 10    #U, p0, s, p1, s_t
        self.numSensors = 10

        self.verboseMode = verboseMode

        self.modelName = 'hydraulics'
        self.modelNameShort = 'hyd'

        self.scalPressures1   = 2.00e7 #this is the approx. size of pressures
        self.scalPressures2   = 2.00e7 #this is the approx. size of pressures

        self.scalStroke1      = L_Cyl1+L_Pis1
        self.scaldStroke1     = 0.1
        
        self.scalStroke2      = L_Cyl2+L_Pis2
        self.scaldStroke2     = 0.15
        
        self.scalU1          = 1
        self.scalU2          = 1
        self.scalOut         = 2000e-6
        
        if system:
            self.inputScaling = np.hstack((self.scalU1*np.ones(1*(self.nStepsTotal)), #U1
                                       self.scalU2*np.ones(1*(self.nStepsTotal)),     #U2
                                       self.scalStroke1*np.ones(1*(self.nStepsTotal)), #s1
                                       self.scalStroke2*np.ones(1*(self.nStepsTotal)), #s2
                                       self.scaldStroke1*np.ones(1*(self.nStepsTotal)), #ds1
                                       self.scaldStroke2*np.ones(1*(self.nStepsTotal)), #ds2
                                       self.scalPressures1*np.ones(1*(self.nStepsTotal)), #p1
                                       self.scalPressures1*np.ones(1*(self.nStepsTotal)), #p2
                                       self.scalPressures2*np.ones(1*(self.nStepsTotal)), #p3
                                       self.scalPressures2*np.ones(1*(self.nStepsTotal)) #p4
                                       )) 
        else:
            self.inputScaling = np.hstack((self.scalU1*np.ones(1*(self.nStepsTotal)),       #U1
                                           self.scalStroke1*np.ones(1*(self.nStepsTotal)),  #s1
                                           self.scaldStroke1*np.ones(1*(self.nStepsTotal)), #ds1 
                                           self.scalPressures1*np.ones(1*(self.nStepsTotal)), #p1
                                           self.scalPressures1*np.ones(1*(self.nStepsTotal)) #p2
                                               ))
            
        
        self.outputScaling = self.scalOut*np.ones((self.nStepsTotal, self.nOutputs))
        
        self.timeVecOut = np.arange(1,self.nStepsTotal+1)/self.nStepsTotal*self.endTime
        

    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
    
    def CreateModel(self):
        self.SC  = exu.SystemContainer()
        self.mbs = self.SC.AddSystem()
        
        #mbs can be used to transfer data ...
        

        

    #get time vector according to output data
    def GetOutputXAxisVector(self):
        return self.timeVecOut

    #create a randomized input vector
    #relCnt can be used to create different kinds of input vectors (sinoid, noise, ...)
    #isTest is True in case of test data creation
    
    def create_random_input_signal(self, SLIDEwindow=False):
        ten_percent             = int(0.1 * self.nStepsTotal)  
        # 20 % step signals
        segment1               = np.ones(2*ten_percent)               # 20% of nStepsTotal at 1  
        segment2               = -1 * np.ones(2*ten_percent)          # 20% of nStepsTotal at -1
        segment3               = np.zeros(2*ten_percent)
        
        if SLIDEwindow==False: 
            num_segments = np.random.randint(5, 1*ten_percent) #randomly selecting the number of segments 
            segment_lengths = np.zeros(num_segments, dtype=int)
            remaining_length = 2*ten_percent - num_segments

            for i in range(num_segments - 1):
                # Randomly allocate a portion of the remaining length to this segment
                segment_lengths[i] = 1 + np.random.randint(0, remaining_length)
                remaining_length -= (segment_lengths[i] - 1)
                
            segment_lengths[-1] = 1 + remaining_length
            start_values = np.random.uniform(1, -1, num_segments)
            end_values = np.random.uniform(-1, 1, num_segments)
            ramp = []
            for i in range(num_segments):
                segment_length = np.random.randint(5, ten_percent)
                segment = np.linspace(start_values[i], end_values[i], segment_lengths[i], endpoint=False)
                ramp.append(segment)
                
            segment4 = np.concatenate(ramp)   

            segment5 = np.random.uniform(low=-1, high=1, size=2*ten_percent)
            segment5 = segment5[(segment5 != -1) & (segment5 != 0) & (segment5 != 1)]

            # Concatenate all segments
            segments = [segment1, segment2, segment3, segment4, segment5] 
            np.random.shuffle(segments)

            random_signal = np.concatenate(segments)
            
            def custom_shuffle(array, frequency):
                                    
                for _ in range(frequency):
                    # Randomly select two indices to swap
                    i, j = np.random.randint(0, self.nStepsTotal , size=2)
                    # Swap elements
                    array[i], array[j] = array[j], array[i]
                    
            #Swapping frequency
            num_swaps = np.random.randint(1, 0.1*self.nStepsTotal)  #self.nStepsTotal
            custom_shuffle(random_signal, num_swaps)
        else:
            segments = [segment1, np.zeros(8*ten_percent)]
            random_signal         = np.concatenate(segments)   
        return random_signal
    
    def CreateLiftBoomInputVector(self, relCnt = 0, isTest=False, SLIDEwindow=False):
        #in this case, the input is based on 
        vec = np.zeros(self.GetInputScaling().shape)
        U  = np.zeros(self.nStepsTotal)
        U   = self.create_random_input_signal(SLIDEwindow)
        angleInit1  = np.random.rand()*(self.angleMaxDeg1-self.angleMinDeg1)+self.angleMinDeg1

        vec[0:self.nStepsTotal] = U
        vec[self.nStepsTotal*2] = angleInit1 #workaround to initialize model: use initial angle
        vec[self.nStepsTotal*4] = 0 #initial velocity
        
        return vec
    
    def CreatePATUInputVector(self, relCnt = 0, isTest=False, SLIDEwindow=False):
        
        vec = np.zeros(self.GetInputScaling().shape)
    
        U1  = np.zeros(self.nStepsTotal)
        U2  = np.zeros(self.nStepsTotal)
        
        U1          = self.create_random_input_signal(SLIDEwindow)
        U2          = self.create_random_input_signal(SLIDEwindow)
        angleInit1  = np.random.rand()*(self.angleMaxDeg1-self.angleMinDeg1)+self.angleMinDeg1
        angleInit2  = np.random.rand()*(self.angleMaxDeg2-self.angleMinDeg2)+self.angleMinDeg2
        vec[0:self.nStepsTotal]                     = U1
        vec[self.nStepsTotal:2*self.nStepsTotal]    = U2 
        vec[self.nStepsTotal*3]                     = angleInit1 
        vec[self.nStepsTotal*4]                     = angleInit2 
        
        
        return vec
    
    
    def InputVector_Simulation(self, relCnt = 0, isTest=False, SLIDEwindow=False):
        
        vec = np.zeros(self.GetInputScaling().shape)
    
        U1  = np.zeros(self.nStepsTotal)
        U2  = np.zeros(self.nStepsTotal)

        def create_random_input_signal():
            
            ten_percent             = int(0.1 * self.nStepsTotal)  
            
            # 20 % step signals
            segment1               = np.ones(2*ten_percent)               # 20% of nStepsTotal at 1  
            segment2               = -1 * np.ones(2*ten_percent)          # 20% of nStepsTotal at -1
            segment3               = np.zeros(2*ten_percent)
            
            if SLIDEwindow==False: 

                # 20 % ramp signal: randomly between 1, -1
                num_segments = np.random.randint(5, 1*ten_percent) #randomly selecting the number of segments 
                segment_lengths = np.zeros(num_segments, dtype=int)
                remaining_length = 2*ten_percent - num_segments   


                for i in range(num_segments - 1):
                    # Randomly allocate a portion of the remaining length to this segment
                    segment_lengths[i] = 1 + np.random.randint(0, remaining_length)
                    remaining_length -= (segment_lengths[i] - 1)

                segment_lengths[-1] = 1 + remaining_length
                
                start_values = np.random.uniform(1, -1, num_segments)
                end_values = np.random.uniform(-1, 1, num_segments)

                ramp = []
                for i in range(num_segments):
                    segment_length = np.random.randint(5, ten_percent)
                    segment = np.linspace(start_values[i], end_values[i], segment_lengths[i], endpoint=False)
                    ramp.append(segment)
                    
                segment4 = np.concatenate(ramp)   

                # 30 % random values between -1 and 1
                segment5 = np.random.uniform(low=-1, high=1, size=2*ten_percent)
                segment5 = segment5[(segment5 != -1) & (segment5 != 0) & (segment5 != 1)]
            
                # Concatenate all segments
                segments = [segment1, segment2, segment3, segment4, segment5] 
                np.random.shuffle(segments)
    
                random_signal = np.concatenate(segments)
                #np.random.shuffle(random_signal)
                
                def custom_shuffle(array, frequency):
                                        
                    for _ in range(frequency):
                        # Randomly select two indices to swap
                        i, j = np.random.randint(0, self.nStepsTotal , size=2)
                        # Swap elements
                        array[i], array[j] = array[j], array[i]
                        
                
                #Swapping frequency
                num_swaps = np.random.randint(1, 0.1*self.nStepsTotal)  #self.nStepsTotal
                custom_shuffle(random_signal, num_swaps)


            else:
                        
                segments = [segment1, np.zeros(8*ten_percent)]
                random_signal         = np.concatenate(segments)
                
            return random_signal
        
        U1          = create_random_input_signal()
        U2          = 1*create_random_input_signal()
        angleInit1  = np.random.rand()*(self.angleMaxDeg1-self.angleMinDeg1)+self.angleMinDeg1
        angleInit2  = np.random.rand()*(self.angleMaxDeg2-self.angleMinDeg2)+self.angleMinDeg2
        
        #angleInit1  = 14.6
        #angleInit2  = -58.8 #np.deg2rad(-58.8)

        vec[0:self.nStepsTotal]                     = U1
        vec[self.nStepsTotal:2*self.nStepsTotal]    = U2 
        vec[self.nStepsTotal*3]                     = angleInit1 
        vec[self.nStepsTotal*4]                     = angleInit2 
        
        
        return vec
    


            
    #get number of simulation steps
    def GetNSimulationSteps(self):
        return self.nStepsTotal # x finer simulation than output

    #split input data into initial values, forces or other inputs
    #return dict with 'data' and possibly 'initialODE2' and 'initialODE2_t'
    def SplitLiftInputData(self, inputData):
        rv = {}

        rv['U'] = inputData[0:self.nStepsTotal] 
        rv['s'] = inputData[1*(self.nStepsTotal):2*(self.nStepsTotal)]
        rv['ds'] = inputData[2*(self.nStepsTotal):3*(self.nStepsTotal)]  
        rv['p1'] = inputData[3*(self.nStepsTotal):4*(self.nStepsTotal)]
        rv['p2'] = inputData[4*(self.nStepsTotal):5*(self.nStepsTotal)]
        return rv
    
    def SplitPATUInputData(self, inputData):
        rv = {}
        
        # rv['t']          = inputData[0:self.nStepsTotal]       
        rv['U1']         = inputData[0*(self.nStepsTotal):1*(self.nStepsTotal)]      
        rv['U2']         = inputData[1*(self.nStepsTotal):2*(self.nStepsTotal)]   
        rv['s1']         = inputData[2*(self.nStepsTotal):3*(self.nStepsTotal)]    
        rv['s2']         = inputData[3*(self.nStepsTotal):4*(self.nStepsTotal)]       
        rv['ds1']        = inputData[4*(self.nStepsTotal):5*(self.nStepsTotal)]    
        rv['ds2']        = inputData[5*(self.nStepsTotal):6*(self.nStepsTotal)]
        rv['p1']         = inputData[6*(self.nStepsTotal):7*(self.nStepsTotal)]    
        rv['p2']         = inputData[7*(self.nStepsTotal):8*(self.nStepsTotal)]
        rv['p3']         = inputData[8*(self.nStepsTotal):9*(self.nStepsTotal)]       
        rv['p4']         = inputData[9*(self.nStepsTotal):10*(self.nStepsTotal)]       

        # rv['theta1']    = inputData[2*(self.nStepsTotal):3*(self.nStepsTotal)]
         
        # rv['data'] = data
        return rv
    

    #split output data to get ODE2 values (and possibly other data, such as ODE2)
    #return dict {'ODE2':[], 'ODE2_t':[]}
    def SplitOutputData(self, outputData):
        rv = {}
        
        rv['uTip'] = outputData[0*self.nStepsTotal:self.nStepsTotal]
        return rv
    

    #initialState contains position and velocity states as list of two np.arrays 
    def ComputeModel(self, inputData,hiddenData=None, SLIDEwindow=None, Patu=None, 
                                                     verboseMode = 0, solutionViewer = False):
        self.CreateModel()
        # print('compute model')
        self.verboseMode = verboseMode
        #set input data ...
        
        self.inputTimeU1 = np.zeros((self.nStepsTotal,2))
        self.inputTimeU2 = np.zeros((self.nStepsTotal,2))
        self.inputTimeU1[:,0] = self.timeVecOut

        self.n_td = np.array([0])
        if Patu==False:
            inputDict = self.SplitLiftInputData(np.array(inputData))
            self.inputTimeU1[:,1] = inputDict['U']
            self.mbs.variables['inputTimeU1'] = self.inputTimeU1            
            self.mbs.variables['theta1'] = inputData[self.nStepsTotal*2]
            
            self.FlexibleLiftBoom(self.mbs.variables['theta1'], self.p1Init, self.p2Init)
            
            if SLIDEwindow:
                print('Computing SLIDE window')
                self.ComputeSLIDE(Patu)
                
            DS = self.dictSensors
            sensorTip = self.dictSensors['sensorTip']
            inputData[0:self.nStepsTotal] =  inputData[0:self.nStepsTotal]
            inputData[1*self.nStepsTotal:2*self.nStepsTotal] = self.mbs.GetSensorStoredData(DS['sDistance'])[0:self.nStepsTotal,1]
            inputData[2*self.nStepsTotal:3*self.nStepsTotal] = self.mbs.GetSensorStoredData(DS['sVelocity'])[0:self.nStepsTotal,1]
            inputData[3*self.nStepsTotal:4*self.nStepsTotal] = self.mbs.GetSensorStoredData(DS['sPressures'])[0:self.nStepsTotal,1]
            inputData[4*self.nStepsTotal:5*self.nStepsTotal] = self.mbs.GetSensorStoredData(DS['sPressures'])[0:self.nStepsTotal,2]

       
        else: 
            
            inputDict = self.SplitPATUInputData(np.array(inputData))
            self.inputTimeU1[:,1] = inputDict['U1']
            self.mbs.variables['inputTimeU1'] = self.inputTimeU1 
            self.inputTimeU2[:,0] = self.timeVecOut
            self.inputTimeU2[:,1] = inputDict['U2']
            self.mbs.variables['inputTimeU2'] = self.inputTimeU2
            self.mbs.variables['theta1'] = inputData[self.nStepsTotal*3]
            self.mbs.variables['theta2'] = inputData[self.nStepsTotal*4]
            
            self.PatuCrane(self.mbs.variables['theta1'], self.mbs.variables['theta2'],
                                   self.p1, self.p2, self.p3, self.p4)
            if SLIDEwindow:
                print('Computing SLIDE window')
                self.ComputeSLIDE(Patu)
                
            DS = self.dictSensors
            
            inputData[0:self.nStepsTotal]     =  inputData[0:self.nStepsTotal]
            inputData[1*self.nStepsTotal:2*self.nStepsTotal]     =  inputData[1*self.nStepsTotal:2*self.nStepsTotal]
            inputData[2*self.nStepsTotal:3*self.nStepsTotal]     =  self.mbs.GetSensorStoredData(DS['sDistance1'])[0:1*self.nStepsTotal,1]
            inputData[3*self.nStepsTotal:4*self.nStepsTotal]     =  self.mbs.GetSensorStoredData(DS['sDistance2'])[0:1*self.nStepsTotal,1]
            inputData[4*self.nStepsTotal:5*self.nStepsTotal]     =  self.mbs.GetSensorStoredData(DS['sVelocity1'])[0:1*self.nStepsTotal,1]
            inputData[5*self.nStepsTotal:6*self.nStepsTotal]     =  self.mbs.GetSensorStoredData(DS['sVelocity2'])[0:1*self.nStepsTotal,1]       
            inputData[6*self.nStepsTotal:7*self.nStepsTotal]     =  self.mbs.GetSensorStoredData(DS['sPressuresL'])[0:1*self.nStepsTotal,1]
            inputData[7*self.nStepsTotal:8*self.nStepsTotal]     =  self.mbs.GetSensorStoredData(DS['sPressuresL'])[0:1*self.nStepsTotal,2]
            inputData[8*self.nStepsTotal:9*self.nStepsTotal]     =  self.mbs.GetSensorStoredData(DS['sPressuresT'])[0:1*self.nStepsTotal,1]
            inputData[9*self.nStepsTotal:10*self.nStepsTotal]    =  self.mbs.GetSensorStoredData(DS['sPressuresT'])[0:1*self.nStepsTotal,2]
        
        
        if self.Visualization:
           self.mbs.SolutionViewer()
           
        
        
        #++++++++++++++++++++++++++
        #Output data
        sensorTip       = self.dictSensors['sensorTip']
        output          = 0*self.GetOutputScaling()
        output          = self.mbs.GetSensorStoredData(sensorTip)[0:self.nStepsTotal,2]

        inputData       = inputData/self.GetInputScaling()
        output          = self.mbs.GetSensorStoredData(sensorTip)[0:self.nStepsTotal,2]/self.GetOutputScaling().squeeze()
        
        slideSteps      =  np.array([self.n_td])
            
        return [inputData, output, slideSteps] 
    

   #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                                       # --LIFT BOOM---  
                                       
    def FlexibleLiftBoom(self, theta1, p1, p2,
                        ):


        self.theta1 = theta1
        self.p1     = p1
        self.p2     = p2                         
        self.dictSensors = {}
        
        self.StaticInitialization = True
    
        #Ground body
        oGround         = self.mbs.AddObject(ObjectGround())
        markerGround    = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=oGround, localPosition=[0, 0, 0]))

        iCube1          = RigidBodyInertia(mass=93.26, com=pMid1,
                                           inertiaTensor=np.array([[16.358844,-1.27808, 1.7e-5],
                                                                   [-1.27808, 0.612552, -5.9e-5],
                                                                   [1.7e-5,  -5.9e-5  , 16.534255]]),
                                                                       inertiaTensorAtCOM=True)
        
        # graphicsBody1   = GraphicsDataFromSTLfile(fileName1, color4black,verbose=False, invertNormals=True,invertTriangles=True)
        # graphicsBody1   = AddEdgesAndSmoothenNormals(graphicsBody1, edgeAngle=0.25*pi,addEdges=True, smoothNormals=True)
        # graphicsCOM1    = GraphicsDataBasis(origin=iCube1.com, length=2*W1)
        
        # Definintion of pillar as body in Exudyn and node n1
        [n1, b1]        = AddRigidBody(mainSys=self.mbs,
                                     inertia=iCube1,  # includes COM
                                     nodeType=exu.NodeType.RotationEulerParameters,
                                     position=PillarP,
                                     rotationMatrix=np.diag([1, 1, 1]),
                                     gravity=g,
                                     #graphicsDataList=[graphicsCOM1, graphicsBody1]
                                     )
        
        #Pillar
        Marker3         = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b1, localPosition=[0, 0, 0]))                     #With Ground
        Marker4         = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b1, localPosition=[-0.09, 1.4261, 0]))            #Lift Boom
        Marker5         = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b1, localPosition=[0.17, 0.386113249, 0]))        # Cylinder 1 position
        
        # Fixed joint between Pillar and Ground
        self.mbs.AddObject(GenericJoint(markerNumbers=[markerGround, Marker3],constrainedAxes=[1, 1, 1, 1, 1, 1],
        visualization=VObjectJointGeneric(axesRadius=0.2*W1,axesLength=1.4*W1)))
        
        filePath        = '' #To load fine mesh data from Abaqus
        folder_name     = os.path.join(filePath, f"theta1_{self.theta1:.2f}")

        fileNameL       = 'ABAQUS/LiftBoom/OneArm/liftboom-free-050623' #To load fine mesh data from Abaqus
    
        if not self.loadFromSavedNPY: 
            start_time                      = time.time()
            nodes                           = feL.ImportFromAbaqusInputFile(fileNameL+'.inp', typeName='Part', 
                                                                            name='P000524_A_1-Nostopuomi_v2_fem_st')
            feL.ReadMassMatrixFromAbaqus(fileName=fileNameL + '_MASS2.mtx')             #Load mass matrix
            feL.ReadStiffnessMatrixFromAbaqus(fileName=fileNameL + '_STIF2.mtx')        #Load stiffness matrix
            feL.SaveToFile(fileNameL)
            if self.verboseMode:
                print("--- saving LiftBoom FEM Abaqus data took: %s seconds ---" % (time.time() - start_time)) 
                 
        else:       
            if self.verboseMode:
                print('importing Abaqus FEM data structure of Lift Boom...')
            start_time = time.time()
            feL.LoadFromFile(fileNameL)
            cpuTime = time.time() - start_time
            if self.verboseMode:
                print("--- importing FEM data took: %s seconds ---" % (cpuTime))
                    
                     
                         
        # Boundary condition at pillar
        p2                  = [0, 0,-10e-2]
        p1                  = [0, 0, 10e-2]
        radius1             = 1.99e-002
        nodeListJoint1      = feL.GetNodesOnCylinder(p1, p2, radius1, tolerance=1e-2) 
        pJoint1             = feL.GetNodePositionsMean(nodeListJoint1)
        nodeListJoint1Len   = len(nodeListJoint1)
        noodeWeightsJoint1  = [1/nodeListJoint1Len]*nodeListJoint1Len
        noodeWeightsJoint1  =feL.GetNodeWeightsFromSurfaceAreas(nodeListJoint1)
           
        # Boundary condition at Piston 1
        p4                  = [0.3025,-0.1049,-10e-2]
        p3                  = [0.3025,-0.1049, 10e-2]
        radius2             = 3.6e-002
        nodeListPist1       = feL.GetNodesOnCylinder(p3, p4, radius2, tolerance=1e-2)  
        pJoint2             = feL.GetNodePositionsMean(nodeListPist1)
        nodeListPist1Len    = len(nodeListPist1)
        noodeWeightsPist1   = [1/nodeListPist1Len]*nodeListPist1Len
        
        if self.mL != 0:
            p10                 = [2.89,0.0246,-7.4e-2]
            p9                  = [2.89,0.0246, 7.4e-2]
            pdef                = [2.89,0.0246, 0]
            radius5             = 5.2e-002
            nodeListJoint3      = feL.GetNodesOnCylinder(p9, p10, radius5, tolerance=1e-2)  
            pJoint5             = feL.GetNodePositionsMean(nodeListJoint3)
            nodeListJoint3Len   = len(nodeListJoint3)
            noodeWeightsJoint3  = [1/nodeListJoint3Len]*nodeListJoint3Len
            noodeWeightsJoint3  =feL.GetNodeWeightsFromSurfaceAreas(nodeListJoint3)
            
            # STEP 2: Craig-Bampton Modes
            boundaryList        = [nodeListJoint1, nodeListPist1, nodeListJoint3]
            
        else: 
             # STEP 2: Craig-Bampton Modes
             boundaryList        = [nodeListJoint1, nodeListPist1]
    
        
        start_time          = time.time()

        if self.loadFromSavedNPY:
            feL.LoadFromFile('ABAQUS/feL.npy')
        else:
            feL.ComputeHurtyCraigBamptonModes(boundaryNodesList=boundaryList, nEigenModes=self.nModes, 
                                                    useSparseSolver=True,computationMode = HCBstaticModeSelection.RBE2)
            feL.SaveToFile('ABAQUS/feL.npy')
        if self.verboseMode:
            print("Hurty-Craig Bampton modes... ")
            print("eigen freq.=", feL.GetEigenFrequenciesHz())
            print("HCB modes needed %.3f seconds" % (time.time() - start_time))  

        colLift = color4blue
       
        LiftBoom            = ObjectFFRFreducedOrderInterface(feL)
        
        LiftBoomFFRF        = LiftBoom.AddObjectFFRFreducedOrder(self.mbs, positionRef=np.array([-0.09, 1.4261, 0]), 
                                          initialVelocity=[0,0,0], 
                                          initialAngularVelocity=[0,0,0],
                                          rotationMatrixRef  = RotationMatrixZ(mt.radians(self.theta1)),
                                          gravity=g,
                                          #massProportionalDamping = 0, stiffnessProportionalDamping = 1e-5 ,
                                         massProportionalDamping = 0, stiffnessProportionalDamping = self.ReD ,
                                         color=colLift,)
    
        Marker7             = self.mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=LiftBoomFFRF['oFFRFreducedOrder'],
                                          meshNodeNumbers=np.array(nodeListJoint1), #these are the meshNodeNumbers
                                          weightingFactors=noodeWeightsJoint1))
        Marker8             = self.mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=LiftBoomFFRF['oFFRFreducedOrder'],
                                          meshNodeNumbers=np.array(nodeListPist1), #these are the meshNodeNumbers
                                          weightingFactors=noodeWeightsPist1))
        
        
    
        #Revolute Joint
        self.mbs.AddObject(GenericJoint(markerNumbers=[Marker4, Marker7],constrainedAxes=[1,1,1,1,1,0],
                       visualization=VObjectJointGeneric(axesRadius=0.18*0.263342,axesLength=1.1*0.263342)))
    
       
        # Add load 
        if self.mL != 0:
            Marker9             = self.mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=LiftBoomFFRF['oFFRFreducedOrder'],
                                              meshNodeNumbers=np.array(nodeListJoint3), #these are the meshNodeNumbers
                                              weightingFactors=noodeWeightsJoint3))
            
            pos = self.mbs.GetMarkerOutput(Marker9, variableType=exu.OutputVariableType.Position, configuration=exu.ConfigurationType.Reference)
            #print('pos=', pos)
            bMass = self.mbs.CreateMassPoint(physicsMass=self.mL, referencePosition=pos, show=True, gravity=g,
                                     graphicsDataList=[GraphicsDataSphere(radius=0.04, color=color4red)])
            mMass = self.mbs.AddMarker(MarkerBodyPosition(bodyNumber=bMass))
            self.mbs.AddObject(SphericalJoint(markerNumbers=[Marker9, mMass], visualization=VSphericalJoint(show=False)))

            #self.mbs.AddLoad(LoadForceVector(markerNumber=Marker9, bodyFixed=True, loadVector=[0,-self.mL,0]))
       
        colCyl              = color4orange
        colPis              = color4grey 
    
    
        #ODE1 for pressures:
        nODE1            = self.mbs.AddNode(NodeGenericODE1(referenceCoordinates=[0,0],
                                initialCoordinates=[self.p1,
                                                    self.p2], #initialize with 20 bar
                                numberOfODE1Coordinates=2))
    
        #Not used
        def CylinderFriction1(mbs, t, itemNumber, u, v, k, d, F0):

                Ff = 1*StribeckFunction(v, muDynamic=1, muStaticOffset=1.5, regVel=1e-4)+(k*(u) + d*v + k*(u)**3-F0)
                #print(Ff)
                return Ff
            
        def UFfrictionSpringDamper(mbs, t, itemIndex, u, v, k, d, f0):
            return   1*(Fc*tanh(4*(abs(v    )/vs))+(Fs-Fc)*((abs(v    )/vs)/((1/4)*(abs(v    )/vs)**2+3/4)**2))*np.sign(v )+sig2*v    *tanh(4)
          
            
        oFriction1       = self.mbs.AddObject(ObjectConnectorSpringDamper(markerNumbers=[Marker5, Marker8], referenceLength=0.001,stiffness=2000,
                                                            damping=5000, force=80, velocityOffset = 0., activeConnector = True,
                                                            springForceUserFunction=CylinderFriction1,
                                                              visualization=VSpringDamper(show=False) ))

        oHA1 = None
        if True:
            oHA1                = self.mbs.AddObject(HydraulicActuatorSimple(name='LiftCylinder', markerNumbers=[ Marker5, Marker8], 
                                                    nodeNumbers=[nODE1], offsetLength=L_Cyl1, strokeLength=L_Pis1, chamberCrossSection0=A[0], 
                                                    chamberCrossSection1=A[1], hoseVolume0=V1, hoseVolume1=V2, valveOpening0=0, 
                                                    valveOpening1=0, actuatorDamping=4.2e5, oilBulkModulus=Bo, cylinderBulkModulus=Bc, 
                                                    hoseBulkModulus=Bh, nominalFlow=Qn, systemPressure=pS, tankPressure=pT, 
                                                    useChamberVolumeChange=True, activeConnector=True, 
                                                    visualization={'show': True, 'cylinderRadius': 50e-3, 'rodRadius': 28e-3, 
                                                                    'pistonRadius': 0.04, 'pistonLength': 0.001, 'rodMountRadius': 0.0, 
                                                                    'baseMountRadius': 20.0e-3, 'baseMountLength': 20.0e-3, 'colorCylinder': color4orange,
                                                                'colorPiston': color4grey}))
            self.oHA1 = oHA1

        if self.StaticCase or self.StaticInitialization:
            #compute reference length of distance constraint 
            self.mbs.Assemble()
            mGHposition = self.mbs.GetMarkerOutput(Marker5, variableType=exu.OutputVariableType.Position, 
                                             configuration=exu.ConfigurationType.Initial)
            mRHposition = self.mbs.GetMarkerOutput(Marker8, variableType=exu.OutputVariableType.Position, 
                                             configuration=exu.ConfigurationType.Initial)
            
            dLH0 = NormL2(mGHposition - mRHposition)
            if self.verboseMode:
                print('dLH0=', dLH0)
            
            #use distance constraint to compute static equlibrium in static case
            oDC = self.mbs.AddObject(DistanceConstraint(markerNumbers=[Marker5, Marker8], distance=dLH0))

        self.mbs.variables['isStatics'] = False
        from exudyn.signalProcessing import GetInterpolatedSignalValue

        #function which updates hydraulics values input        
        def PreStepUserFunction(mbs, t):
            if not mbs.variables['isStatics']: #during statics, valves must be closed
                Av0 = GetInterpolatedSignalValue(t, mbs.variables['inputTimeU1'], timeArray= [], dataArrayIndex= 1, 
                                            timeArrayIndex= 0, rangeWarning= False)
                # Av0 = 10
                distance = mbs.GetObjectOutput(self.oHA1, exu.OutputVariableType.Distance)

                if distance < 0.75 or distance > 1.2: #limit stroke of actuator
                # if distance < 0.9 or distance > 1:
                    Av0 = 0

                # Av0 = U[mt.trunc(t/h)]
                Av1 = -Av0
            
                if oHA1 != None:
                    mbs.SetObjectParameter(oHA1, "valveOpening0", Av0)
                    mbs.SetObjectParameter(oHA1, "valveOpening1", Av1)

            return True
    
        self.mbs.SetPreStepUserFunction(PreStepUserFunction) 
        if self.verboseMode:
            print('#joint nodes=',len(nodeListJoint3))
        # SensLoc           = self.mbs.AddMarker(MarkerSuperElementRigid(bodyNumber= LiftBoomFFRF['oFFRFreducedOrder'], 
        #                                                     meshNodeNumbers=np.array(nodeListJoint3),
        #                                                     #referencePosition=pdef,
        #                                                      #useAlternativeApproach=altApproach,
        #                                                    weightingFactors=noodeWeightsJoint3))  
        
        nMid        = feL.GetNodeAtPoint(np.array([1.22800004,  0.244657859, -0.0602990314]))
        MarkerTip   = feL.GetNodeAtPoint(np.array([2.829, 0.0151500003,  0.074000001]))
        
        
        if self.verboseMode:
            print("nMid=",nMid)
            print("nMid=",MarkerTip)
         
        # Add Sensor for deflection
        DeflectionF          = self.mbs.AddSensor(SensorSuperElement(bodyNumber=LiftBoomFFRF['oFFRFreducedOrder'], meshNodeNumber=MarkerTip, 
                                                               storeInternal=True, outputVariableType=exu.OutputVariableType.DisplacementLocal ))
        self.dictSensors['sensorTip']=DeflectionF
           
        if oHA1 != None:
            sForce          = self.mbs.AddSensor(SensorObject(objectNumber=oHA1, storeInternal=True, outputVariableType=exu.OutputVariableType.Force))
            self.dictSensors['sForce']=sForce
            sDistance       = self.mbs.AddSensor(SensorObject(objectNumber=oHA1, storeInternal=True, outputVariableType=exu.OutputVariableType.Distance))
            self.dictSensors['sDistance']=sDistance
    
            sVelocity       = self.mbs.AddSensor(SensorObject(objectNumber=oHA1, storeInternal=True, outputVariableType=exu.OutputVariableType.VelocityLocal))
            self.dictSensors['sVelocity']=sVelocity
            sPressures      = self.mbs.AddSensor(SensorNode(nodeNumber=nODE1, storeInternal=True,outputVariableType=exu.OutputVariableType.Coordinates))   
            self.dictSensors['sPressures']=sPressures
            sPressures_t    = self.mbs.AddSensor(SensorNode(nodeNumber=nODE1, storeInternal=True,outputVariableType=exu.OutputVariableType.Coordinates_t))   
            self.dictSensors['sPressures_t']=sPressures_t
            
            def UFsensor(mbs, t, sensorNumbers, factors, configuration):
                val = mbs.GetObjectParameter(self.oHA1, 'valveOpening0')
                return [val] #return angle in degree

            # sInput = self.mbs.AddSensor(SensorUserFunction(sensorNumbers=[], factors=[], 
            #                                                sensorUserFunction=UFsensor, storeInternal=True))
            # self.dictSensors['sInput'] = sInput

        #+++++++++++++++++++++++++++++++++++++++++++++++++++
        #assemble and solve    
        self.mbs.Assemble()
        self.simulationSettings = exu.SimulationSettings()   
        self.simulationSettings.solutionSettings.sensorsWritePeriod = self.endTime / (self.nStepsTotal)
        
        self.simulationSettings.timeIntegration.numberOfSteps            = self.GetNSimulationSteps()
        self.simulationSettings.timeIntegration.endTime                  = self.endTime
        self.simulationSettings.timeIntegration.verboseModeFile          = 0
        self.simulationSettings.timeIntegration.verboseMode              = self.verboseMode
        self.simulationSettings.timeIntegration.newton.useModifiedNewton = True
        self.simulationSettings.linearSolverType                         = exu.LinearSolverType.EigenSparse
        self.simulationSettings.timeIntegration.stepInformation         += 8
        self.simulationSettings.displayStatistics                        = True
        # self.simulationSettings.displayComputationTime                   = True
        self.simulationSettings.linearSolverSettings.ignoreSingularJacobian=True
        self.simulationSettings.timeIntegration.generalizedAlpha.spectralRadius  = 0.7
        self.SC.visualizationSettings.nodes.show = False
        
        if self.Visualization:
            self.SC.visualizationSettings.window.renderWindowSize            = [1600, 1200]        
            self.SC.visualizationSettings.openGL.multiSampling               = 4        
            self.SC.visualizationSettings.openGL.lineWidth                   = 3  
            self.SC.visualizationSettings.general.autoFitScene               = False      
            self.SC.visualizationSettings.nodes.drawNodesAsPoint             = False        
            self.SC.visualizationSettings.nodes.showBasis                    = True 
            #self.SC.visualizationSettings.markers.                    = True
            exu.StartRenderer()
        
        if self.StaticCase or self.StaticInitialization:
            self.mbs.variables['isStatics'] = True
            self.simulationSettings.staticSolver.newton.relativeTolerance = 1e-7
            # self.simulationSettings.staticSolver.stabilizerODE2term = 2
            self.simulationSettings.staticSolver.verboseMode = self.verboseMode
            self.simulationSettings.staticSolver.numberOfLoadSteps = 1
            self.simulationSettings.staticSolver.constrainODE1coordinates = True #constrain pressures to initial values
        
            exu.SuppressWarnings(True)
            self.mbs.SolveStatic(self.simulationSettings, 
                            updateInitialValues=True) #use solution as new initial values for next simulation
            exu.SuppressWarnings(False)

            #now deactivate distance constraint:
            # self.mbs.SetObjectParameter(oDC, 'activeConnector', False)
            force = self.mbs.GetObjectOutput(oDC, variableType=exu.OutputVariableType.Force)
            if self.verboseMode:
                print('initial force=', force)
            
            #deactivate distance constraint
            self.mbs.SetObjectParameter(oDC, 'activeConnector', False)
            
            #overwrite pressures:
            if oHA1 != None:
                dictHA1 = self.mbs.GetObject(oHA1)
                #print('dictHA1=',dictHA1)
                nodeHA1 = dictHA1['nodeNumbers'][0]
                A_HA1 = dictHA1['chamberCrossSection0']
                pDiff = force/A_HA1
            
                # #now we would like to reset the pressures:
                # #2) change the initial values in the system vector
            
                sysODE1 = self.mbs.systemData.GetODE1Coordinates(configuration=exu.ConfigurationType.Initial)
                nODE1index = self.mbs.GetNodeODE1Index(nodeHA1) #coordinate index for node nodaHA1
                if self.verboseMode:
                    # print('sysODE1=',sysODE1)
                    print('p0,p1=',sysODE1[nODE1index],sysODE1[nODE1index+1])
                sysODE1[nODE1index] += pDiff #add required difference to pressure
    
                #now write the updated system variables:
                self.mbs.systemData.SetODE1Coordinates(coordinates=sysODE1, configuration=exu.ConfigurationType.Initial)
    
                if self.verboseMode:
                    print('new p0,p1=',sysODE1[nODE1index],sysODE1[nODE1index+1])

            self.mbs.variables['isStatics'] = False


        if not self.StaticCase:
            exu.SolveDynamic(self.mbs, simulationSettings=self.simulationSettings,
                                solverType=exu.DynamicSolverType.TrapezoidalIndex2)

        if self.Visualization:
            # self.SC.WaitForRenderEngineStopFlag()
            exu.StopRenderer()
            
   #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                                       # --PATU CRANE---   
    #main function to create hydraulics arm
    def PatuCrane(self, theta1, theta2, p1, p2, p3, p4 
                 ):

        self.theta1 = theta1
        self.theta2 = theta2
        self.p1     = p1
        self.p2     = p2                         #Pressure_RigidModel(theta1, p1, p2)
        self.p3     = p3
        self.p4     = p4
        
        self.StaticInitialization = True
        
        self.dictSensors = {}
        
        oGround         = self.mbs.AddObject(ObjectGround())
        markerGround    = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=oGround, localPosition=[0, 0, 0]))

        iCube1          = RigidBodyInertia(mass=93.26, com=pMid1,
                                           inertiaTensor=np.array([[16.358844,-1.27808, 1.7e-5],
                                                                   [-1.27808, 0.612552, -5.9e-5],
                                                                   [1.7e-5,  -5.9e-5  , 16.534255]]),
                                                                       inertiaTensorAtCOM=True)
        
        # graphicsBody1   = GraphicsDataFromSTLfile(fileName1, color4black,verbose=False, invertNormals=True,invertTriangles=True)
        # graphicsBody1   = AddEdgesAndSmoothenNormals(graphicsBody1, edgeAngle=0.25*pi,addEdges=True, smoothNormals=True)
        # graphicsCOM1    = GraphicsDataBasis(origin=iCube1.com, length=2*W1)
        
        # Definintion of pillar as body in Exudyn and node n1
        [n1, b1]        = AddRigidBody(mainSys=self.mbs,
                                     inertia=iCube1,  # includes COM
                                     nodeType=exu.NodeType.RotationEulerParameters,
                                     position=PillarP,
                                     rotationMatrix=np.diag([1, 1, 1]),
                                     gravity=g,
                                     graphicsDataList=[graphicsCOM1, graphicsBody1]
                                     )
        
        #Pillar
        Marker3         = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b1, localPosition=[0, 0, 0]))                     #With Ground
        Marker4         = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b1, localPosition=[-0.09, 1.4261, 0]))            #Lift Boom
        Marker5         = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b1, localPosition=[0.17, 0.386113249, 0]))        # Cylinder 1 position
        
        # Fixed joint between Pillar and Ground
        self.mbs.AddObject(GenericJoint(markerNumbers=[markerGround, Marker3],constrainedAxes=[1, 1, 1, 1, 1, 1],
        visualization=VObjectJointGeneric(axesRadius=0.2*W1,axesLength=1.4*W1)))
        
        
        if self.Flexible:
            filePath        = '' #To load fine mesh data from Abaqus
            folder_name     = os.path.join(filePath, f"theta1_{self.theta1:.2f}")
        
            fileNameL       = 'ABAQUS/LiftBoom/liftboom-free-050623'        #To load fine mesh data from ABAQUS
            fileNameT       = 'ABAQUS/TiltBoom/tiltboom-120623'             #To load fine mesh data from ABAQUS
                    
            if not self.loadFromSavedNPY: 
                start_time                      = time.time()
                nodesL                          = feL.ImportFromAbaqusInputFile(fileNameL+'.inp', typeName='Part', 
                                                                                name='P000524_A_1-Nostopuomi_v2_fem_st')
                feL.ReadMassMatrixFromAbaqus(fileName=fileNameL + '_MASS2.mtx')             #Load mass matrix
                feL.ReadStiffnessMatrixFromAbaqus(fileName=fileNameL + '_STIF2.mtx')        #Load stiffness matrix
                feL.SaveToFile(fileNameL)
                            
                nodesT                          = feT.ImportFromAbaqusInputFile(fileNameT+'.inp', typeName='Part', name='P000516_A_1-Taittopuomi_fem')
                feT.ReadMassMatrixFromAbaqus(fileName=fileNameT + '_MASS1.mtx')             #Load mass matrix
                feT.ReadStiffnessMatrixFromAbaqus(fileName=fileNameT + '_STIF1.mtx')        #Load stiffness matrix
                feT.SaveToFile(fileNameT)
                print("--- saving LiftBoom and TiltBoom FEM Abaqus data took: %s seconds ---" % (time.time() - start_time)) 
                       
            else:       
                print('importing Abaqus FEM data structure of Lift Boom...')
                start_time = time.time()
                feL.LoadFromFile(fileNameL)
                feT.LoadFromFile(fileNameT)
                cpuTime = time.time() - start_time
                print("--- importing FEM data took: %s seconds ---" % (cpuTime))
                            
            # Boundary condition at pillar
            p2                  = [0, 0,-10e-2]
            p1                  = [0, 0, 10e-2]
            radius1             = 2.5e-002
            nodeListJoint1      = feL.GetNodesOnCylinder(p1, p2, radius1, tolerance=1e-2) 
            pJoint1             = feL.GetNodePositionsMean(nodeListJoint1)
            nodeListJoint1Len   = len(nodeListJoint1)
            noodeWeightsJoint1  = [1/nodeListJoint1Len]*nodeListJoint1Len 
                
            # Boundary condition at Piston 1
            p4                  = [0.3027,-0.1049+2e-3,-10e-2]
            p3                  = [0.3027,-0.1049+2e-3, 10e-2]
            radius2             = 3.6e-002
            nodeListPist1       = feL.GetNodesOnCylinder(p3, p4, radius2, tolerance=1e-2)  
            pJoint2             = feL.GetNodePositionsMean(nodeListPist1)
            nodeListPist1Len    = len(nodeListPist1)
            noodeWeightsPist1   = [1/nodeListPist1Len]*nodeListPist1Len
                   
                   
            # Boundary condition at cylinder 1
            p6                  = [1.265-6e-3,0.2080-14e-3,-8.02e-2]
            p5                  = [1.265-6e-3,0.2080-14e-3, 8.02e-2]
            radius3             = 3.2e-002
            nodeListCyl2        = feL.GetNodesOnCylinder(p5, p6, radius3, tolerance=1e-2)  
            pJoint3             = feL.GetNodePositionsMean(nodeListCyl2)
            nodeListCyl2Len     = len(nodeListCyl2)
            noodeWeightsCyl2    = [1/nodeListCyl2Len]*nodeListCyl2Len 
                        
            # Boundary condition at Joint 2
            p8                  = [2.69+1.25e-3,2.38e-02-16.5e-3,-7.4e-2]
            p7                  = [2.69+1.25e-3,2.38e-02-16.5e-3, 7.4e-2]
            radius4             = 3.7e-002
            nodeListJoint2      = feL.GetNodesOnCylinder(p7, p8, radius4, tolerance=1e-2)  
            pJoint4             = feL.GetNodePositionsMean(nodeListJoint2)
            nodeListJoint2Len   = len(nodeListJoint2)
            noodeWeightsJoint2  = [1/nodeListJoint2Len]*nodeListJoint2Len
                    
            # Joint 3
            p10                 = [2.89,0.0246,-7.4e-2]
            p9                  = [2.89,0.0246, 7.4e-2]
            radius5             = 5.2e-002
            nodeListJoint3      = feL.GetNodesOnCylinder(p9, p10, radius5, tolerance=1e-2)  
            pJoint5             = feL.GetNodePositionsMean(nodeListJoint3)
            nodeListJoint3Len   = len(nodeListJoint3)
            noodeWeightsJoint3  = [1/nodeListJoint3Len]*nodeListJoint3Len
                    
            # Boundary condition at pillar
            p12                 = [9.92e-15, 2.70e-3,-9.63e-2]
            p11                 = [9.92e-15, 2.70e-3, 9.63e-2]
            radius6             = 4.82e-002
            nodeListJoint1T     = feT.GetNodesOnCylinder(p11, p12, radius6, tolerance=1e-2) 
            pJoint1T            = feT.GetNodePositionsMean(nodeListJoint1T)
            nodeListJoint1TLen  = len(nodeListJoint1T)
            noodeWeightsJoint1T = [1/nodeListJoint1TLen]*nodeListJoint1TLen
                        
            # Boundary condition at Piston 1
            p14                 = [-9.5e-2,0.24,-7.15e-2]
            p13                 = [-9.5e-2,0.24, 7.15e-2]
            radius7             = 2.5e-002
            nodeListPist1T      = feT.GetNodesOnCylinder(p13, p14, radius7, tolerance=1e-2)  
            pJoint2T            = feT.GetNodePositionsMean(nodeListPist1T)
            nodeListPist1TLen   = len(nodeListPist1T)
            noodeWeightsPist1T  = [1/nodeListPist1TLen]*nodeListPist1TLen
                                
            
            # Boundary condition at extension boom
            p16                 = [-0.415,0.295,-0.051783]
            p15                 = [-0.415,0.295, 0.051783]
            radius8             = 2.5e-002
            nodeListExtT        = feT.GetNodesOnCylinder(p15, p16, radius8, tolerance=1e-2)  
            pExtT               = feT.GetNodePositionsMean(nodeListExtT)
            nodeListExTLen      = len(nodeListExtT)
            noodeWeightsExt1T   = [1/nodeListExTLen]*nodeListExTLen
                    
            boundaryListL   = [nodeListJoint1,nodeListPist1, nodeListJoint2,  nodeListJoint3] 
            boundaryListT  = [nodeListJoint1T, nodeListPist1T,nodeListExtT]
            
            
                
            if not self.loadFromSavedNPY:
                start_time      = time.time()
                feL.ComputeHurtyCraigBamptonModes(boundaryNodesList=boundaryListL, nEigenModes=self.nModes, 
                                                                useSparseSolver=True,computationMode = HCBstaticModeSelection.RBE2)
                feT.ComputeHurtyCraigBamptonModes(boundaryNodesList=boundaryListT, nEigenModes=self.nModes, 
                                                                useSparseSolver=True,computationMode = HCBstaticModeSelection.RBE2) 
                feL.SaveToFile('ABAQUS/LiftBoom/feL.npy')
                feT.SaveToFile('ABAQUS/TiltBoom/feT.npy')

            else:
                feL.LoadFromFile('ABAQUS/LiftBoom/feL.npy')
                feT.LoadFromFile('ABAQUS/TiltBoom/feT.npy')

            
            LiftBoom            = ObjectFFRFreducedOrderInterface(feL)
            TiltBoom            = ObjectFFRFreducedOrderInterface(feT)
                    
            LiftBoomFFRF        = LiftBoom.AddObjectFFRFreducedOrder(self.mbs, positionRef=np.array([-0.09, 1.4261, 0]), 
                                                          initialVelocity=[0,0,0], 
                                                          initialAngularVelocity=[0,0,0],
                                                          rotationMatrixRef  = RotationMatrixZ(mt.radians(self.theta1)),
                                                          gravity=g,
                                                          color=colLift,)
            
            # Bundary conditions: Markers on Lift boom
            Marker7         = self.mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=LiftBoomFFRF['oFFRFreducedOrder'],
                                                          meshNodeNumbers=np.array(nodeListJoint1), #these are the meshNodeNumbers
                                                           weightingFactors=noodeWeightsJoint1))
            Marker8             = self.mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=LiftBoomFFRF['oFFRFreducedOrder'],
                                                           meshNodeNumbers=np.array(nodeListPist1), #these are the meshNodeNumbers
                                                           weightingFactors=noodeWeightsPist1))           #With Cylinder 1
                    
            Marker9             = self.mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=LiftBoomFFRF['oFFRFreducedOrder'],
                                                           meshNodeNumbers=np.array(nodeListCyl2), #these are the meshNodeNumbers
                                                           weightingFactors=noodeWeightsCyl2)) 
            Marker10        = self.mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=LiftBoomFFRF['oFFRFreducedOrder'],
                                                           meshNodeNumbers=np.array(nodeListJoint2), #these are the meshNodeNumbers
                                                           weightingFactors=noodeWeightsJoint2))      
                    
            Marker11        = self.mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=LiftBoomFFRF['oFFRFreducedOrder'],
                                                         meshNodeNumbers=np.array(nodeListJoint3), #these are the meshNodeNumbers
                                                         weightingFactors=noodeWeightsJoint3))
            
            if self.StaticCase or self.StaticInitialization:
                #compute reference length of distance constraint 
                self.mbs.Assemble()
                
                TiltP = self.mbs.GetMarkerOutput(Marker11, variableType=exu.OutputVariableType.Position, 
                                                 configuration=exu.ConfigurationType.Initial)
            
            
                    
            TiltBoomFFRF        = TiltBoom.AddObjectFFRFreducedOrder(self.mbs, positionRef=TiltP, #2.879420180699481+27e-3, -0.040690041435711005+8.3e-2, 0
                                                          initialVelocity=[0,0,0], 
                                                          initialAngularVelocity=[0,0,0],
                                                          rotationMatrixRef  = RotationMatrixZ(mt.radians(self.theta2))   ,
                                                          gravity=g,
                                                          color=colLift,)

                    
            Marker13        = self.mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=TiltBoomFFRF['oFFRFreducedOrder'], 
                                                                                  meshNodeNumbers=np.array(nodeListJoint1T), #these are the meshNodeNumbers
                                                                                  weightingFactors=noodeWeightsJoint1T))
                    
            Marker14        = self.mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=TiltBoomFFRF['oFFRFreducedOrder'], 
                                                                                  meshNodeNumbers=np.array(nodeListPist1T), #these are the meshNodeNumbers
                                                                                  weightingFactors=noodeWeightsPist1T))  
            
            
            MarkerEx        = self.mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=TiltBoomFFRF['oFFRFreducedOrder'], 
                                                                                  meshNodeNumbers=np.array(nodeListExtT), #these are the meshNodeNumbers
                                                                                  weightingFactors=noodeWeightsExt1T))  
            
            MarkerTip   = feT.GetNodeAtPoint(np.array([1.80499995,  0.266000003, 0.0510110967]))
       
        else:
            
            #Rigid pillar
            pMid2           = np.array([1.229248, 0.055596, 0])  
            iCube2          = RigidBodyInertia(mass=143.66, com=pMid2,
                                            inertiaTensor=np.array([[1.295612, 2.776103,  -0.000003],
                                                                    [ 2.776103,  110.443667, 0],
                                                                    [ -0.000003,              0  ,  110.452812]]),
                                            inertiaTensorAtCOM=True)
            
            graphicsBody2   = GraphicsDataFromSTLfile(fileName2, color4blue,
                                            verbose=False, invertNormals=True,
                                            invertTriangles=True)
            graphicsBody2   = AddEdgesAndSmoothenNormals(graphicsBody2, edgeAngle=0.25*pi,
                                                addEdges=True, smoothNormals=True)
            graphicsCOM2    = GraphicsDataBasis(origin=iCube2.com, length=2*W2)
            [n2, b2]        = AddRigidBody(mainSys=self.mbs,
                                inertia=iCube2,  # includes COM
                                nodeType=exu.NodeType.RotationEulerParameters,
                                position=LiftP,  # pMid2
                                rotationMatrix=RotationMatrixZ(mt.radians(self.theta1)),
                                gravity=g,
                                graphicsDataList=[graphicsCOM2, graphicsBody2])
            
            Marker7         = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b2, localPosition=[0, 0, 0]))                       #With Pillar    
            Marker8         = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b2, localPosition=[0.3025, -0.105, 0]))             #With Cylinder 1
            Marker9         = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b2, localPosition=[1.263, 0.206702194, 0]))         #With Cylinder 2
            Marker10        = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b2, localPosition=[2.69107, -6e-3, 0]))         #With Bracket 1  
            Marker11        = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b2, localPosition=[2.886080943, 0.019892554, 0]))   #With Tilt Boom
            
            if self.StaticCase or self.StaticInitialization:
                #compute reference length of distance constraint 
                self.mbs.Assemble()
                
                TiltP = self.mbs.GetMarkerOutput(Marker11, variableType=exu.OutputVariableType.Position, 
                                                 configuration=exu.ConfigurationType.Initial)
                
            # Third Body: TiltBoom+ExtensionBoom   
            pMid3           = np.array([ 0.659935,  0.251085, 0])  # center of mass
            iCube3          = RigidBodyInertia(mass=83.311608+ 9.799680, com=pMid3,
                                                  inertiaTensor=np.array([[0.884199, 1.283688, -0.000001],
                                                                        [1.283688,  35.646721,    0.000000],
                                                                        [ -0.000001, 0,        36.035964]]),
                                                                          inertiaTensorAtCOM=True)
            graphicsBody3   = GraphicsDataFromSTLfile(fileName3, color4blue,verbose=False, invertNormals=True,invertTriangles=True)
            graphicsBody3   = AddEdgesAndSmoothenNormals(graphicsBody3, edgeAngle=0.25*pi,addEdges=True, smoothNormals=True)
            graphicsCOM3    = GraphicsDataBasis(origin=iCube3.com, length=2*W3)
            [n3, b3]        = AddRigidBody(mainSys=self.mbs,
                                    inertia=iCube3,  # includes COM
                                    nodeType=exu.NodeType.RotationEulerParameters,
                                    position=TiltP,  # pMid2
                                    rotationMatrix=RotationMatrixZ(mt.radians(self.theta2)),
                                    gravity=g,
                                    graphicsDataList=[graphicsCOM3, graphicsBody3])
            Marker13        = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b3, localPosition=[0, 0, 0]))                        #With LIft Boom 
            Marker14        = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b3, localPosition=[-0.095, 0.24043237, 0])) 
            MarkerEx        = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b3, localPosition=[-0.415,0.295, 0]))
       
        # ##########################################
                # --- UPDATE POSITIONS ---
        ###########################################
        if self.StaticCase or self.StaticInitialization:
            #compute reference length of distance constraint 
            self.mbs.Assemble()
            Bracket1L = self.mbs.GetMarkerOutput(Marker10, variableType=exu.OutputVariableType.Position, 
                                             configuration=exu.ConfigurationType.Initial)
            
            Bracket1B = self.mbs.GetMarkerOutput(Marker14, variableType=exu.OutputVariableType.Position, 
                                             configuration=exu.ConfigurationType.Initial)
            
            ExtensionP = self.mbs.GetMarkerOutput(MarkerEx, variableType=exu.OutputVariableType.Position, 
                                             configuration=exu.ConfigurationType.Initial)
            
            #exu.StartRenderer()
            
            def AnalyticalEq(p):
                theta3_degrees, theta4_degrees = p
                
                # if theta3_degrees < 125:
                #     theta3_degrees = 125
                # elif theta3_degrees > 150:
                #     theta3_degrees = 150

                # if theta4_degrees < 120:
                #     theta4_degrees = 120
                # elif theta4_degrees > 225:
                #     theta4_degrees = 225
                
                theta3 = np.radians(theta3_degrees)
                theta4 = np.radians(theta4_degrees) 
                                
                eq1 =    l3 * cos(theta3) + l4 * cos(np.pi-theta4) + l2x +d1
                eq2 =    l3 * sin(theta3) + l4 * sin(np.pi-theta4) + l2y +h1
                
                return [eq1, eq2]
            
            l3         = NormL2(np.array([ 0,  0, 0]) - np.array([0.456, -0.0405, 0]))  
            l4         = NormL2(np.array([ 0,  0, 0]) - np.array([0.48, 0, 0]))  
            d1         = Bracket1L[0]-TiltP[0]
            h1         = Bracket1L[1]-TiltP[1]
            l2x        = -(Bracket1B[0]-TiltP[0])
            l2y        = -(Bracket1B[1]-TiltP[1])
            
            # alpha0     = degrees(atan2((Bracket1B[1] - TiltP[1]), (Bracket1B[0] - TiltP[0])))
            
            initialangles  = [130.562128044041, 180.26679642675617]
            solutions = fsolve(AnalyticalEq, initialangles)
            self.theta3, self.theta4 = solutions
            
            #print(self.theta3,self.theta4)


        # # # # # # 4th Body: Bracket 1
        #Bracket1L       = LiftP + np.array([2.579002-6e-3, 0.7641933967880499+5e-3, 0])
        pMid4           = np.array([0.257068, 0.004000 , 0])  # center of mass, body0,0.004000,-0.257068
        iCube4          = RigidBodyInertia(mass=11.524039, com=pMid4,
                                                    inertiaTensor=np.array([[0.333066, 0.017355, 0],
                                                                            [0.017355, 0.081849, 0],
                                                                            [0,              0, 0.268644]]),
                                                                              inertiaTensorAtCOM=True)

        graphicsBody4   = GraphicsDataFromSTLfile(fileName4, color4blue,verbose=False, invertNormals=True,invertTriangles=True)
        graphicsBody4   = AddEdgesAndSmoothenNormals(graphicsBody4, edgeAngle=0.25*pi,addEdges=True, smoothNormals=True)
        graphicsCOM4    = GraphicsDataBasis(origin=iCube4.com, length=2*W4)
        [n4, b4]        = AddRigidBody(mainSys=self.mbs,inertia=iCube4,  # includes COM
                                                  nodeType=exu.NodeType.RotationEulerParameters,
                                                  position=Bracket1L,  # pMid2
                                                  rotationMatrix=RotationMatrixZ(mt.radians(self.theta3)), #-0.414835768117858
                                                  gravity=g,graphicsDataList=[graphicsCOM4, graphicsBody4])
                
        # # # # 5th Body: Bracket 2
        pMid5           = np.array([0.212792, 0, 0])  # center of mass, body0
        iCube5          = RigidBodyInertia(mass=7.900191, com=pMid5,
                                                      inertiaTensor=np.array([[0.052095, 0, 0],
                                                                              [0,  0.260808, 0],
                                                                              [0,              0,  0.216772]]),
                                                                              inertiaTensorAtCOM=True)
                
        graphicsBody5   = GraphicsDataFromSTLfile(fileName5, color4blue,verbose=False, invertNormals=True,invertTriangles=True)
        graphicsBody5   = AddEdgesAndSmoothenNormals(graphicsBody5, edgeAngle=0.25*pi,addEdges=True, smoothNormals=True)
        graphicsCOM5    = GraphicsDataBasis(origin=iCube5.com, length=2*W5)
        [n5, b5]        = AddRigidBody(mainSys=self.mbs,inertia=iCube5,  # includes COM
                                                  nodeType=exu.NodeType.RotationEulerParameters,
                                                  position=Bracket1B,  # pMid2
                                                  rotationMatrix=RotationMatrixZ(mt.radians(self.theta4)) ,   #-5, 140
                                                  gravity=g,graphicsDataList=[graphicsCOM5, graphicsBody5])
                
            
                
        Marker15        = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b4, localPosition=[0, 0, 0]))                        #With LIft Boom 
        Marker16        = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b4, localPosition=[0.456, -0.0405 , 0]))  
        Marker18        = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b5, localPosition=[0, 0, 0]))                        #With LIft Boom 
        Marker19        = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b5, localPosition=[0.475+5e-3, 0, 0]))                   #With LIft Boom,-0.475 


        # # # # 5th Body: Bracket 2
        pMid6           = np.array([1.15, 0.06, 0])  # center of mass, body0
        iCube6          = RigidBodyInertia(mass=58.63, com=pMid6,
                                               inertiaTensor=np.array([[0.13, 0, 0],
                                                                       [0.10,  28.66, 0],
                                                                       [0,              0,  28.70]]),
                                                                       inertiaTensorAtCOM=True)
         
        graphicsBody6   = GraphicsDataFromSTLfile(fileName6, color4blue,verbose=False, invertNormals=True,invertTriangles=True)
        graphicsBody6   = AddEdgesAndSmoothenNormals(graphicsBody6, edgeAngle=0.25*pi,addEdges=True, smoothNormals=True)
        graphicsCOM6    = GraphicsDataBasis(origin=iCube6.com, length=2*W5)
        [n6, b6]        = AddRigidBody(mainSys=self.mbs,inertia=iCube6,  # includes COM
                                           nodeType=exu.NodeType.RotationEulerParameters,
                                           position=ExtensionP+np.array([0, -0.10, 0]),  # pMid2
                                           rotationMatrix=RotationMatrixZ(mt.radians(self.theta2)) ,   #-5, 140
                                           gravity=g,graphicsDataList=[graphicsCOM6, graphicsBody6]) 
        
        Marker20        = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b6, localPosition=[0, 0.1, 0]))
        
        if self.mL != 0:
            TipLoadMarker     = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b6, localPosition=[2.45, 0.05, 0]))
            pos = self.mbs.GetMarkerOutput(TipLoadMarker, variableType=exu.OutputVariableType.Position, 
                                           configuration=exu.ConfigurationType.Reference)
            #print('pos=', pos)
            bMass = self.mbs.CreateMassPoint(physicsMass=self.mL, referencePosition=pos,gravity=g, show=True,
                                     graphicsDataList=[GraphicsDataSphere(radius=0.1, color=color4red)])
            mMass = self.mbs.AddMarker(MarkerBodyPosition(bodyNumber=bMass))
            self.mbs.AddObject(SphericalJoint(markerNumbers=[TipLoadMarker, mMass], visualization=VSphericalJoint(show=False)))

        # ##########################################
                # --- UPDATE ANGLES ---
        ###########################################
        # if self.StaticCase or self.StaticInitialization:
        #     #compute reference length of distance constraint 
        #     self.mbs.Assemble()
        #     #exu.StartRenderer()
            
            #currentState = self.mbs.systemData.GetSystemState()
            #self.mbs.WaitForUserToContinue()
                
        #++++++++++++ Add joints
        # #Add Revolute Joint btw Pillar and LiftBoom
        self.mbs.AddObject(GenericJoint(markerNumbers=[Marker4, Marker7],constrainedAxes=[1,1,1,1,1,0],
                                  visualization=VObjectJointGeneric(axesRadius=0.18*W2,axesLength=0.18)))   
                
        # # #Add Revolute Joint btw LiftBoom and TiltBoom
        self.mbs.AddObject(GenericJoint(markerNumbers=[Marker11, Marker13],constrainedAxes=[1,1,1,1,1,0],
                                    visualization=VObjectJointGeneric(axesRadius=0.22*W3,axesLength=0.16)))     
        
         
        # # # # #Add Revolute Joint btw LiftBoom and Bracket 1
        self.mbs.AddObject(GenericJoint(markerNumbers=[Marker10, Marker15],constrainedAxes=[1,1,1,1,1,0],
                                  visualization=VObjectJointGeneric(axesRadius=0.32*W4,axesLength=0.20)))   
                
        # # # #Add Revolute Joint btw Bracket 1 and Bracket 2
        self.mbs.AddObject(GenericJoint(markerNumbers=[Marker16, Marker19],constrainedAxes=[1,1,1,1,1,0],
                                    visualization=VObjectJointGeneric(axesRadius=0.28*W5,axesLength=2.0*W5)))  
                
        # # # # Revolute joint between Bracket 2 and TiltBoom
        self.mbs.AddObject(GenericJoint(markerNumbers=[Marker18, Marker14],constrainedAxes=[1,1,0,0,0,0],
                                        visualization=VObjectJointGeneric(axesRadius=0.23*W5,axesLength=0.20)))  
        
        # Fixed joint between Tilt boom and extension boom
        self.mbs.AddObject(GenericJoint(markerNumbers=[MarkerEx, Marker20],constrainedAxes=[1, 1, 1, 1, 1, 1],
        visualization=VObjectJointGeneric(axesRadius=0.2*W1,axesLength=1.4*W1)))
        

        #add hydraulics actuator:
        colCyl              = color4orange
        colPis              = color4grey 
        LH1                 = L_Cyl1                        #zero length of actuator
        LH2                 = L_Cyl2                        #zero length of actuator
                    
        nODE1               = self.mbs.AddNode(NodeGenericODE1(referenceCoordinates=[0, 0],
                                                initialCoordinates=[self.p1,
                                                                    self.p2],  # initialize with 20 bar
                                                                    numberOfODE1Coordinates=2))

        nODE2               = self.mbs.AddNode(NodeGenericODE1(referenceCoordinates=[0, 0],
                                                initialCoordinates=[self.p3,
                                                                    self.p4],  # initialize with 20 bar
                                                                    numberOfODE1Coordinates=2))
           
        def CylinderFriction1(mbs, t, itemNumber, u, v, k, d, F0):

                Ff = 1*StribeckFunction(v, muDynamic=1, muStaticOffset=1.5, regVel=1e-4)+(k*(u) + d*v + k*(u)**3-F0)
                #print(Ff)
                return Ff
            
        def CylinderFriction2(mbs, t, itemNumber, u, v, k, d, F0):

                Ff =  1*StribeckFunction(v, muDynamic=0.5, muStaticOffset=0.5, regVel=1e-2) - (k*(u) - d*v + k*(u)**3 -F0)
                #print(Ff)
                return Ff
            
        oFriction1       = self.mbs.AddObject(ObjectConnectorSpringDamper(markerNumbers=[Marker5, Marker8], referenceLength=0,stiffness=2000,
                                                            damping=5000, force=80, velocityOffset = 0., activeConnector = True,
                                                            springForceUserFunction=CylinderFriction1,
                                                              visualization=VSpringDamper(show=False) ))
                        
        oFriction2       = self.mbs.AddObject(ObjectConnectorSpringDamper(markerNumbers=[Marker9, Marker16], referenceLength=0,stiffness=1250,
                                                              damping=5000, force=50, velocityOffset = 0, activeConnector = True,
                                                              springForceUserFunction=CylinderFriction2,
                                                                visualization=VSpringDamper(show=False) ))
        oHA1 = None
        oHA2 = None

        if True: 
                oHA1                = self.mbs.AddObject(HydraulicActuatorSimple(name='LiftCylinder', markerNumbers=[ Marker5, Marker8], 
                                                        nodeNumbers=[nODE1], offsetLength=LH1, strokeLength=L_Pis1, chamberCrossSection0=A[0], 
                                                        chamberCrossSection1=A[1], hoseVolume0=V1, hoseVolume1=V2, valveOpening0=0, 
                                                        valveOpening1=0, actuatorDamping=4.2e5, oilBulkModulus=Bo, cylinderBulkModulus=Bc, 
                                                        hoseBulkModulus=Bh, nominalFlow=Qn1, systemPressure=pS, tankPressure=pT, 
                                                        useChamberVolumeChange=True, activeConnector=True, 
                                                        visualization={'show': True, 'cylinderRadius': 50e-3, 'rodRadius': 28e-3, 
                                                                        'pistonRadius': 0.04, 'pistonLength': 0.001, 'rodMountRadius': 0.0, 
                                                                        'baseMountRadius': 20.0e-3, 'baseMountLength': 20.0e-3, 'colorCylinder': color4orange,
                                                                    'colorPiston': color4grey}))
                    
                oHA2 = self.mbs.AddObject(HydraulicActuatorSimple(name='TiltCylinder', markerNumbers=[Marker9, Marker16], 
                                                          nodeNumbers=[nODE2], offsetLength=LH2, strokeLength=L_Pis2, chamberCrossSection0=A[0], 
                                                          chamberCrossSection1=A[1], hoseVolume0=V1, hoseVolume1=V2, valveOpening0=0, 
                                                          valveOpening1=0, actuatorDamping=2.5e5, oilBulkModulus=Bo, cylinderBulkModulus=Bc, 
                                                          hoseBulkModulus=Bh, nominalFlow=Qn2, systemPressure=pS, tankPressure=pT, 
                                                          useChamberVolumeChange=True, activeConnector=True, 
                                                          visualization={'show': True, 'cylinderRadius': 50e-3, 'rodRadius': 28e-3, 
                                                                          'pistonRadius': 0.04, 'pistonLength': 0.001, 'rodMountRadius': 0.0, 
                                                                          'baseMountRadius': 0.0, 'baseMountLength': 0.0, 'colorCylinder': color4orange,
                                                                          'colorPiston': color4grey}))
            
                self.oHA1 = oHA1
                self.oHA2 = oHA2
            
        if self.StaticCase or self.StaticInitialization:
            #compute reference length of distance constraint 
            self.mbs.Assemble()
            mGHposition = self.mbs.GetMarkerOutput(Marker5, variableType=exu.OutputVariableType.Position, 
                                             configuration=exu.ConfigurationType.Initial)
            mRHposition = self.mbs.GetMarkerOutput(Marker8, variableType=exu.OutputVariableType.Position, 
                                             configuration=exu.ConfigurationType.Initial)
            
            mGHpositionT = self.mbs.GetMarkerOutput(Marker9, variableType=exu.OutputVariableType.Position, 
                                             configuration=exu.ConfigurationType.Initial)
            mRHpositionT = self.mbs.GetMarkerOutput(Marker16, variableType=exu.OutputVariableType.Position, 
                                             configuration=exu.ConfigurationType.Initial)
            
            
            dLH0 = NormL2(mGHposition - mRHposition)
            dLH1 = NormL2(mGHpositionT - mRHpositionT)
            
            if self.verboseMode:
                print('dLH0=', dLH0)
            
            oDC = self.mbs.AddObject(DistanceConstraint(markerNumbers=[Marker5, Marker8], distance=dLH0))
            oDCT = self.mbs.AddObject(DistanceConstraint(markerNumbers=[Marker9, Marker16], distance=dLH1))

        self.mbs.variables['isStatics'] = False
        from exudyn.signalProcessing import GetInterpolatedSignalValue
            
        def PreStepUserFunction(mbs, t):
                if not mbs.variables['isStatics']: 
                    Av0 = GetInterpolatedSignalValue(t, mbs.variables['inputTimeU1'], timeArray= [], dataArrayIndex= 1, 
                                            timeArrayIndex= 0, rangeWarning= False)
                    Av2 = GetInterpolatedSignalValue(t, mbs.variables['inputTimeU2'], timeArray= [], dataArrayIndex= 1, 
                                                   timeArrayIndex= 0, rangeWarning= False)
                

                    Av1 = -Av0
                    Av3 = -Av2
           
                    if oHA1 != None:
                       mbs.SetObjectParameter(oHA1, "valveOpening0", Av0)
                       mbs.SetObjectParameter(oHA1, "valveOpening1", Av1)
                       mbs.SetObjectParameter(oHA2, "valveOpening0", Av2)
                       mbs.SetObjectParameter(oHA2, "valveOpening1", Av3)
                        
                return True

        self.mbs.SetPreStepUserFunction(PreStepUserFunction)  
        
        if self.Flexible:
            #Angle1       = self.mbs.AddSensor(SensorNode( nodeNumber=LiftBoomFFRF['nRigidBody'], storeInternal=True, 
             #                                           fileName = f"solution/Simulation_Flexible_f_modes_{self.nModes}_angle1.txt", outputVariableType=exu.OutputVariableType.Rotation))
            #Angle2       = self.mbs.AddSensor(SensorNode(nodeNumber= TiltBoomFFRF['nRigidBody'], storeInternal=True,
              #                                           fileName = f"solution/Simulation_Flexible_f_modes_{self.nModes}_angle2.txt", outputVariableType=exu.OutputVariableType.Rotation))
            #AngVelocity1       = self.mbs.AddSensor(SensorNode( nodeNumber=LiftBoomFFRF['nRigidBody'], storeInternal=True,
               #                                                fileName = f"solution/Simulation_Flexible_f_modes_{self.nModes}_angularVelocity1.txt", 
                #                                               outputVariableType=exu.OutputVariableType.AngularVelocity))
            #AngVelocity2       = self.mbs.AddSensor(SensorNode(nodeNumber= TiltBoomFFRF['nRigidBody'], storeInternal=True,
                 #                                              fileName = f"solution/Simulation_Flexible_f_modes_{self.nModes}_angularVelocity2.txt", outputVariableType=exu.OutputVariableType.AngularVelocity))
           
            sForce1          = self.mbs.AddSensor(SensorObject(objectNumber=oHA1, storeInternal=True, 
                                                               outputVariableType=exu.OutputVariableType.Force))
            sForce2          = self.mbs.AddSensor(SensorObject(objectNumber=oHA2, storeInternal=True, 
                                                               outputVariableType=exu.OutputVariableType.Force))
            sDistance1       = self.mbs.AddSensor(SensorObject(objectNumber=oHA1, storeInternal=True, 
                                                               outputVariableType=exu.OutputVariableType.Distance))
            sDistance2       = self.mbs.AddSensor(SensorObject(objectNumber=oHA2, storeInternal=True, 
                                                               outputVariableType=exu.OutputVariableType.Distance))

            
            sVelocity1       = self.mbs.AddSensor(SensorObject(objectNumber=oHA1, storeInternal=True, 
                                                               outputVariableType=exu.OutputVariableType.VelocityLocal))
            sVelocity2       = self.mbs.AddSensor(SensorObject(objectNumber=oHA2, storeInternal=True, 
                                                               outputVariableType=exu.OutputVariableType.VelocityLocal))

            
            
            sPressuresL      = self.mbs.AddSensor(SensorNode(nodeNumber=nODE1, storeInternal=True,
                                                             outputVariableType=exu.OutputVariableType.Coordinates))   
            sPressuresT      = self.mbs.AddSensor(SensorNode(nodeNumber=nODE2, storeInternal=True,
                                                             outputVariableType=exu.OutputVariableType.Coordinates))   
            
            DeflectionF          = self.mbs.AddSensor(SensorSuperElement(bodyNumber=TiltBoomFFRF['oFFRFreducedOrder'], meshNodeNumber=MarkerTip, 
                                                                   storeInternal=True, outputVariableType=exu.OutputVariableType.DisplacementLocal ))
            
            
         
            
            # sPressures_t    = self.mbs.AddSensor(SensorNode(nodeNumber=nODE1, storeInternal=True,outputVariableType=exu.OutputVariableType.Coordinates_t))   
            # self.dictSensors['sPressures_t']=sPressures_t
            
            # def UFsensor(mbs, t, sensorNumbers, factors, configuration):
            #     val1 = mbs.GetObjectParameter(self.oHA1, 'valveOpening0')
            #     val2 = mbs.GetObjectParameter(self.oHA2, 'valveOpening0')
                
            #     return [val1, val2] #return angle in degree
        else:
            #Angle1       = self.mbs.AddSensor(SensorBody( bodyNumber=b2,localPosition=pMid2, storeInternal=True, 
             #                                            fileName = 'solution/Simulation_Rigid_angle1.txt', outputVariableType=exu.OutputVariableType.Rotation))
            #Angle2       = self.mbs.AddSensor(SensorBody(bodyNumber=b3, localPosition= pMid3, storeInternal=True,
             #                                            fileName = 'solution/Simulation_Rigid_angle2.txt', outputVariableType=exu.OutputVariableType.Rotation))
            #AngVelocity1       = self.mbs.AddSensor(SensorBody( bodyNumber=b2,localPosition=pMid2, storeInternal=True,
             #                                                  fileName = 'solution/Simulation_Rigid_angularVelocity1.txt', 
              #                                                 outputVariableType=exu.OutputVariableType.AngularVelocity))
            #AngVelocity2       = self.mbs.AddSensor(SensorBody(bodyNumber=b3, localPosition= pMid3, storeInternal=True,
             #                                                  fileName = 'solution/Simulation_Rigid_angularVelocity2.txt', outputVariableType=exu.OutputVariableType.AngularVelocity))
            sForce1       = self.mbs.AddSensor(SensorObject(objectNumber=oHA1, storeInternal=True, fileName = 'solution/sForce1.txt', outputVariableType=exu.OutputVariableType.Force))
            sForce2       = self.mbs.AddSensor(SensorObject(objectNumber=oHA2, storeInternal=True, fileName = 'solution/sForce2.txt', outputVariableType=exu.OutputVariableType.Force))
            sPressuresL      = self.mbs.AddSensor(SensorNode(nodeNumber=nODE1, storeInternal=True,fileName = 'solution/sPressuresL.txt', outputVariableType=exu.OutputVariableType.Coordinates)) 
            sPressuresT      = self.mbs.AddSensor(SensorNode(nodeNumber=nODE2, storeInternal=True,fileName = 'solution/sPressuresT.txt', outputVariableType=exu.OutputVariableType.Coordinates)) 
            sDistance1       = self.mbs.AddSensor(SensorObject(objectNumber=oHA1, storeInternal=True, 
                                                   outputVariableType=exu.OutputVariableType.Distance))
            sDistance2       = self.mbs.AddSensor(SensorObject(objectNumber=oHA2, storeInternal=True, 
                                                   outputVariableType=exu.OutputVariableType.Distance))
            sVelocity1       = self.mbs.AddSensor(SensorObject(objectNumber=oHA1, storeInternal=True, 
                                                               outputVariableType=exu.OutputVariableType.VelocityLocal))
            sVelocity2       = self.mbs.AddSensor(SensorObject(objectNumber=oHA2, storeInternal=True, 
                                                               outputVariableType=exu.OutputVariableType.VelocityLocal))
            
            DeflectionF      = self.mbs.AddSensor(SensorBody(bodyNumber=b6, localPosition=[2.45, 0.05, 0], 
                                                                   storeInternal=True, outputVariableType=exu.OutputVariableType.Displacement ))

        
        # self.dictSensors['angle1']=Angle1
        # self.dictSensors['angle2']=Angle2
        # self.dictSensors['anularVelocity1']=AngVelocity1
        # self.dictSensors['anularVelocity2']=AngVelocity2
        self.dictSensors['sForce1']=sForce1
        self.dictSensors['sForce2']=sForce2 
        self.dictSensors['sDistance1']=sDistance1
        self.dictSensors['sDistance2']=sDistance2
        self.dictSensors['sVelocity1']=sVelocity1 
        self.dictSensors['sVelocity2']=sVelocity2
        self.dictSensors['sPressuresL']=sPressuresL
        self.dictSensors['sPressuresT']=sPressuresT
        self.dictSensors['sensorTip']=DeflectionF



        #assemble and solve    
        self.mbs.Assemble()
        self.simulationSettings = exu.SimulationSettings()   
        self.simulationSettings.solutionSettings.sensorsWritePeriod = (self.endTime / (self.nStepsTotal))
        
        self.simulationSettings.timeIntegration.numberOfSteps            = self.GetNSimulationSteps()
        self.simulationSettings.timeIntegration.endTime                  = self.endTime
        self.simulationSettings.timeIntegration.verboseModeFile          = 0
        #self.simulationSettings.timeIntegration.verboseMode              = self.verboseMode
        self.simulationSettings.timeIntegration.newton.useModifiedNewton = True
        self.simulationSettings.linearSolverType                         = exu.LinearSolverType.EigenSparse
        self.simulationSettings.timeIntegration.stepInformation         += 8
        #self.simulationSettings.displayStatistics                        = True
        #self.simulationSettings.displayComputationTime                   = True
        self.simulationSettings.linearSolverSettings.ignoreSingularJacobian=True
        #self.simulationSettings.timeIntegration.generalizedAlpha.spectralRadius  = 0.7

        self.SC.visualizationSettings.nodes.show = False
        
        if self.Visualization:
            self.SC.visualizationSettings.window.renderWindowSize            = [1600, 1200]        
            self.SC.visualizationSettings.openGL.multiSampling               = 4        
            self.SC.visualizationSettings.openGL.lineWidth                   = 3  
            self.SC.visualizationSettings.general.autoFitScene               = False      
            self.SC.visualizationSettings.nodes.drawNodesAsPoint             = False        
            self.SC.visualizationSettings.nodes.showBasis                    = True 
            #self.SC.visualizationSettings.markers.                    = True
            #exu.StartRenderer()
        
        
        
        if self.StaticCase or self.StaticInitialization:
            self.mbs.variables['isStatics'] = True
            self.simulationSettings.staticSolver.newton.relativeTolerance = 1e-7
            # self.simulationSettings.staticSolver.stabilizerODE2term = 2
            self.simulationSettings.staticSolver.verboseMode = self.verboseMode
            self.simulationSettings.staticSolver.numberOfLoadSteps = 1
            self.simulationSettings.staticSolver.constrainODE1coordinates = True #constrain pressures to initial values
    
            exu.SuppressWarnings(True)
            self.mbs.SolveStatic(self.simulationSettings, 
                        updateInitialValues=True) #use solution as new initial values for next simulation
            exu.SuppressWarnings(False)
            #now deactivate distance constraint:
            force1 = self.mbs.GetObjectOutput(oDC, variableType=exu.OutputVariableType.Force)
            force2 = self.mbs.GetObjectOutput(oDCT, variableType=exu.OutputVariableType.Force)
            
            if self.verboseMode:
                print('initial force=', force)
            
            #deactivate distance constraint
            self.mbs.SetObjectParameter(oDC, 'activeConnector', False)
            self.mbs.SetObjectParameter(oDCT, 'activeConnector', False)
            
            
            #overwrite pressures:
            if oHA1 != None:
                dictHA1 = self.mbs.GetObject(oHA1)
                dictHA2 = self.mbs.GetObject(oHA2)
                
                nodeHA1 = dictHA1['nodeNumbers'][0]
                nodeHA2 = dictHA2['nodeNumbers'][0]
                
                A_HA1 = dictHA1['chamberCrossSection0']
                pDiff1 = force1/A_HA1
                pDiff2 = force2/A_HA1
            
                # #now we would like to reset the pressures:
                # #2) change the initial values in the system vector
            
                sysODE1 = self.mbs.systemData.GetODE1Coordinates(configuration=exu.ConfigurationType.Initial)
                
                nODE1index = self.mbs.GetNodeODE1Index(nodeHA1) #coordinate index for node nodaHA1
                nODE2index = self.mbs.GetNodeODE1Index(nodeHA2) #coordinate index for node nodaHA1
                
                if self.verboseMode:
                    # print('sysODE1=',sysODE1)
                    print('p0,p1=',sysODE1[nODE1index],sysODE1[nODE1index+1])
                    print('p2,p3=',sysODE1[nODE2index],sysODE1[nODE2index+1])
                
                sysODE1[nODE1index] += pDiff1 #add required difference to pressure
                sysODE1[nODE2index] += pDiff2 #add required difference to pressure

    
                #now write the updated system variables:
                self.mbs.systemData.SetODE1Coordinates(coordinates=sysODE1, configuration=exu.ConfigurationType.Initial)
                
                #exu.StartRenderer()
               
                if self.verboseMode:
                    print('new p0,p1=',sysODE1[nODE1index],sysODE1[nODE1index+1])
                    print('new p2,p3=',sysODE1[nODE2index],sysODE1[nODE2index+1])

            self.mbs.variables['isStatics'] = False

        if self.Visualization:
            # self.SC.WaitForRenderEngineStopFlag()
            exu.StopRenderer()
        
        
        if not self.StaticCase:
            exu.SolveDynamic(self.mbs, simulationSettings=self.simulationSettings,
                                solverType=exu.DynamicSolverType.TrapezoidalIndex2)

###############################################################################
                            # --- SLIDE TEST ---
###############################################################################
    def ComputeSLIDE(self, Patu):
        
        # Inputs and outputs from the simulation.
        from matplotlib.patches import Rectangle
        DS                      = self.dictSensors
        sensorTip               = self.dictSensors['sensorTip']
        dampedSteps             = np.array([-1])
        data1                   =  self.inputTimeU1
        Time                    = self.timeVecOut
        self.deflection         = np.zeros((self.nStepsTotal,2))
        self.deflection[:,0]    = Time
        self.deflection[:,1]    = self.mbs.GetSensorStoredData(sensorTip)[0:self.nStepsTotal,2]
        
        
        #Assumptions
        # threshold_deflection    = 0.20*np.max(np.abs(self.deflection[0:self.nStepsTotal,1]))
        threshold_deflection    = 1.05*np.mean((self.deflection[0:self.nStepsTotal,1]))
        min_consecutive_steps   = int(0.10*self.nStepsTotal)
        consecutive_count       = 0
        SLIDE_time              = 0
        
        ##################################################################
                            #SLIDE Computing#
        ##################################################################
        for i in range(len(self.deflection[:,1])):
                if (self.deflection[i, 1]) < threshold_deflection:
                    consecutive_count += 1
                    
                    if consecutive_count == 1: 
                        SLIDE_time = self.deflection[i, 0]
                        
                    if consecutive_count >= min_consecutive_steps: 
                        SLIDE_time =SLIDE_time
                        
                        return SLIDE_time, i
                else:
                  consecutive_count = 0 
            
        self.n_td    =   int(SLIDE_time*self.nStepsTotal)  
        
        print(f'SLIDE window:{SLIDE_time} s')
        print(f'Steps in SLIDE window:{self.n_td}')
        
        
        if Patu==False:
            # Control signal
           a40_width_inches, a40_height_inches = 8.3 / 2, 11.7 / 4  
           fig, ax = plt.subplots(figsize=(a40_width_inches, a40_height_inches))
           
           ax.plot(data1[:,0], data1[:,1], color='k')
           ax.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': 10})
           ax.set_ylabel(r'Control signal, V', fontdict={'family': 'Times New Roman', 'size': 10})
           
           ax.axvspan(0, 0.201, ymin = 0.5, ymax = 0.90, color='lightgreen', 
                      alpha=0.2, label='Spool open')
           
           ax.text(0.18, -0.60, 'Spool open', horizontalalignment='center', color='black', fontsize=8, 
                       bbox=dict(facecolor='white', edgecolor='black', pad=4.0))
           
           ax.annotate('',                      # no text
                       xy=(0.203, -0.25),         # head of the arrow (end point)
                       xytext=(-0.003, -0.25),        # tail of the arrow (start point)
                       arrowprops=dict(arrowstyle="<->", color='black', lw=2))
           
           ax.set_xlim(0, 1)
           # Set ticks
           ax.set_xticks(np.linspace(0, 1, 6))
           ax.set_yticks(np.linspace(-1.25, 1.25, 6))
           ax.tick_params(axis='both', labelsize=8)  # Change 8 to the desired font size
           # Enable grid and legend
           ax.grid(True)
           # Adjust layout and save the figure
           plt.tight_layout()
           plt.savefig('SLIDE/LiftBoom/TestControl_Plot.png', format='png', dpi=300)
           plt.show()
            
           # Deflection
           fig, ax = plt.subplots(figsize=(a40_width_inches, a40_height_inches))
           ax.plot(Time, 1000*self.deflection[:,1], color='gray')
           ax.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': 10})
           ax.set_ylabel(r'${\delta}_y$, mm', fontdict={'family': 'Times New Roman', 'size': 10})
           ax.text(0.34 + 0.16, 1.525, r'${{\delta}_y}^{*}$', horizontalalignment='center', color='black', fontsize=10, 
                     bbox=dict(facecolor='white', edgecolor='black', pad=4.0))
           ax.axvline(0.34, color='red', alpha=0.5, linestyle='-.', linewidth=2, label='Midpoint of Open')
           ax.axvspan(0, 0.34, ymin = 0, ymax = 1, color='lightgreen', 
                       alpha=0.2, label='Spool open')            
           ax.text(0.1, 2.60, r'$t_d$', horizontalalignment='center', color='black', fontsize=10, 
                        bbox=dict(facecolor='white', edgecolor='black', pad=4.0))
           ax.annotate('',                      # no text
                        xy=(0.343, 3.8),         # head of the arrow (end point)
                        xytext=(-0.003, 3.8),        # tail of the arrow (start point)
                        arrowprops=dict(arrowstyle="<->", color='black', lw=2))
           center_x, center_y = 0.34, 0.47162  # These replace your damped_time -0.52
           width, height = 0.02, 0.18  # Size of the rectangle, adjust as necessary
           rectangle = Rectangle((center_x - width / 2, center_y - height / 2), width, height,
                                  edgecolor='black', facecolor='black', fill=True)
           ax.add_patch(rectangle)
           ax.annotate('',  # No text, just the arrow
                        xy=(center_x, center_y),  # Arrow points to the center of the rectangle
                        xytext=(0.458, 1.525),  # Starting point of the arrow at text location
                        arrowprops=dict(arrowstyle="->", color='black'))
           ax.set_xlim(0, 1)
           # Set ticks
           ax.set_xticks(np.linspace(0, 1, 6))
           ax.set_yticks(np.linspace(-10, 10, 6))
           ax.tick_params(axis='both', labelsize=8)  # Change 8 to the desired font size
           # Enable grid and legend
           ax.grid(True)
           # Adjust layout and save the figure
           plt.tight_layout()
           plt.savefig('SLIDE/LiftBoom/TestDef_Plot.png', format='png', dpi=300)
           plt.show()
            
        else:
            data2  =  self.inputTimeU2
           
            # For publication
            a40_width_inches, a40_height_inches = 8.3 / 2, 11.7 / 4  
            fig, ax = plt.subplots(figsize=(a40_width_inches, a40_height_inches))           
            ax.plot(Time, data1[:,1], color='k')
            ax.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': 10})
            ax.set_ylabel(r'Control signal, V', fontdict={'family': 'Times New Roman', 'size': 10})            
            ax.axvspan(0, Time[-1]*0.2015, ymin = 0.5, ymax = 0.90, color='lightgreen', 
                       alpha=0.2, label='Spool open')            
            ax.text(Time[-1]*0.15, -0.60, 'Spool open', horizontalalignment='center', color='black', fontsize=8, 
                        bbox=dict(facecolor='white', edgecolor='black', pad=4.0))           
            ax.annotate('',                      # no text
                        xy=(Time[-1]*0.2015, -0.25),         # head of the arrow (end point)
                        xytext=(-0.003, -0.25),        # tail of the arrow (start point)
                        arrowprops=dict(arrowstyle="<->", color='black', lw=2))
            ax.set_xlim(0, Time[-1])
            # Set ticks
            ax.set_xticks(np.linspace(0, Time[-1], 6))
            ax.set_yticks(np.linspace(-1.25, 1.25, 6))
            ax.tick_params(axis='both', labelsize=8)  # Change 8 to the desired font size
            # Enable grid and legend
            ax.grid(True)
            # Adjust layout and save the figure
            plt.tight_layout()
            plt.savefig('SLIDE/PATU/SLIDE_Liftcontrol_Plot.png', format='png', dpi=300)
            plt.show()
            
            
            # For publication
            a40_width_inches, a40_height_inches = 8.3 / 2, 11.7 / 4  
            fig, ax = plt.subplots(figsize=(a40_width_inches, a40_height_inches))           
            ax.plot(Time, data2[:,1], color='k')
            ax.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': 10})
            ax.set_ylabel(r'Control signal, V', fontdict={'family': 'Times New Roman', 'size': 10})            
            ax.axvspan(0, Time[-1]*0.2015, ymin = 0.1, ymax = 0.50, color='lightgreen', 
                       alpha=0.2, label='Spool open')            
            ax.text(Time[-1]*0.15, 0.60, 'Spool open', horizontalalignment='center', color='black', fontsize=8, 
                        bbox=dict(facecolor='white', edgecolor='black', pad=4.0))           
            ax.annotate('',                      # no text
                        xy=(Time[-1]*0.2015, 0.25),         # head of the arrow (end point)
                        xytext=(-0.003, 0.25),        # tail of the arrow (start point)
                        arrowprops=dict(arrowstyle="<->", color='black', lw=2))
            ax.set_xlim(0, Time[-1])
            # Set ticks
            ax.set_xticks(np.linspace(0, Time[-1], 6))
            ax.set_yticks(np.linspace(-1.25, 1.25, 6))
            ax.tick_params(axis='both', labelsize=8)  # Change 8 to the desired font size
            # Enable grid and legend
            ax.grid(True)
            # Adjust layout and save the figure
            plt.tight_layout()
            plt.savefig('SLIDE/PATU/SLIDE_Tiltcontrol_Plot.png', format='png', dpi=300)
            plt.show()
            

            # Deflection
            fig, ax = plt.subplots(figsize=(a40_width_inches, a40_height_inches))
            ax.plot(Time, 1000*self.deflection[:,1] , color='gray')
            ax.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': 10})
            ax.set_ylabel(r'${\delta}_y$, mm', fontdict={'family': 'Times New Roman', 'size': 10})
            ax.text(0.34 + 0.16, 0.80, r'${{\delta}_y}^{*}$', horizontalalignment='center', color='black', fontsize=10, 
                     bbox=dict(facecolor='white', edgecolor='black', pad=4.0))
            ax.axvline(0.34, color='red', alpha=0.5, linestyle='-.', linewidth=2, label='Midpoint of Open')
        
            ax.axvspan(0, 0.34, ymin = 0, ymax = 1, color='lightgreen', 
                       alpha=0.2, label='Spool open')
            
            ax.text(0.1, 0.60, r'$t_d$', horizontalalignment='center', color='black', fontsize=10, 
                        bbox=dict(facecolor='white', edgecolor='black', pad=4.0))
            
            ax.annotate('',                      # no text
                        xy=(0.343, 0.85),         # head of the arrow (end point)
                        xytext=(-0.003, 0.85),        # tail of the arrow (start point)
                        arrowprops=dict(arrowstyle="<->", color='black', lw=2))
            
            center_x, center_y = 0.34, 0.47162  # These replace your damped_time, -0.52
            width, height = 0.02, 0.18  # Size of the rectangle, adjust as necessary

            rectangle = Rectangle((center_x - width / 2, center_y - height / 2), width, height,
                                  edgecolor='black', facecolor='black', fill=True)
            ax.add_patch(rectangle)
            
            ax.annotate('',  # No text, just the arrow
                        xy=(center_x, center_y),  # Arrow points to the center of the rectangle
                        xytext=(0.458, 0.525),  # Starting point of the arrow at text location
                        arrowprops=dict(arrowstyle="->", color='black'))
            ax.set_xlim(0, 1)
            ax.set_ylim(-1, 1)
            # Set ticks
            ax.set_xticks(np.linspace(0, 1, 6))
            ax.set_yticks(np.linspace(-1, 1, 6))
            ax.tick_params(axis='both', labelsize=8)  # Change 8 to the desired font size
            # Enable grid and legend
            ax.grid(True)
            # Adjust layout and save the figure
            plt.tight_layout()
            plt.savefig('SLIDE/PATU/SLIDE_deflection_Plot.png', format='png', dpi=300)
            plt.show()
                
        return self.n_td
    
    
        # data_dict = data[1]
        
        # for key, value in data_dict.items(): 
        #         title = f' {key}'  # Customize your title
        #         plotData = np.column_stack([Time, value])
        #         self.mbs.PlotSensor(plotData, title=title, newFigure = True, closeAll = False)

        
        #return    
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__': #include this to enable parallel processing

    model = NNHydraulics(nStepsTotal=250, endTime=1, 
                         verboseMode=1)
    
    inputData = model.CreateInputVector(0)
    [inputData, output, slideSteps] = model.ComputeModel(inputData, verboseMode=True, 
                                                                     solutionViewer=False)
    