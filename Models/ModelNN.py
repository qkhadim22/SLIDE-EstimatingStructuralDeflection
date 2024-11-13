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

import exudyn as exu
from exudyn.itemInterface import *
from exudyn.utilities import *
from exudyn.FEM import *
import matplotlib.pyplot as plt
from exudyn.plot import PlotSensor, listMarkerStyles


import scipy.io
import os
import numpy as np

import math as mt
from math import sin, cos, sqrt, pi, tanh
import time


from SLIDE.fnnModels import ModelComputationType, NNtestModel

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
pMid2           = np.array([1.229248, 0.055596, 0])  # center of mass

L3              = 2.580         # Length in x-direction
H3              = 0.419         # Height in y-direction
W3              = 0.220         # Width in z-direction

L4              = 0.557227    # Length in x-direction
H4              = 0.1425      # Height in y-direction
W4              = 0.15        # Width in z-direction

L5              = 0.569009       # Length in x-direction
H5              = 0.078827       # Height in y-direction
W5              = 0.15           # Width in z-direction
        
pS              = 100e5
pT              = 1e5                           # Tank pressure
Qn1             = 10*(18/60000)/(sqrt(20e5)*9.9)                      # Nominal flow rate of valve at 18 l/min under
Qn2             = 10*(22/60000)/(sqrt(20e5)*9.9)                      # Nominal flow rate of valve at 18 l/min under

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

#fileNameT       = 'TiltBoomANSYS/TiltBoom' #for load/save of FEM data

feL             = FEMinterface()
feT             = FEMinterface()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class NNHydraulics():

    #initialize class 
    def __init__(self, nStepsTotal=100, endTime=0.5, ReD = 3.35e-3, mL= 50, nnType='FFN',
                 nModes = 2, 
                 Flexible = True,
                 loadFromSavedNPY=True,
                 visualization = False,
                 verboseMode = 0):

        NNtestModel.__init__(self)

        self.nStepsTotal = nStepsTotal
        self.nnType = nnType
        self.endTime = endTime
        
        #+++++++++++++++++++++++++++++
        #from hydraulics:
        self.nModes = nModes
        self.Flexible =True
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

        self.p1     = 2e6
        self.p2     = 2e6
        self.p3     = 2e6
        self.p4     = 2e6
        
        self.angleMinDeg1 = -10
        self.angleMaxDeg1 = 50
        
        self.angleMinDeg2 = -60
        self.angleMaxDeg2 = -10
        
        self.nOutputs = 1    #tip deflection; number of outputs to be predicted
        #self.nInputs = 5    #U, p0, p1, s, s_t
        
        self.nInputs = 10    #U, p0, s, p1, s_t
        self.numSensors = 10

        self.verboseMode = verboseMode

        self.modelName = 'hydraulics'
        self.modelNameShort = 'hyd'

        self.scalPressures1   = 3.50e7 #this is the approx. size of pressures
        self.scalPressures2   = 3.50e7 #this is the approx. size of pressures

        self.scalStroke1      = L_Cyl1+L_Pis1
        self.scaldStroke1     = 0.25
        
        self.scalStroke2      = L_Cyl2+L_Pis2
        self.scaldStroke2     = 0.25
        
        self.scalU1          = 1
        self.scalU2          = 1
        self.scalOut         = 12e-3
        
        self.inputScaling = np.hstack((self.scalU1*np.ones(1*(self.nStepsTotal)), 
                                       self.scalU2*np.ones(1*(self.nStepsTotal)),
                                       self.scalStroke1*np.ones(1*(self.nStepsTotal)),
                                       self.scalStroke2*np.ones(1*(self.nStepsTotal)),
                                       self.scaldStroke1*np.ones(1*(self.nStepsTotal)),
                                       self.scaldStroke2*np.ones(1*(self.nStepsTotal)),
                                       self.scalPressures1*np.ones(1*(self.nStepsTotal)),
                                       self.scalPressures1*np.ones(1*(self.nStepsTotal)), 
                                       self.scalPressures2*np.ones(1*(self.nStepsTotal)),
                                       self.scalPressures2*np.ones(1*(self.nStepsTotal))))
        
        self.outputScaling = self.scalOut*np.ones((self.nStepsTotal, self.nOutputs))
        
        self.timeVecOut = np.arange(1,self.nStepsTotal+1)/self.nStepsTotal*self.endTime
        

    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def IsFFN(self):
        return self.nnType == 'FFN'
    
    def GetInputScaling(self):
        return self.inputScalingFactor*self.inputScaling
    
    #create initialization of (couple of first) hidden states
    def CreateHiddenInit(self, isTest):
        return np.array([])
    
    def GetOutputScaling(self):
        return self.outputScalingFactor*self.outputScaling
    
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
    def CreateInputVector(self, relCnt = 0, isTest=False, dampedwindow=False):
        
        vec = np.zeros(self.GetInputScaling().shape)
    
        U1  = np.zeros(self.nStepsTotal)
        U2  = np.zeros(self.nStepsTotal)

        def create_random_input_signal():
            
            ten_percent             = int(0.1 * self.nStepsTotal)  
            
            # 20 % step signals
            segment1               = np.ones(2*ten_percent)               # 20% of nStepsTotal at 1  
            segment2               = -1 * np.ones(2*ten_percent)          # 20% of nStepsTotal at -1
            segment3               = np.zeros(2*ten_percent)
            
            if dampedwindow==False: 

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
        U2          = create_random_input_signal()
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
    def SplitInputData(self, inputData):
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
    def ComputeModel(self, inputData, verboseMode = 0, solutionViewer = False):
        self.CreateModel()
        # print('compute model')
        self.verboseMode = verboseMode
        output = []
        #set input data ...
        inputDict = self.SplitInputData(np.array(inputData))
        outputData = self.SplitOutputData(np.array(output))
        

        self.inputTimeU1 = np.zeros((self.nStepsTotal,2))
        self.inputTimeU2 = np.zeros((self.nStepsTotal,2))

        self.inputTimeU1[:,0] = self.timeVecOut
        self.inputTimeU1[:,1] = inputDict['U1']
        
        self.inputTimeU2[:,0] = self.timeVecOut
        self.inputTimeU2[:,1] = inputDict['U2']

        self.mbs.variables['inputTimeU1'] = self.inputTimeU1
        self.mbs.variables['inputTimeU2'] = self.inputTimeU2

        
        self.mbs.variables['theta1'] = inputData[self.nStepsTotal*3]
        self.mbs.variables['theta2'] = inputData[self.nStepsTotal*4]
        
        self.PatuCrane(self.mbs.variables['theta1'], self.mbs.variables['theta2'],
                                   self.p1, self.p2, self.p3, self.p4)

        if self.Visualization:
           self.mbs.SolutionViewer()
            
        # Inputs and outputs from the simulation.
        DS = self.dictSensors
        
        
        #++++++++++++++++++++++++++
        #Input data
        #inputDict['t']  =  self.timeVecOut
        inputDict['U1']     =  inputData[0:self.nStepsTotal]
        inputDict['U2']     =  inputData[1*self.nStepsTotal:2*self.nStepsTotal]
        inputDict['s1']     =  self.mbs.GetSensorStoredData(DS['sDistance1'])[0:1*self.nStepsTotal,1:2]
        inputDict['s2']     =  self.mbs.GetSensorStoredData(DS['sDistance2'])[0:1*self.nStepsTotal,1:2]
        inputDict['ds1']    =  self.mbs.GetSensorStoredData(DS['sVelocity1'])[0:1*self.nStepsTotal,1:2]
        inputDict['ds2']    =  self.mbs.GetSensorStoredData(DS['sVelocity2'])[0:1*self.nStepsTotal,1:2]       
        inputDict['p1']     =  self.mbs.GetSensorStoredData(DS['sPressuresL'])[0:1*self.nStepsTotal,1:2]
        inputDict['p2']     =  self.mbs.GetSensorStoredData(DS['sPressuresL'])[0:1*self.nStepsTotal,2:3]
        inputDict['p3']     =  self.mbs.GetSensorStoredData(DS['sPressuresT'])[0:1*self.nStepsTotal,1:2]
        inputDict['p4']     =  self.mbs.GetSensorStoredData(DS['sPressuresT'])[0:1*self.nStepsTotal,2:3]
        
        #Ouputs
        
        outputData          = 0*self.GetOutputScaling()
        outputData[:,0]     = self.mbs.GetSensorStoredData(DS['sensorTip'])[0:self.nStepsTotal,2]
        
        inputDict           = inputDict/self.GetInputScaling()
        outputData          = outputData/self.GetOutputScaling()  
        
        dampedSteps         =  np.array([])

        # dampedSteps         =  np.array([self.hist_window])
            
        return [inputDict, outputData, dampedSteps] 
    



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
        
        graphicsBody1   = GraphicsDataFromSTLfile(fileName1, color4black,verbose=False, invertNormals=True,invertTriangles=True)
        graphicsBody1   = AddEdgesAndSmoothenNormals(graphicsBody1, edgeAngle=0.25*pi,addEdges=True, smoothNormals=True)
        graphicsCOM1    = GraphicsDataBasis(origin=iCube1.com, length=2*W1)
        
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
            p14                 = [9.5e-2,0.24,-7.15e-2]
            p13                 = [9.5e-2,0.24, 7.15e-2]
            radius7             = 2.5e-002
            nodeListPist1T      = feT.GetNodesOnCylinder(p13, p14, radius7, tolerance=1e-2)  
            pJoint2T            = feT.GetNodePositionsMean(nodeListPist1T)
            nodeListPist1TLen   = len(nodeListPist1T)
            noodeWeightsPist1T  = [1/nodeListPist1TLen]*nodeListPist1TLen
                    
            boundaryListL   = [nodeListJoint1,nodeListPist1, nodeListJoint2,  nodeListJoint3] 
            boundaryListT  = [nodeListJoint1T, nodeListPist1T]
            
            
                
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
            
            
                    
            TiltBoomFFRF        = TiltBoom.AddObjectFFRFreducedOrder(self.mbs, positionRef=np.array([-0.09, 1.4261, 0])+
                                                                      + np.array([2.76-12e-3,  0.846520-12e-3, 0]), #2.879420180699481+27e-3, -0.040690041435711005+8.3e-2, 0
                                                          initialVelocity=[0,0,0], 
                                                          initialAngularVelocity=[0,0,0],
                                                          rotationMatrixRef  = RotationMatrixZ(mt.radians(self.theta2))   ,
                                                          gravity=g,
                                                          color=colLift,)
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
                    
            Marker13        = self.mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=TiltBoomFFRF['oFFRFreducedOrder'], 
                                                                                  meshNodeNumbers=np.array(nodeListJoint1T), #these are the meshNodeNumbers
                                                                                  weightingFactors=noodeWeightsJoint1T))
                    
            Marker14        = self.mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=TiltBoomFFRF['oFFRFreducedOrder'], 
                                                                                  meshNodeNumbers=np.array(nodeListPist1T), #these are the meshNodeNumbers
                                                                                  weightingFactors=noodeWeightsPist1T))  
            
            MarkerTip   = feT.GetNodeAtPoint(np.array([1.80499995,  0.266000003, 0.0510110967]))
       
        else:
            iCube2          = RigidBodyInertia(mass=143.66, com=pMid2,
                                            inertiaTensor=np.array([[1.055433, 1.442440,  -0.000003],
                                                                    [ 1.442440,  66.577004, 0],
                                                                    [ -0.000003,              0  ,  67.053707]]),
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
                                    position=LiftP + np.array([2.76,  0.846520, 0]),  # pMid2
                                    rotationMatrix=RotationMatrixZ(mt.radians(self.theta2)),
                                    gravity=g,
                                    graphicsDataList=[graphicsCOM3, graphicsBody3])
            Marker13        = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b3, localPosition=[0, 0, 0]))                        #With LIft Boom 
            Marker14        = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b3, localPosition=[-0.095, 0.24043237, 0])) 
            
        # # # # # # 4th Body: Bracket 1
        Bracket1L       = LiftP + np.array([2.579002-6e-3, 0.7641933967880499+5e-3, 0])
        pMid4           = np.array([0.004000, 0.257068, 0])  # center of mass, body0,0.004000,-0.257068
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
                                                  rotationMatrix=RotationMatrixZ(mt.radians(20.6)), #-0.414835768117858
                                                  gravity=g,graphicsDataList=[graphicsCOM4, graphicsBody4])
                
        # # # # 5th Body: Bracket 2
        pMid5           = np.array([0.212792, 0, 0])  # center of mass, body0
        Bracket1B       = LiftP + np.array([2.791569774713015+112e-3, 1.08619-40e-3, 0])  #0.285710892999728-1.8*0.0425, -0.356968041652145+0.0525
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
                                                  rotationMatrix=RotationMatrixZ(mt.radians(160.1)) ,   #-5, 140
                                                  gravity=g,graphicsDataList=[graphicsCOM5, graphicsBody5])
                
            
                
        Marker15        = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b4, localPosition=[0, 0, 0]))                        #With LIft Boom 
        Marker16        = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b4, localPosition=[0.0425, 0.472227402-15e-3, 0]))  
        Marker18        = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b5, localPosition=[0, 0, 0]))                        #With LIft Boom 
        Marker19        = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b5, localPosition=[0.475+5e-3, 0, 0]))                   #With LIft Boom,-0.475 
                
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
        def UFfrictionSpringDamper1(mbs, t, itemIndex, u, v, k, d, f0): 
                return   1*(Fc*tanh(4*(abs(v    )/vs))+(Fs-Fc)*((abs(v    )/vs)/((1/4)*(abs(v    )/vs)**2+3/4)**2))*np.sign(v )+sig2*v    *tanh(4)

                    
        def UFfrictionSpringDamper2(mbs, t, itemIndex, u, v, k, d, f0): 
                 return   0.5*(Fc*tanh(4*(abs(v    )/vs))+(Fs-Fc)*((abs(v    )/vs)/((1/4)*(abs(v    )/vs)**2+3/4)**2))*np.sign(v )+sig2*v    *tanh(4)
                    
                    
        oFriction1       = self.mbs.AddObject(ObjectConnectorSpringDamper(markerNumbers=[Marker5, Marker8], referenceLength=0.001,stiffness=0,
                                                            damping=0, force=0, velocityOffset = 0., activeConnector = True,
                                                            springForceUserFunction=UFfrictionSpringDamper1,
                                                              visualization=VSpringDamper(show=False) ))
                        
        oFriction2       = self.mbs.AddObject(ObjectConnectorSpringDamper(markerNumbers=[Marker9, Marker16], referenceLength=0.001,stiffness=0,
                                                              damping=0, force=0, velocityOffset = 0., activeConnector = True,
                                                              springForceUserFunction=UFfrictionSpringDamper2,
                                                                visualization=VSpringDamper(show=False) ))
        oHA1 = None
        oHA2 = None

        if True: 
                oHA1                = self.mbs.AddObject(HydraulicActuatorSimple(name='LiftCylinder', markerNumbers=[ Marker5, Marker8], 
                                                        nodeNumbers=[nODE1], offsetLength=LH1, strokeLength=L_Pis1, chamberCrossSection0=A[0], 
                                                        chamberCrossSection1=A[1], hoseVolume0=V1, hoseVolume1=V2, valveOpening0=0, 
                                                        valveOpening1=0, actuatorDamping=5e5, oilBulkModulus=Bo, cylinderBulkModulus=Bc, 
                                                        hoseBulkModulus=Bh, nominalFlow=Qn1, systemPressure=pS, tankPressure=pT, 
                                                        useChamberVolumeChange=True, activeConnector=True, 
                                                        visualization={'show': True, 'cylinderRadius': 50e-3, 'rodRadius': 28e-3, 
                                                                        'pistonRadius': 0.04, 'pistonLength': 0.001, 'rodMountRadius': 0.0, 
                                                                        'baseMountRadius': 20.0e-3, 'baseMountLength': 20.0e-3, 'colorCylinder': color4orange,
                                                                    'colorPiston': color4grey}))
                    
                oHA2 = self.mbs.AddObject(HydraulicActuatorSimple(name='TiltCylinder', markerNumbers=[Marker9, Marker16], 
                                                          nodeNumbers=[nODE2], offsetLength=LH2, strokeLength=L_Pis2, chamberCrossSection0=A[0], 
                                                          chamberCrossSection1=A[1], hoseVolume0=V1, hoseVolume1=V2, valveOpening0=0, 
                                                          valveOpening1=0, actuatorDamping=0.06e5, oilBulkModulus=Bo, cylinderBulkModulus=Bc, 
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
        self.simulationSettings.displayComputationTime                   = True
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
            #self.simulationSettings.staticSolver.verboseMode = self.verboseMode
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

            
        
    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    def Plotting(self, data): 
        Time = self.timeVecOut
       
        sDistance1         = np.loadtxt(f"solution/Simulation_Flexible_f_modes_{self.nModes}_angle1.txt", delimiter=',')
        sDistance2         = np.loadtxt(f"solution/Simulation_Flexible_f_modes_{self.nModes}_angle2.txt", delimiter=',')
        
        # sDistance1    = np.loadtxt(f"solution/Simulation_Flexible_f_modes_{self.nModes}_sDistance1.txt", delimiter=',')
        # sDistance2    = np.loadtxt(f"solution/Simulation_Flexible_f_modes_{self.nModes}_sDistance2.txt", delimiter=',')
        sVelocity1   = np.loadtxt(f"solution/Simulation_Flexible_f_modes_{self.nModes}_angularVelocity1.txt", delimiter=',')
        sVelocity2   = np.loadtxt(f"solution/Simulation_Flexible_f_modes_{self.nModes}_angularVelocity2.txt", delimiter=',')
        
        
        # Lift actuator
        plt.figure(figsize=(10, 5))
        plt.plot(sDistance1[:, 0], np.rad2deg(sDistance1[:, 3]), label='Simulation', linestyle='--', marker='x', 
         linewidth=1, markersize=2, color='black')
        plt.xlabel('Time, s')  # Adjust label as appropriate
        plt.ylabel('Angle, degree')  # Adjust label as appropriate
        plt.legend(loc='upper left')  # Specify legend location
        plt.grid(True)  # Add grid
        # Set axis limits
        plt.xlim(0, 10)
        plt.ylim(0, 50)
        plt.tight_layout()
        plt.savefig(f"solution/Flexible_f_modes_{self.nModes}_angle1.png")
        plt.show()


        plt.figure(figsize=(10, 5))
        plt.plot(sDistance1[:, 0], np.rad2deg(sVelocity1[:, 3]), label='Simulation', linestyle='--', marker='x', 
         linewidth=1, markersize=2, color='black')
        plt.xlabel('Time, s')  # Adjust label as appropriate
        plt.ylabel('Angular velocity, deg/s')  # Adjust label as appropriate
        plt.legend(loc='upper left')  # Specify legend location
        plt.grid(True)  # Add grid
        # Set axis limits
        plt.xlim(0, 10)
        plt.ylim(-20, 20)
        plt.savefig(f"solution/Flexible_f_modes_{self.nModes}_angularvelocity1.png")
        plt.show()
        
        
        # Lift actuator
        plt.figure(figsize=(10, 5))
        plt.plot(sDistance1[:, 0], np.rad2deg(sDistance2[:, 3]-sDistance1[:, 3]), label='Simulation', linestyle='--', marker='x', 
        linewidth=1, markersize=2, color='black')
        plt.xlabel('Time, s')  # Adjust label as appropriate
        plt.ylabel('Angle, degree')  # Adjust label as appropriate
        plt.legend(loc='upper right')  # Specify legend location
        plt.grid(True)  # Add grid
        # Set axis limits
        plt.xlim(0, 10)
        plt.ylim(-60, -10)
        plt.tight_layout()
        plt.savefig(f"solution/Flexible_f_modes_{self.nModes}_angle2.png")
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(sDistance1[:, 0], np.rad2deg(sVelocity2[:, 3]-sVelocity1[:, 3]), label='Simulation', linestyle='--', marker='x', 
                 linewidth=1, markersize=2, color='black')
        plt.xlabel('Time, s')  # Adjust label as appropriate
        plt.ylabel('Angular velocity, deg/s')  # Adjust label as appropriate
        plt.legend(loc='upper left')  # Specify legend location
        plt.grid(True)  # Add grid
        # Set axis limits
        plt.xlim(0, 10)
        plt.ylim(-25, 25)
        plt.savefig(f"solution/Flexible_f_modes_{self.nModes}angularvelocity2.png")
        plt.show()
        

        
        
        # data_dict = data[1]
        
        # for key, value in data_dict.items(): 
        #         title = f' {key}'  # Customize your title
        #         plotData = np.column_stack([Time, value])
        #         self.mbs.PlotSensor(plotData, title=title, newFigure = True, closeAll = False)

        
        return    
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__': #include this to enable parallel processing

    model = NNHydraulics(nStepsTotal=250, endTime=1, 
                         verboseMode=1)
    
    inputData = model.CreateInputVector(0)
    output = model.ComputeModel(inputData, verboseMode=True, 
                                solutionViewer=False)
    