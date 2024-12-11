#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  library to support creation and testing of RNN networks
#
# Author:   Johannes Gerstmayr
# Date:     2023-06-23
#
# Copyright:This file is part of Exudyn. Exudyn is free software. You can redistribute it and/or modify it under the terms of the Exudyn license. See 'LICENSE.txt' for more details.
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import exudyn as exu
from exudyn.utilities import *
from exudyn.signalProcessing import GetInterpolatedSignalValue
from exudyn.physics import StribeckFunction

import sys
import numpy as np
from math import sin, cos, pi, tan, exp, sqrt, atan2

from enum import Enum #for data types

class ModelComputationType(Enum):
    dynamicExplicit = 1         #time integration
    dynamicImplicit = 2         #time integration
    static = 3         #time integration
    eigenAnalysis = 4         #time integration

    #allows to check a = ModelComputationType.dynamicImplicit for a.IsDynamic()    
    def IsDynamic(self):
        return (self == ModelComputationType.dynamicExplicit or 
                self == ModelComputationType.dynamicImplicit)

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
    

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class NonlinearOscillator(NNtestModel):

    #initialize class 
    def __init__(self, nStepsTotal=100, useVelocities=True, useInitialValues=True, useFriction=False,
                 nMasses=1, endTime=1, nnType='RNN', variationMKD=True, 
                 nStepForces=1, useHarmonicExcitation=False):
        NNtestModel.__init__(self)

        #required in base:
        self.nStepsTotal = nStepsTotal
        self.nnType = nnType
        self.endTime = endTime

        self.nMasses = nMasses
        self.useFriction = useFriction
        self.nODE2 = 1 #always 1, as only one is measured / actuated
        self.useVelocities = useVelocities
        self.useInitialValues = useInitialValues
        self.useHarmonicExcitation = useHarmonicExcitation
        self.nStepForces = nStepForces
        
        self.computationType = ModelComputationType.dynamicImplicit

        self.inputStep = False #only constant step functions as input
        self.modelName = 'DoublePendulum'
        self.modelNameShort = 'nl-osc'
        
        if useInitialValues and nMasses>1:
            raise ValueError('NonlinearOscillator: in case of useInitialValues=True, nMasses must be 1')


        self.variationMKD = variationMKD #add factors for mass, spring, damper (factors 0.5..2)
        self.initHidden = True and self.IsRNN() #for RNN
        self.smallFact = 1. #0.1    #use small values for inputs to remove tanh nonlinearity

        self.nInit = 2*self.nODE2
        if self.initHidden or not useInitialValues: 
            self.nInit=0
        
        scalForces = 1000.
        self.scalVelocities = 40.
        self.inputScaling = np.ones((self.nStepsTotal, self.nInit+1+self.variationMKD*3)) #2 initial cond., 1 force, 3*MKD
        if self.IsFFN():
            self.inputScaling = np.ones(2*self.useInitialValues+
                                        self.variationMKD*3+
                                        self.nStepsTotal #forces
                                        ) #2 initial cond., 1 force, 3*MKD
            self.inputScaling[self.nInit+self.variationMKD*3:] *= scalForces
        else:
            self.inputScaling[:,self.nInit] *= scalForces


        self.outputScaling = np.ones((self.nStepsTotal, 1+int(self.useVelocities) )) #displacement + velocity

        if self.useVelocities:
            self.outputScaling[:,1] /= self.scalVelocities

        # else:
        #     self.inputScaling = np.array([1.]*int(self.useInitialValues)*2*self.nODE2 + [scalForces]*self.nStepsTotal)
        #     self.outputScaling = np.array([1.]*self.nStepsTotal + [1.]*(int(self.useVelocities)*self.nStepsTotal ))
        

    def CreateModel(self):
        self.SC = exu.SystemContainer()
        self.mbs = self.SC.AddSystem()
    
        L = 1 #distance of masses, length of springs
        self.mass = 1 #weight of one mass
        #self.nMasses = 1 #number of masses
        self.spring = 1.6e3
        self.damper = 0.005*self.spring
        rMass = 0.1*L
        
        self.resonanceFreq = np.sqrt(self.spring/self.mass)/(2*np.pi) #6.36
        omega = 2*np.pi*self.resonanceFreq
    
        gGround = [GraphicsDataOrthoCubePoint(size=[0.1,0.1,0.1], color=color4grey)]
        oGround = self.mbs.AddObject(ObjectGround(visualization=VObjectGround(graphicsData=gGround)) )

        #ground node for first spring
        nGround=self.mbs.AddNode(NodePointGround(referenceCoordinates = [0,0,0]))
        groundMarker=self.mbs.AddMarker(MarkerNodeCoordinate(nodeNumber= nGround, coordinate = 0))
        prevMarker = groundMarker
        
        def UFspring(mbs, t, itemNumber, u, v, k, d, F0):
            #return k*u + 100*np.sign(v) #friction
            return k*u + 50*StribeckFunction(v, muDynamic=1, muStaticOffset=1.5, regVel=1e-4)

        if not self.useFriction:
            UFspring = 0

        gSphere = GraphicsDataSphere(point=[0,0,0], radius=rMass, color=color4blue, nTiles=16)
        lastBody = oGround
        for i in range(self.nMasses):
            node = self.mbs.AddNode(Node1D(referenceCoordinates = [L*(1+i)],
                                      initialCoordinates=[0.],
                                      initialVelocities=[0.]))
            self.massPoint = self.mbs.AddObject(Mass1D(nodeNumber = node, physicsMass=self.mass,
                                             referencePosition=[0,0,0],
                                             visualization=VMass1D(graphicsData=[gSphere])))

            nodeMarker =self.mbs.AddMarker(MarkerNodeCoordinate(nodeNumber= node, coordinate = 0))
            
            #Spring-Damper between two marker coordinates
            self.oSD = self.mbs.AddObject(CoordinateSpringDamper(markerNumbers = [prevMarker, nodeMarker], 
                                                 stiffness = self.spring, damping = self.damper, 
                                                 springForceUserFunction = UFspring,
                                                 visualization=VCoordinateSpringDamper(drawSize=rMass))) 
            prevMarker = nodeMarker
                
        self.timeVecIn = np.arange(0,self.nStepsTotal)/self.nStepsTotal*self.endTime
        self.timeVecOut = np.arange(1,self.nStepsTotal+1)/self.nStepsTotal*self.endTime
        self.mbs.variables['timeVecOut'] = self.timeVecOut
        
        self.fVec = None
 
        self.mbs.variables['fVec'] = self.fVec
    
        def UFforce(mbs, t, load):
            return GetInterpolatedSignalValue (t, mbs.variables['fVec'], mbs.variables['timeVecOut'],
                                               rangeWarning=False)
            
        
        load = self.mbs.AddLoad(LoadCoordinate(markerNumber=prevMarker, load=0, 
                                loadUserFunction=UFforce))
        
        #coordinates of last node are output:
        self.sCoordinates = self.mbs.AddSensor(SensorNode(nodeNumber=node, storeInternal=True,
                                                          outputVariableType=exu.OutputVariableType.Coordinates))
        self.sCoordinates_t = self.mbs.AddSensor(SensorNode(nodeNumber=node, storeInternal=True,
                                                          outputVariableType=exu.OutputVariableType.Coordinates_t))
    
        self.mbs.Assemble()

        self.simulationSettings 
        self.simulationSettings = exu.SimulationSettings() #takes currently set values or default values
        self.simulationSettings.solutionSettings.sensorsWritePeriod = self.endTime / self.nStepsTotal
        self.simulationSettings.timeIntegration.numberOfSteps = self.GetNSimulationSteps()
        self.simulationSettings.timeIntegration.endTime = self.endTime

    #get time vector according to output data
    def GetOutputXAxisVector(self):
        return self.timeVecOut

    #create a randomized input vector
    #relCnt can be used to create different kinds of input vectors (sinoid, noise, ...)
    #isTest is True in case of test data creation
    def CreateInputVector(self, relCnt = 0, isTest=False):

        vec = np.zeros(self.GetInputScaling().shape)
        forces = np.zeros(self.nStepsTotal)
        if self.nStepForces:
            steps = self.smallFact*(2.*np.random.rand(self.nStepForces)-1.) #force values interpolated
            for i, t in enumerate(self.timeVecOut):
                forces[i] = steps[int(self.nStepForces*i/len(self.timeVecOut))]
        if self.useHarmonicExcitation:
            omega = 12*2*np.pi*np.random.rand() #6.4 Hz is eigenfrequency of one-mass oscillator
            amp = np.random.rand()
            phi = 2.*np.pi*np.random.rand()
            forces = amp * np.sin(omega*self.timeVecOut+phi) #gives np.array

        MKD = []
        if self.variationMKD:
            MKD = np.zeros(3)
            #M,K,D must be all the same for one input vector!
            #these are factors!
            rangeMKD = 4 #4.
            a = 1./rangeMKD
            b = 1-a
            MKD[0] = a+b*np.random.rand() #mass
            MKD[1] = a+b*np.random.rand() #spring
            MKD[2] = a+b*np.random.rand() #damper

        initVals = self.smallFact*0.5*(2*np.random.rand(2)-1.) 

        if not self.IsFFN():
            if not self.initHidden and self.useInitialValues:
                vec[:,0] = initVals[0]
                vec[:,1] = initVals[1]
    
            for i, mkd in enumerate(MKD):
                vec[:,(self.nInit+1+i)] = mkd
    
            vec[:,self.nInit] =forces
        else:
            if self.useInitialValues:
                vec[0] = initVals[0]
                vec[1] = initVals[1]
    
            for i, mkd in enumerate(MKD):
                vec[(self.nInit+i)] = mkd
            #print('vec shape', vec.shape, ', force.shape',forces.shape, self.nInit)
            vec[self.nInit+len(MKD):] = forces
            
        return vec

    #create initialization of (couple of first) hidden states (RNN)
    def CreateHiddenInit(self, isTest):
        if self.initHidden:
            vec = np.zeros(2)
            vec[0] = self.smallFact*0.5*(2*np.random.rand()-1.) #initialODE2
            vec[1] = self.smallFact*0.5*(2*np.random.rand()-1.) #initialODE2_t
            return vec
        else:
            return np.array([])
            
    #get number of simulation steps
    def GetNSimulationSteps(self):
        return self.nStepsTotal*10 #10 x finer simulation than output

    #split input data into initial values, forces or other inputs
    #return dict with 'data' and possibly 'initialODE2' and 'initialODE2_t'
    def SplitInputData(self, inputData, hiddenData=None):
        data = np.array(self.GetInputScaling()*inputData)
        rv = {}
        if not self.IsFFN():
            if self.initHidden:
                rv['initialODE2'] = [hiddenData[0]]   
                rv['initialODE2_t'] = [hiddenData[1]]#*self.scalVelocities] 
            elif self.useInitialValues:
                rv['initialODE2'] = data[0,0:self.nODE2]
                rv['initialODE2_t'] = data[0,self.nODE2:(2*self.nODE2)]
    
            if self.variationMKD:
                rv['MKD'] = data[0,(self.nInit+1):(self.nInit+4)] #MKD are the same for all sequences
            rv['data'] = data[:,self.nInit] #forces
        else:
            off = 0
            if self.useInitialValues:
                rv['initialODE2'] = data[0:self.nODE2]
                rv['initialODE2_t'] = data[self.nODE2:(2*self.nODE2)]
                off+=2
    
            if self.variationMKD:
                rv['MKD'] = data[off:(off+3)] #MKD are the same for all sequences
                off+=3

            rv['data'] = data[off:] #forces
        
        return rv

    #split output data to get ODE2 values (and possibly other data, such as ODE2)
    #return dict {'ODE2':[], 'ODE2_t':[]}
    def SplitOutputData(self, outputData):
        rv = {}
        data = outputData
        if outputData.ndim == 1:
            data = outputData.reshape((self.nStepsTotal,1+self.useVelocities))
        rv['ODE2'] = data[:,0]
        if self.useVelocities:
            rv['ODE2_t'] = data[:,1]
        return rv
    
    #convert all output vectors into plottable data (e.g. [time, x, y])
    #the size of data allows to decide how many columns exist
    def OutputData2PlotData(self, outputData, forSolutionViewer=False):
        if self.nMasses != 1 and forSolutionViewer:
            raise ValueError('NonlinearOscillator.OutputData2PlotData: nMasses > 1 is not suitable for SolutionViewer!')
        timeVec = self.GetOutputXAxisVector()
        dataDict = self.SplitOutputData(outputData)
        
        if 'ODE2_t' in dataDict and not forSolutionViewer:
            data = np.vstack((timeVec, dataDict['ODE2'].T, dataDict['ODE2_t'].T)).T
        else:
            data = np.vstack((timeVec, dataDict['ODE2'].T)).T
            
        return data

    #return dict of names to columns for plotdata        
    def PlotDataColumns(self):
        d = {'time':0, 'ODE2':1}
        if self.useVelocities:
            d['ODE2_t'] = 2
        
        return d

    #get compute model with given input data and return output data
    #initialState contains position and velocity states as list of two np.arrays 
    def ComputeModel(self, inputData, hiddenData=None, verboseMode = 0, solutionViewer = False):
        #set input data ...
        inputDict = self.SplitInputData(np.array(inputData), hiddenData)

        if self.variationMKD:
            #mass enters the equations reciprocal ...
            self.mbs.SetObjectParameter(self.massPoint, 'physicsMass', self.mass/(2*inputDict['MKD'][0]))
            self.mbs.SetObjectParameter(self.oSD, 'stiffness', self.spring*2*inputDict['MKD'][1])
            self.mbs.SetObjectParameter(self.oSD, 'damping', self.damper*1*inputDict['MKD'][2])
            
            #self.CreateModel() #must be created newly for each test ...


        # inputDict = self.SplitInputData(self.GetInputScaling() * np.array(inputData))

        if 'initialODE2' in inputDict:
            #print('initODE2',inputDict['initialODE2'])
            self.mbs.systemData.SetODE2Coordinates(inputDict['initialODE2'], configuration=exu.ConfigurationType.Initial)
            self.mbs.systemData.SetODE2Coordinates_t(inputDict['initialODE2_t'], configuration=exu.ConfigurationType.Initial)

        self.mbs.variables['fVec'] = inputDict['data']

        self.simulationSettings.timeIntegration.verboseMode = verboseMode
        self.simulationSettings.solutionSettings.writeSolutionToFile = solutionViewer 
 
        solverType = exu.DynamicSolverType.TrapezoidalIndex2
        if self.useFriction:
            solverType = exu.DynamicSolverType.ExplicitMidpoint
        #solverType = exu.DynamicSolverType.ExplicitEuler
        
        self.mbs.SolveDynamic(self.simulationSettings, 
                         solverType = solverType)
        if solutionViewer:
            self.mbs.SolutionViewer()

        #get sensor data and apply piecewise scaling:
        output = 0*self.GetOutputScaling()
        output[:,0] = self.mbs.GetSensorStoredData(self.sCoordinates)[1:,1] #sensordata includes time
        if self.useVelocities:
            output[:,1] = self.mbs.GetSensorStoredData(self.sCoordinates_t)[1:,1] #sensordata includes time
        
        output = self.GetOutputScaling()*output
        return output







#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class NonlinearBeamStatic(NNtestModel):

    #initialize class 
    def __init__(self, nBeams=16):
        NNtestModel.__init__(self)

        self.nBeams = nBeams
        self.nNodes = self.nBeams+1
        self.nODE2 = (self.nNodes)*4
        self.computationType = ModelComputationType.static

        self.modelName = 'NonlinearBeamStatic'
        self.modelNameShort = 'nl-beam'
        
        scalForces = 2.
        self.inputScaling = np.array([scalForces,scalForces]) #Fx, Fy
        self.outputScaling = np.array([1.]*(self.nNodes*2) ) #x/y positions of nodes
        

    def CreateModel(self):
        self.SC = exu.SystemContainer()
        self.mbs = self.SC.AddSystem()
    
        L = 1 #total beam length
            
        E=2.e11                # Young's modulus of ANCF element in N/m^2
        rho=7800               # density of ANCF element in kg/m^3
        b=0.001                # width of rectangular ANCF element in m
        h=0.002                # height of rectangular ANCF element in m
        A=b*h                  # cross sectional area of ANCF element in m^2
        I=b*h**3/12            # second moment of area of ANCF element in m^4
        
        #generate ANCF beams with utilities function
        cableTemplate = Cable2D(#physicsLength = L / nElements, #set in GenerateStraightLineANCFCable2D(...)
                                physicsMassPerLength = rho*A,
                                physicsBendingStiffness = E*I,
                                physicsAxialStiffness = E*A,
                                physicsBendingDamping=E*I*0.05,
                                useReducedOrderIntegration = 1,
                                #nodeNumbers = [0, 0], #will be filled in GenerateStraightLineANCFCable2D(...)
                                )
        
        positionOfNode0 = [0, 0, 0] # starting point of line
        positionOfNode1 = [L, 0, 0] # end point of line
        
        self.xAxis = np.arange(0,L,L/(self.nBeams+1))
        
        #alternative to mbs.AddObject(Cable2D(...)) with nodes:
        ancf=GenerateStraightLineANCFCable2D(self.mbs,
                        positionOfNode0, positionOfNode1,
                        self.nBeams,
                        cableTemplate, #this defines the beam element properties
                        #massProportionalLoad = [0,-9.81*0,0], #optionally add gravity
                        fixedConstraintsNode0 = [1,1,0,1], #add constraints for pos and rot (r'_y)
                        fixedConstraintsNode1 = [0,0,0,0])
        mANCFLast = self.mbs.AddMarker(MarkerNodePosition(nodeNumber=ancf[0][-1])) #ancf[0][-1] = last node
        
        self.lTipLoad = self.mbs.AddLoad(Force(markerNumber = mANCFLast, 
                                         loadVector = [0, 0, 0], )) 
            
        self.listPosSensors = []
        for node in ancf[0]:
            sPos = self.mbs.AddSensor(SensorNode(nodeNumber=node, storeInternal=True,
                                                 outputVariableType=exu.OutputVariableType.Position))
            self.listPosSensors += [sPos]

    
        self.mbs.Assemble()

        self.simulationSettings 
        self.simulationSettings = exu.SimulationSettings() #takes currently set values or default values
        self.simulationSettings.staticSolver.numberOfLoadSteps = 20
        #reduce tolerances to achieve convergence also for small loads
        self.simulationSettings.staticSolver.newton.absoluteTolerance = 1e-6
        self.simulationSettings.staticSolver.newton.relativeTolerance = 1e-6

    #get time vector according to output data
    def GetOutputXAxisVector(self):
        return self.xAxis

    #create a randomized input vector
    #relCnt can be used to create different kinds of input vectors (sinoid, noise, ...)
    #isTest is True in case of test data creation
    def CreateInputVector(self, relCnt = 0, isTest=False):
        vec = 2.*np.random.rand(*self.GetInputScaling().shape)-1.

        return vec
            
    #split input data into initial values, forces or other inputs
    #return dict with 'data' and possibly 'initialODE2' and 'initialODE2_t'
    def SplitInputData(self, inputData, hiddenData):
        data = np.array(inputData)
        rv = {}
        rv['data'] = data
        return rv

    #split output data to get ODE2 values (and possibly other data, such as ODE2)
    #return dict {'ODE2':[], 'ODE2_t':[]}
    def SplitOutputData(self, outputData):
        rv = {}
        rv['ODE2_x'] = outputData[0:self.nNodes]
        rv['ODE2_y'] = outputData[self.nNodes:]
        return rv
    
    #convert all output vectors into plottable data (e.g. [time, x, y])
    #the size of data allows to decide how many columns exist
    def OutputData2PlotData(self, outputData, forSolutionViewer=False):
        if forSolutionViewer:
            raise ValueError('NonlinearBeamStatic.OutputData2PlotData: this model is not suitable for SolutionViewer!')
        dataDict = self.SplitOutputData(outputData)
        
        data = np.vstack((self.GetOutputXAxisVector(), dataDict['ODE2_x'].T, dataDict['ODE2_y'].T)).T
        return data

    #return dict of names to columns for plotdata        
    def PlotDataColumns(self):
        return {'x':0, 'ODE2_x':1, 'ODE2_y':2}

    #get compute model with given input data and return output data
    #initialState contains position and velocity states as list of two np.arrays 
    def ComputeModel(self, inputData, hiddenData=None, verboseMode = 0, solutionViewer = False):
        #set input data ...
        inputDict = self.SplitInputData(self.GetInputScaling() * np.array(inputData))
        loadVector = list(inputDict['data'])+[0]
        
        self.mbs.SetLoadParameter(self.lTipLoad, 'loadVector', loadVector)

        self.simulationSettings.staticSolver.verboseMode = verboseMode
        self.simulationSettings.solutionSettings.writeSolutionToFile = solutionViewer 
 
 
        self.mbs.SolveStatic(self.simulationSettings)
        if solutionViewer:
            print('load=',loadVector)
            self.mbs.SolutionViewer()

        #get sensor data and apply piecewise scaling:
        output = 0*self.GetOutputScaling()
        for i, sPos in enumerate(self.listPosSensors):
            p = self.mbs.GetSensorValues(sPos)
            output[i] = p[0] #x
            output[i+self.nNodes] = p[1] #y

        output = self.GetOutputScaling()*output
        return output

    def SolutionViewer(self, outputData, outputDataColumns = [0]):
        print('NonlinearBeamStatic.SolutionViewer: not available')




#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#follows Multibody System Dynamics 2021 paper of Choi et al. "Data-driven simulation for general-purpose multibody dynamics using Deep Neural Networks"
#NOTE: wrong theta0 in paper!
class DoublePendulum(NNtestModel):

    #initialize class 
    def __init__(self, nStepsTotal=100, useInitialVelocities=True, useInitialAngles=True, 
                 addTime=False, endTime=1, nnType='RNN', variationLengths=True):
        NNtestModel.__init__(self)

        #required in base:
        self.nStepsTotal = nStepsTotal
        self.useInitialVelocities = useInitialVelocities
        self.useInitialAngles = useInitialAngles
        self.endTime = endTime
        self.nnType = nnType
        self.variationLengths = variationLengths

        self.smallFact = 1 #0.1    #use small values for inputs to remove tanh nonlinearity
        self.addTime = addTime #add time as input

        self.nODE2 = 2 
        self.m0 = 2 #kg
        self.m1 = 1 #kg
        self.L0 = 1 #m
        self.L1 = 2 #m
        self.gravity = 9.81 #m/s**2
        # self.initAngles = np.array([0.5*pi,0.5*pi]) #initial angle from vertical equilibrium
        self.initAngles = np.array([1.6,2.2]) #initial angle ranges from vertical equilibrium
        self.initAngles_t = 0*np.array([0.1,0.5]) #initial angular velocity ranges

        #default values, if not varied:
        self.phi0 = 1.6 #self.smallFact*0.5*pi #will be set later
        self.phi1 = 2.2 #self.smallFact*0.5*pi #will be set later; WRONG in paper!
        self.phi0_t = 0 #will be set later
        self.phi1_t = 0 #will be set later
        
        self.computationType = ModelComputationType.dynamicImplicit

        self.modelName = 'DoublePendulum'
        self.modelNameShort = 'd-pend'
        
        self.initHidden = True and self.IsRNN() #for RNN

        #for FFN, we can test also with no initial angles; for RNN it is always needed
        #nInit: 0..1=angle0/1, 2..3=vel0/1
        self.nInit = self.nODE2*(int(useInitialVelocities) + int(useInitialAngles))
        if self.initHidden: 
            self.nInit=0
        
        self.scalVelocities = 5.
        #inputscaling is difficult for RNN (hidden states), better not to use
        if self.IsFFN():
            self.inputScaling = np.ones(self.nInit + self.variationLengths*2) 
        else:
            self.inputScaling = np.ones((self.nStepsTotal, 2*int(self.variationLengths) + 
                                         int(self.addTime))) #2 lengths + time


        self.outputScaling = np.ones((self.nStepsTotal, 2*self.nODE2 )) #2*displacements + 2*velocities)
        #if nnType != 'RNN':
        self.outputScaling[:,2] /= self.scalVelocities
        self.outputScaling[:,3] /= self.scalVelocities

        

    def CreateModel(self):
        self.SC = exu.SystemContainer()
        self.mbs = self.SC.AddSystem()
    
        
    
        b = 0.1
        gGround = [GraphicsDataOrthoCubePoint(centerPoint=[0,0.5*b,0],size=[b,b,b], color=color4grey)]
        oGround = self.mbs.AddObject(ObjectGround(visualization=VObjectGround(graphicsData=gGround)) )

        #put double pendulum in 
        refP0 = [0.,-self.L0,0.]
        p0 = np.array([self.L0*sin(self.phi0),-self.L0*cos(self.phi0),0.])
        omega0 = [0.,0.,self.phi0_t]
        v0 = np.cross(omega0, p0)
        o0 = self.mbs.CreateMassPoint(referencePosition=refP0,
                                      initialDisplacement=p0-refP0,
                                      initialVelocity=v0,
                                      physicsMass=self.m0,
                                      gravity=[0.,-self.gravity,0.],
                                      create2D=True,
                                      drawSize = b,color=color4red)

        refP1 = [0.,-self.L0-self.L1,0.]
        p1 = p0 + [self.L1*sin(self.phi1),-self.L1*cos(self.phi1),0.]
        omega1 = [0.,0.,self.phi1_t]
        v1 = np.cross(omega1, p1-p0)+v0
        o1 = self.mbs.CreateMassPoint(referencePosition=refP1,
                                      initialDisplacement=p1-refP1,
                                      initialVelocity=v1,
                                      physicsMass=self.m1,
                                      gravity=[0.,-self.gravity,0.],
                                      create2D=True,
                                      drawSize = b,color=color4dodgerblue)
        #print('p0=',p0, ', p1=', p1)
        
        self.mbs.CreateDistanceConstraint(bodyOrNodeList=[oGround,o0], distance=self.L0)
        self.mbs.CreateDistanceConstraint(bodyOrNodeList=[o0,o1], distance=self.L1)
        
        self.sPos0 = self.mbs.AddSensor(SensorBody(bodyNumber=o0, storeInternal=True,
                                              outputVariableType=exu.OutputVariableType.Position))
        self.sPos1 = self.mbs.AddSensor(SensorBody(bodyNumber=o1, storeInternal=True,
                                              outputVariableType=exu.OutputVariableType.Position))
        self.sVel0 = self.mbs.AddSensor(SensorBody(bodyNumber=o0, storeInternal=True,
                                              outputVariableType=exu.OutputVariableType.Velocity))
        self.sVel1 = self.mbs.AddSensor(SensorBody(bodyNumber=o1, storeInternal=True,
                                              outputVariableType=exu.OutputVariableType.Velocity))
        
        def UFsensor(mbs, t, sensorNumbers, factors, configuration):
            p0 = mbs.GetSensorValues(sensorNumbers[0]) 
            p1 = mbs.GetSensorValues(sensorNumbers[1]) 
            v0 = mbs.GetSensorValues(sensorNumbers[2]) 
            v1 = mbs.GetSensorValues(sensorNumbers[3]) 
            phi0 = atan2(p0[0],-p0[1]) #compute angle; straight down is zero degree
            dp1 = p1-p0 #relative position
            dv1 = v1-v0 #relative velocity
            phi1 = atan2(dp1[0],-dp1[1]) 
            
            nom0 = p0[0]**2+p0[1]**2
            phi0_t = -p0[1]/nom0 * v0[0] + p0[0]/nom0 * v0[1]

            nom1 = dp1[0]**2+dp1[1]**2
            phi1_t = -dp1[1]/nom1 * dv1[0] + dp1[0]/nom1 * dv1[1]
            
            return [phi0,phi1,phi0_t,phi1_t] 

        self.sAngles = self.mbs.AddSensor(SensorUserFunction(sensorNumbers=[self.sPos0,self.sPos1,self.sVel0,self.sVel1], 
                                                             #factors=[self.L0, self.L1],
                                 storeInternal=True,sensorUserFunction=UFsensor))
        
    
        self.timeVecOut = np.arange(1,self.nStepsTotal+1)/self.nStepsTotal*self.endTime
    
        self.mbs.Assemble()

        self.simulationSettings 
        self.simulationSettings = exu.SimulationSettings() #takes currently set values or default values
        self.simulationSettings.solutionSettings.sensorsWritePeriod = self.endTime / self.nStepsTotal
        self.simulationSettings.timeIntegration.numberOfSteps = self.GetNSimulationSteps()
        self.simulationSettings.timeIntegration.endTime = self.endTime

    #get time vector according to output data
    def GetOutputXAxisVector(self):
        return self.timeVecOut

    #create a randomized input vector
    #relCnt can be used to create different kinds of input vectors (sinoid, noise, ...)
    #isTest is True in case of test data creation
    def CreateInputVector(self, relCnt = 0, isTest=False):

        vec = np.zeros(self.GetInputScaling().shape)
        #print(vec.shape)
        lengths = []
        if self.variationLengths:
            lengths = np.zeros(2)

            lengths[0] = 1.+1.*np.random.rand() #L0
            lengths[1] = 2.+1.*np.random.rand() #L1

        randAngles = self.smallFact*(self.initAngles*(np.random.rand(2)))
        randVels = self.smallFact*self.initAngles_t*np.random.rand(2)

        #for RNN, we can also avoid variations:
        if not self.useInitialAngles:
            randAngles = self.initAngles
        if not self.useInitialVelocities:
            randVels = [0.,0.]

        if not self.IsFFN():
            for i, length in enumerate(lengths):
                vec[:,i] = length
            if self.addTime:
                vec[:,0+len(lengths)] = self.timeVecOut
        else:
            off = 0
            if self.useInitialAngles:
                vec[0] = randAngles[0]
                vec[1] = randAngles[1]
                off += 2
            if self.useInitialVelocities:
                vec[0+off] = randVels[0]
                vec[1+off] = randVels[1]
                off += 2
    
            for i, length in enumerate(lengths):
                vec[off+i] = length
            off += len(lengths)

            # if self.addTime:
            #     vec[off:] = self.timeVecOut
            
        return vec

    #create initialization of (couple of first) hidden states (RNN)
    def CreateHiddenInit(self, isTest):
        if self.initHidden:
            vec = np.zeros(2*self.nODE2)
            randAngles = self.smallFact*self.initAngles * (2*np.random.rand(self.nODE2)-1.) 
            randVels = self.smallFact*self.initAngles_t*np.random.rand(self.nODE2)
            
            vec[0:2] = randAngles
            vec[2:4] = randVels

            return vec
        else:
            return np.array([])
            
    #get number of simulation steps
    def GetNSimulationSteps(self):
        return self.nStepsTotal*10 #10 x finer simulation than output

    #split input data into initial values, forces or other inputs
    #return dict with 'data' and possibly 'initialODE2' and 'initialODE2_t'
    def SplitInputData(self, inputData, hiddenData=None):
        data = np.array(self.GetInputScaling()*inputData)
        rv = {}
        if not self.IsFFN():
            if self.initHidden: #always true for RNN
                rv['phi0'] = hiddenData[0]  
                rv['phi1'] = hiddenData[1]  
                rv['phi0_t'] = hiddenData[2]
                rv['phi1_t'] = hiddenData[3]
    
            if self.variationLengths:
                rv['L0'] = data[0,0] #lengths are the same for all sequences
                rv['L1'] = data[0,1] #lengths are the same for all sequences
            if self.addTime: #not needed ...
                rv['time'] = data[:,2*int(self.variationLengths)] #lengths are the same for all sequences
                
        else:
            off = 0
            #default values, if not otherwise set
            if self.useInitialAngles:
                rv['phi0'] = data[0]
                rv['phi1'] = data[1]
                off+=2
            if self.useInitialVelocities:
                rv['phi0_t'] = data[off+0]
                rv['phi1_t'] = data[off+1]
                off+=2
    
            if self.variationLengths:
                rv['L0'] = data[off+0] 
                rv['L1'] = data[off+1] 
                off+=2

            # if self.addTime:
            #     rv['time'] = data[off:] #lengths are the same for all sequences
        
        return rv

    #split output data to get ODE2 values (and possibly other data, such as ODE2)
    #return dict {'ODE2':[], 'ODE2_t':[]}
    def SplitOutputData(self, outputData):
        rv = {}
        data = outputData
        if outputData.ndim == 1:
            data = outputData.reshape((self.nStepsTotal,4))
        rv['phi0'] = data[:,0]
        rv['phi1'] = data[:,1]
        rv['phi0_t'] = data[:,2]
        rv['phi1_t'] = data[:,3]
        return rv
    
    #convert all output vectors into plottable data (e.g. [time, x, y])
    #the size of data allows to decide how many columns exist
    def OutputData2PlotData(self, outputData, forSolutionViewer=False):
        timeVec = self.GetOutputXAxisVector()
        dataDict = self.SplitOutputData(outputData)
        
        data = np.vstack((timeVec, dataDict['phi0'].T, dataDict['phi1'].T,
                          dataDict['phi0_t'].T, dataDict['phi1_t'].T)).T
            
        return data

    #return dict of names to columns for plotdata        
    def PlotDataColumns(self):
        d = {'time':0, 'phi0':1, 'phi1':2, 'phi0_t':3, 'phi1_t':4}
        
        return d

    #get compute model with given input data and return output data
    #initialState contains position and velocity states as list of two np.arrays 
    def ComputeModel(self, inputData, hiddenData=None, verboseMode = 0, solutionViewer = False):
        #set input data ...
        inputDict = self.SplitInputData(np.array(inputData), hiddenData)

        #print('hiddenData=', hiddenData)
        if 'L0' in inputDict:
            self.L0 = inputDict['L0']
            self.L1 = inputDict['L1']
        
        if 'phi0' in inputDict:
            self.phi0 = inputDict['phi0']
            self.phi1 = inputDict['phi1']
        if 'phi0_t' in inputDict:
            self.phi0_t = inputDict['phi0_t']
            self.phi1_t = inputDict['phi1_t']
                    
        self.CreateModel() #must be created newly for each test ...


        self.simulationSettings.timeIntegration.verboseMode = verboseMode
        self.simulationSettings.solutionSettings.writeSolutionToFile = solutionViewer 
 
        self.mbs.SolveDynamic(self.simulationSettings) #GeneralizedAlpha

        if solutionViewer:
            self.mbs.SolutionViewer()

        #get sensor data and apply piecewise scaling:
        output = 0*self.GetOutputScaling()
        
        for i in range(4): #2 x phi and 2 x phi_t 
            #sensordata includes time
            #exclude t=0
            output[:,i] = self.mbs.GetSensorStoredData(self.sAngles)[1:,1+i] 
        
        output = self.GetOutputScaling()*output
        return output

    def SolutionViewer(self, outputData, outputDataColumns = [0]):
        #model is 2D!
        #NOTE: L0 and L1 may be wrong! this is just for visualization!
        nColumns = 2*2 #2 x (x,y)
        angles = self.OutputData2PlotData(outputData, forSolutionViewer=True)
        #print(angles)
        data = np.zeros((self.nStepsTotal, 1+nColumns))
        for i, t in enumerate(self.timeVecOut):
            data[i,0] = t
            data[i,1] = self.L0*sin(angles[i,1])
            data[i,2] = +self.L0*(1-cos(angles[i,1]))
            data[i,3] = data[i,1]+self.L1*sin(angles[i,2])
            data[i,4] = data[i,2]-self.L1*cos(angles[i,2])+self.L1
        
        # print(data)
        
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
    
    #%%
    #reference solution for double pendulum with scipy:
    #NOTE theta1 is relative angle!
    import numpy as np
    from scipy.integrate import odeint
    import matplotlib.pyplot as plt
    
    # Constants
    g = 9.81  # gravity
    L0, L1 = inputData[4], inputData[5]  # lengths of arms
    m0, m1 = 2.0, 1.0  # masses
    theta0, theta1 = inputData[0], inputData[1]  # initial angles
    v0, v1 = inputData[2], inputData[3]  # initial angular velocities
    
    # System of differential equations
    def equations(y, t, L0, L1, m0, m1):
        theta0, z0, theta1, z1 = y
        
        c, s = np.cos(theta0-theta1), np.sin(theta0-theta1)
        theta0_dot = z0
        z0_dot = (m1*g*np.sin(theta1)*c - m1*s*(L0*z0**2*c + L1*z1**2) - (m0+m1)*g*np.sin(theta0)) / L0 / (m0 + m1*s**2)
        theta1_dot = z1
        z1_dot = ((m0+m1)*(L0*z0**2*s - g*np.sin(theta1) + g*np.sin(theta0)*c) + m1*L1*z1**2*s*c) / L1 / (m0 + m1*s**2)
        return theta0_dot, z0_dot, theta1_dot, z1_dot
    
    # Initial conditions: theta0, dtheta0/dt, theta1, dtheta1/dt.
    y0 = [theta0, v0, theta1, v1]
    
    # Time array for solution
    t = np.linspace(0, 5, 1000)
    
    # Solve ODE
    solution = odeint(equations, y0, t, args=(L0, L1, m0, m1))
    scipySol = solution[-1,:]
    print('scipy=\n',scipySol[0],scipySol[2],scipySol[1],scipySol[3])
    # Plot
    if False:
        plt.figure(figsize=(10, 4))
        plt.plot(t, solution[:, 0], label="theta0(t)")
        plt.plot(t, solution[:, 2], label="theta1(t)")
        plt.plot(t, solution[:, 1], label="theta0_t(t)")
        plt.plot(t, solution[:, 3], label="theta1_t(t)")
        plt.legend(loc='best')
        plt.xlabel('Time')
        plt.grid()
        plt.show()


