#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Generate training data for hydraulically actuated 3D FFRF reduced order model with 2 flexible bodies.
#           It includes lift boom (1 DOF) and Patu crane (2-DOF) systems.
#           For closed loop configurations, coordinate partioning method is used.             
#
# Author:   Qasim Khadim,Johannes Gerstmayr
# Date:     2025-06-23
#
# Copyright:This file is part of Exudyn. Exudyn is free software. You can redistribute 
#it and/or modify it under the terms of the Exudyn license. See 'LICENSE.txt' for more details.
#+++++++++++++++++++++++++++0++++++++++++++++++++++++++++++++++++++++++++++++++

from models.container import *
from models.compute_SLIDE import *

class ExudynFlexible():

    #initialize class 
    def __init__(self,parent=None, nStepsTotal=100, endTime=0.5, 
                 nModes = 2,loadFromSavedNPY=True, mL= 50, material='Steel', 
                 visualization = False, dictSensors=None, verboseMode = 0):
       
        self.parent             = parent
        self.mL                 = mL
        self.loadFromSavedNPY   = loadFromSavedNPY
        self.Material           = material
        self.StaticCase         = False
        self.Visualization      = visualization
        self.nStepsTotal        = nStepsTotal
        self.endTime            = endTime
        self.TimeStep           = self.endTime / (self.nStepsTotal) 
        self.verboseMode        = False    
        self.nModes             = nModes
        self.dictSensors        = dictSensors
    
    #get number of simulation steps
    def GetNSimulationSteps(self):
        return self.nStepsTotal # x finer simulation than output
    
    # Materials are defined.   
    def Materials (self):
        if self.Material=='Steel':
             Emodulus,nu,rho     = 2.1e11,0.3, 7850
             filePath            = 'AbaqusMesh/LiftBoom/Steel/Job-1'
             filePath0           = 'AbaqusMesh/LiftBoom/Steel'
             filePath2           = 'AbaqusMesh/TiltBoom/Steel/Job-1'
             filePath20          = 'AbaqusMesh/TiltBoom/Steel'
             
        elif self.Material=='Aluminium':
                 Emodulus,nu,rho     = 6.9e10,0.33, 2700
                 filePath            = 'AbaqusMesh/LiftBoom/Aluminium/Job-1'
                 filePath0           = 'AbaqusMesh/LiftBoom/Aluminium'
                 filePath2           = 'AbaqusMesh/TiltBoom/Aluminium/Job-1'
                 filePath20          = 'AbaqusMesh/TiltBoom/Aluminium'
                 
        elif self.Material=='Titanium':
                Emodulus,nu,rho     = 1.15e11,0.33, 4730
                filePath            = 'AbaqusMesh/LiftBoom/Titanium/Job-1'
                filePath0           = 'AbaqusMesh/LiftBoom/Titanium'
                filePath2           = 'AbaqusMesh/TiltBoom/Titanium/Job-1'
                filePath20          = 'AbaqusMesh/TiltBoom/Titanium'
                 
        elif self.Material=='Composites': #: Graphene-reinforced Aluminum Matrix Composite (Gr-Al MMC)
                Emodulus,nu,rho     = 1530,0.34, 1.398e+11
                filePath            = 'AbaqusMesh/LiftBoom/Composites/Job-1'
                filePath0           = 'AbaqusMesh/LiftBoom/Composites'
                filePath2           = 'AbaqusMesh/TiltBoom/Composites/Job-1'
                filePath20          = 'AbaqusMesh/TiltBoom/Composites'
        else:
            print('Not supported material yet..')
        
        self.filePath   = filePath
        self.filePath0  = filePath0
        self.filePath2  = filePath2
        self.filePath20 = filePath20
        self.mat        = KirchhoffMaterial(Emodulus, nu, rho)
        self.varType    = exu.OutputVariableType.StrainLocal
        return 

    
    # Compute Stresses
    def UFStressData(self, mbs, t, sensorNumbers, factors, configuration):    
           val = mbs.GetSensorValues(sensorNumbers[0])
           StressVec = self.mat.StrainVector2StressVector(val)
           return StressVec

   # Friction model 1
    def CylinderFriction1(self, mbs, t, itemNumber, u, v, k, d, F0):
     return 1*StribeckFunction(v, muDynamic=1, muStaticOffset=1.5, regVel=1e-4)+(k*(u) + d*v + k*(u)**3-F0)

   #Friction model 2
    def CylinderFriction2(mbs, t, itemIndex, u, v, k, d, f0): 
     return   1*(Fc*tanh(4*(abs(v    )/vs))+(Fs-Fc)*((abs(v    )/vs)/((1/4)*(abs(v    )/vs)**2+3/4)**2))*np.sign(v )+sig2*v    *tanh(4)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
                            # --Lift Boom---
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
    def LiftBoom(self,mbs,SC, theta1, p1, p2):
        self.Materials()
        
        self.mbs                  = mbs
        self.SC                   = SC
        feL                       = FEMinterface()
        self.theta1               = theta1
        self.p1                   = p1
        self.p2                   = p2                         
        self.dictSensors          = {}
        self.StaticInitialization = True
   
        #Ground body
        oGround                   = self.mbs.AddObject(ObjectGround(referencePosition=[0,0,0],visualization=VObjectGround(graphicsData=[plane])))
        markerGround              = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=oGround, localPosition=[0, 0, 0]))
        iCube1                    = RigidBodyInertia(mass=m1, com=pMid1,inertiaTensor=Inertia1,inertiaTensorAtCOM=True)
        graphicsCOM1              = GraphicsDataBasis(origin=iCube1.com, length=2*L1)
        
        # Definintion of pillar as body in Exudyn and node n1
        [n1, b1]                  = AddRigidBody(mainSys=self.mbs,inertia=iCube1,nodeType=exu.NodeType.RotationEulerParameters,#graphicsDataList=[graphicsCOM1, graphicsBody1],
                                                    position=PillarP,rotationMatrix=np.diag([1, 1, 1]),gravity=[0, -9.8066, 0])
        Marker3                   = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b1, localPosition=Mark3))                     #With Ground
        Marker4                   = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b1, localPosition=Mark4))            #Lift Boom
        Marker5                   = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b1, localPosition=Mark5))        # Cylinder 1 position
        
        # Fixed joint between Pillar and Ground
        self.mbs.AddObject(GenericJoint(markerNumbers=[markerGround, Marker3],constrainedAxes=[1, 1, 1, 1, 1, 1], visualization=VObjectJointGeneric(axesRadius=0.2*W1,axesLength=1.4*W1)))

        if not self.loadFromSavedNPY: 
            start_time      = time.time()
            nodes1          = feL.ImportFromAbaqusInputFile(self.filePath+'.inp', typeName='Part', name='Job-1')
            feL.ReadMassMatrixFromAbaqus(fileName=self.filePath + '_MASS2.mtx')             #Load mass matrix
            feL.ReadStiffnessMatrixFromAbaqus(fileName=self.filePath + '_STIF2.mtx')        #Load stiffness matrix
            feL.SaveToFile(self.filePath,mode='PKL')
            if self.verboseMode:
                    print("--- saving LiftBoom FEM Abaqus data took: %s seconds ---" % (time.time() - start_time)) 
        else:       
            if self.verboseMode:
                print('importing Abaqus FEM data structure of Lift Boom...')
            start_time = time.time()
            feL.LoadFromFile(self.filePath,mode='PKL')
            cpuTime = time.time() - start_time
            if self.verboseMode:
                print("--- importing FEM data took: %s seconds ---" % (cpuTime))
                    
        p2                  = [0, 0,-100*1e-3]
        p1                  = [0, 0, 100*1e-3]
        radius1             = 25*1e-3
        nodeListJoint1      = feL.GetNodesOnCylinder(p1, p2, radius1, tolerance=1e-4) 
        pJoint1             = feL.GetNodePositionsMean(nodeListJoint1)
        nodeListJoint1Len   = len(nodeListJoint1)
        noodeWeightsJoint1  = [1/nodeListJoint1Len]*nodeListJoint1Len
        noodeWeightsJoint1  =feL.GetNodeWeightsFromSurfaceAreas(nodeListJoint1)
        
        p4                  = [304.19*1e-3,-100.01*1e-3,-100*1e-3]
        p3                  = [304.19*1e-3,-100.01*1e-3, 100*1e-3]
        radius2             = 36*1e-3
        nodeListPist1       = feL.GetNodesOnCylinder(p3, p4, radius2, tolerance=1e-2)  
        pJoint2             = feL.GetNodePositionsMean(nodeListPist1)
        nodeListPist1Len    = len(nodeListPist1)
        noodeWeightsPist1   = [1/nodeListPist1Len]*nodeListPist1Len
            
        if self.mL != 0:
            p10                 = [2875*1e-3,15.15*1e-3,    74*1e-3]
            p9                  = [2875*1e-3,15.15*1e-3,   -74*1e-3]
            radius5             = 46*1e-3
            nodeListJoint3      = feL.GetNodesOnCylinder(p9, p10, radius5, tolerance=1e-4)  
            pJoint5             = feL.GetNodePositionsMean(nodeListJoint3)
            nodeListJoint3Len   = len(nodeListJoint3)
            noodeWeightsJoint3  = [1/nodeListJoint3Len]*nodeListJoint3Len

            # STEP 2: Craig-Bampton Modes
            boundaryList        = [nodeListJoint1, nodeListPist1, nodeListJoint3]    
        else: 
             # STEP 2: Craig-Bampton Modes
             boundaryList        = [nodeListJoint1, nodeListPist1]

        start_time          = time.time()
        if self.loadFromSavedNPY:
           if self.mL != 0:
               feL.LoadFromFile(f"{self.filePath0}/feL_Load_Modes{self.nModes}", mode='PKL')
           else:
              feL.LoadFromFile(f"{self.filePath0}/feL_Modes{self.nModes}", mode='PKL')

        else:
         feL.ComputeHurtyCraigBamptonModes(boundaryNodesList=boundaryList, nEigenModes=self.nModes, useSparseSolver=True,computationMode = HCBstaticModeSelection.RBE2) 
         
         print("ComputePostProcessingModes ... (may take a while)")
         feL.ComputePostProcessingModes(material=self.mat,outputVariableType=self.varType,)
        
         if self.mL != 0:  
             feL.SaveToFile(f"{self.filePath0}/feL_Load_Modes{self.nModes}", mode='PKL')
         else:
             feL.SaveToFile(f"{self.filePath0}/feL_Modes{self.nModes}", mode='PKL')
             
         if self.verboseMode:
           print("Hurty-Craig Bampton modes... ")
           print("eigen freq.=", feL.GetEigenFrequenciesHz())
           print("HCB modes needed %.3f seconds" % (time.time() - start_time))  

        LiftBoom            = ObjectFFRFreducedOrderInterface(feL)
        LiftBoomFFRF        = LiftBoom.AddObjectFFRFreducedOrder(self.mbs, positionRef=np.array(Mark4), 
                                          initialVelocity=[0,0,0], 
                                          initialAngularVelocity=[0,0,0],
                                          rotationMatrixRef  = RotationMatrixZ(mt.radians(self.theta1)),
                                          gravity= [0, -9.8066, 0], #massProportionalDamping = 0, stiffnessProportionalDamping = 1e-5 ,
                                         massProportionalDamping = 0, stiffnessProportionalDamping = 3.35e-3 ,color=color4blue,)
        self.mbs.SetObjectParameter(objectNumber=LiftBoomFFRF['oFFRFreducedOrder'],parameterName='outputVariableTypeModeBasis',value=self.varType)
        Marker7             = self.mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=LiftBoomFFRF['oFFRFreducedOrder'],
                                          meshNodeNumbers=np.array(nodeListJoint1), #these are the meshNodeNumbers
                                          weightingFactors=noodeWeightsJoint1))
        Marker8             = self.mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=LiftBoomFFRF['oFFRFreducedOrder'],
                                          meshNodeNumbers=np.array(nodeListPist1), #these are the meshNodeNumbers
                                          weightingFactors=noodeWeightsPist1))
        
        #Revolute Joint
        self.mbs.AddObject(GenericJoint(markerNumbers=[Marker4, Marker7],constrainedAxes=[1,1,1,1,1,0],visualization=VObjectJointGeneric(axesRadius=0.18*0.263342,axesLength=1.1*0.263342)))
        # Add load 
        if self.mL != 0:
            Marker9             = self.mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=LiftBoomFFRF['oFFRFreducedOrder'],
                                              meshNodeNumbers=np.array(nodeListJoint3), #these are the meshNodeNumbers
                                              weightingFactors=noodeWeightsJoint3))
            
            pos = self.mbs.GetMarkerOutput(Marker9, variableType=exu.OutputVariableType.Position, 
                                           configuration=exu.ConfigurationType.Reference)
            #print('pos=', pos)
            bMass = self.mbs.CreateMassPoint(physicsMass=self.mL, referencePosition=pos, show=True, gravity=[0, -9.8066, 0],
                                     graphicsDataList=[GraphicsDataSphere(radius=0.1, color=color4red)])
            mMass = self.mbs.AddMarker(MarkerBodyPosition(bodyNumber=bMass))
            self.mbs.AddObject(SphericalJoint(markerNumbers=[Marker9, mMass], visualization=VSphericalJoint(show=False)))

        #ODE1 for pressures:
        nODE1            = self.mbs.AddNode(NodeGenericODE1(referenceCoordinates=[0,0], initialCoordinates=[self.p1,self.p2], numberOfODE1Coordinates=2))
        oFriction1       = self.mbs.AddObject(ObjectConnectorSpringDamper(markerNumbers=[Marker5, Marker8], referenceLength=0.001,stiffness=2000,
                                                            damping=1e4, force=0, velocityOffset = 0., activeConnector = True,springForceUserFunction=self.CylinderFriction1,visualization=VSpringDamper(show=False) ))
        
        oHA1 = None
        if True:
            oHA1                = self.mbs.AddObject(HydraulicActuatorSimple(name='LiftCylinder', markerNumbers=[ Marker5, Marker8], 
                                                    nodeNumbers=[nODE1], offsetLength=L_Cyl1, strokeLength=L_Pis1, chamberCrossSection0=A[0], 
                                                    chamberCrossSection1=A[1], hoseVolume0=V1, hoseVolume1=V2, valveOpening0=0, 
                                                    valveOpening1=0, actuatorDamping=6.40e5, oilBulkModulus=Bo, cylinderBulkModulus=Bc, 
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


        TipNode     = feL.GetNodeAtPoint(np.array([2.829, 0.0151500003,  0.074000001]))
        StressNode  = feL.GetNodeAtPoint(np.array([0.639392078,  0.110807151, 0.0799999982]))
        
        if self.verboseMode:
            print("nMid=",nMid)
            print("nMid=",MarkerTip)

  
         
        theta1       = self.mbs.AddSensor(SensorNode( nodeNumber=LiftBoomFFRF['nRigidBody'], storeInternal=True, outputVariableType=exu.OutputVariableType.Rotation))
        dtheta1      = self.mbs.AddSensor(SensorNode( nodeNumber=LiftBoomFFRF['nRigidBody'], storeInternal=True, outputVariableType=exu.OutputVariableType.AngularVelocity))
        ddtheta1     = self.mbs.AddSensor(SensorNode( nodeNumber=LiftBoomFFRF['nRigidBody'], storeInternal=True, outputVariableType=exu.OutputVariableType.AngularAcceleration))
               
        self.dictSensors['theta1']    = theta1 
        self.dictSensors['dtheta1']   = dtheta1 
        self.dictSensors['ddtheta1']  = ddtheta1 
         
        # Add Sensor for deflection
        deltaY  = self.mbs.AddSensor(SensorSuperElement(bodyNumber=LiftBoomFFRF['oFFRFreducedOrder'], meshNodeNumber=TipNode, storeInternal=True, outputVariableType=exu.OutputVariableType.DisplacementLocal ))
        eps1    = self.mbs.AddSensor(SensorSuperElement(bodyNumber=LiftBoomFFRF['oFFRFreducedOrder'], meshNodeNumber=TipNode, storeInternal=True, outputVariableType=self.varType ))
        eps2    = self.mbs.AddSensor(SensorSuperElement(bodyNumber=LiftBoomFFRF['oFFRFreducedOrder'], meshNodeNumber=StressNode, storeInternal=True, outputVariableType=self.varType))

        self.dictSensors['deltaY']   = deltaY
        self.dictSensors['eps1']     = eps1
        self.dictSensors['sig1']     = self.mbs.AddSensor(SensorUserFunction(sensorNumbers=[eps2], storeInternal=True, sensorUserFunction=self.UFStressData))
           
        if oHA1 != None:
            sForce1          = self.mbs.AddSensor(SensorObject(objectNumber=oHA1, storeInternal=True, outputVariableType=exu.OutputVariableType.Force))
            self.dictSensors['sForce1']=sForce1
            sDistance1       = self.mbs.AddSensor(SensorObject(objectNumber=oHA1, storeInternal=True, outputVariableType=exu.OutputVariableType.Distance))
            self.dictSensors['sDistance1']=sDistance1
    
            sVelocity1       = self.mbs.AddSensor(SensorObject(objectNumber=oHA1, storeInternal=True, outputVariableType=exu.OutputVariableType.VelocityLocal))
            self.dictSensors['sVelocity1']=sVelocity1
            sPressures1      = self.mbs.AddSensor(SensorNode(nodeNumber=nODE1, storeInternal=True,outputVariableType=exu.OutputVariableType.Coordinates))   
            self.dictSensors['sPressures1']=sPressures1
           
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
        self.SC.visualizationSettings.contour.outputVariable = self.varType
        
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
            self.simulationSettings.staticSolver.numberOfLoadSteps = 10
            self.simulationSettings.staticSolver.constrainODE1coordinates = True #constrain pressures to initial values
        
            # exu.SuppressWarnings(True)
            exu.config.suppressWarnings = True
            self.mbs.SolveStatic(self.simulationSettings, updateInitialValues=True) 
            exu.config.suppressWarnings = False
            
            
            t_dVec, eVal = EOM_SLIDE(self, A_d, nValues=5)
            
            n_etd   = round(np.max(t_dVec)/self.TimeStep)
            self.dictSensors['n_etd']    = n_etd 
            
            #print(self.n_etd)
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
        
        return self.dictSensors
    


#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
                            # --PATU CRANE---
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            
    def PatuCrane(self,mbs, SC, theta1,theta2, p1, p2,p3, p4):
       self.Materials()
       self.theta1                  = theta1
       self.theta2                  = theta2-self.theta1
       self.p1                      = p1
       self.p2                      = p2    
       self.p3                      = p3
       self.p4                      = p4
       self.mbs                     = mbs
       self.SC                      = SC
       feL                          = FEMinterface()
       feT                          = FEMinterface()

       self.dictSensors             = {}
       self.StaticInitialization    = True
   
       #Ground body
       oGround                     = self.mbs.AddObject(ObjectGround(referencePosition=[0,0,0],visualization=VObjectGround(graphicsData=[plane])))
       markerGround                = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=oGround, localPosition=[0, 0, 0]))
       iCube1                      = RigidBodyInertia(mass=m1, com=pMid1,inertiaTensor=Inertia1,inertiaTensorAtCOM=True)
       graphicsCOM1                = GraphicsDataBasis(origin=iCube1.com, length=2*L1)
       
       # Definintion of pillar as body in Exudyn and node n1
       [n1, b1]                     = AddRigidBody(mainSys=self.mbs,inertia=iCube1,nodeType=exu.NodeType.RotationEulerParameters,#graphicsDataList=[graphicsCOM1, graphicsBody1],
                                                   position=PillarP,rotationMatrix=np.diag([1, 1, 1]),gravity=[0, -9.8066, 0])
       Marker3                      = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b1, localPosition=Mark3))                     #With Ground
       Marker4                      = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b1, localPosition=Mark4))            #Lift Boom
       Marker5                      = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b1, localPosition=Mark5))        # Cylinder 1 position
       
       # Fixed joint between Pillar and Ground
       self.mbs.AddObject(GenericJoint(markerNumbers=[markerGround, Marker3],constrainedAxes=[1, 1, 1, 1, 1, 1], visualization=VObjectJointGeneric(axesRadius=0.2*W1,axesLength=1.4*W1)))
       
       if not self.loadFromSavedNPY: 
          start_time      = time.time()   
          nodes1                    = feL.ImportFromAbaqusInputFile(self.filePath+'.inp', typeName='Part', name='Job-1')
          nodes2                    = feT.ImportFromAbaqusInputFile(self.filePath2+'.inp', typeName='Part', name='Job-2')
          feL.ReadMassMatrixFromAbaqus(fileName=self.filePath + '_MASS2.mtx')             #Load mass matrix
          feL.ReadStiffnessMatrixFromAbaqus(fileName=self.filePath + '_STIF2.mtx')        #Load stiffness matrix
          feL.SaveToFile(self.filePath,mode='PKL')
          feT.ReadMassMatrixFromAbaqus(fileName=self.filePath2 + '_MASS2.mtx')             #Load mass matrix
          feT.ReadStiffnessMatrixFromAbaqus(fileName=self.filePath2 + '_STIF2.mtx')        #Load stiffness matrix
          feT.SaveToFile(self.filePath2,mode='PKL')
          if self.verboseMode:
                  print("--- saving LiftBoom FEM Abaqus data took: %s seconds ---" % (time.time() - start_time)) 
       else:       
          if self.verboseMode:
              print('importing Abaqus FEM data structure of Lift Boom...')
          start_time = time.time()
          feL.LoadFromFile(self.filePath,mode='PKL')
          feT.LoadFromFile(self.filePath2,mode='PKL')
          cpuTime = time.time() - start_time
          if self.verboseMode:
              print("--- importing FEM data took: %s seconds ---" % (cpuTime))
              
       p2                  = [0, 0,-100*1e-3]
       p1                  = [0, 0, 100*1e-3]
       radius1             = 25*1e-3
       nodeListJoint1      = feL.GetNodesOnCylinder(p1, p2, radius1, tolerance=1e-4) 
       pJoint1             = feL.GetNodePositionsMean(nodeListJoint1)
       nodeListJoint1Len   = len(nodeListJoint1)
       noodeWeightsJoint1  = [1/nodeListJoint1Len]*nodeListJoint1Len
       noodeWeightsJoint1  =feL.GetNodeWeightsFromSurfaceAreas(nodeListJoint1)
       
       p4                  = [304.19*1e-3,-100.01*1e-3,-100*1e-3]
       p3                  = [304.19*1e-3,-100.01*1e-3, 100*1e-3]
       radius2             = 36*1e-3
       nodeListPist1       = feL.GetNodesOnCylinder(p3, p4, radius2, tolerance=1e-4)  
       pJoint2             = feL.GetNodePositionsMean(nodeListPist1)
       nodeListPist1Len    = len(nodeListPist1)
       noodeWeightsPist1   = [1/nodeListPist1Len]*nodeListPist1Len
       
       # Boundary condition at cylinder 1
       p6                  = [1258e-3,194.59e-3,  65.701e-3]
       p5                  = [1258e-3,194.59e-3, -65.701e-3]
       radius3             = 32e-3
       nodeListCyl2        = feL.GetNodesOnCylinder(p5, p6, radius3, tolerance=1e-2)  
       pJoint3             = feL.GetNodePositionsMean(nodeListCyl2)
       nodeListCyl2Len     = len(nodeListCyl2)
       noodeWeightsCyl2    = [1/nodeListCyl2Len]*nodeListCyl2Len 
                   
       # Boundary condition at Joint 2
       p8                  = [2685e-3,0.15e-03,  74e-3]
       p7                  = [2685e-3,0.15e-03, -74e-3]
       radius4             = 32e-3
       nodeListJoint2      = feL.GetNodesOnCylinder(p7, p8, radius4, tolerance=1e-4)  
       pJoint4             = feL.GetNodePositionsMean(nodeListJoint2)
       nodeListJoint2Len   = len(nodeListJoint2)
       noodeWeightsJoint2  = [1/nodeListJoint2Len]*nodeListJoint2Len
               
       # Joint 3
       p10                 = [2875e-3,15.15e-3,    74e-3]
       p9                  = [2875e-3,15.15e-3,   -74e-3]
       radius5             = 4.60e-002
       nodeListJoint3      = feL.GetNodesOnCylinder(p9, p10, radius5, tolerance=1e-4)  
       pJoint5             = feL.GetNodePositionsMean(nodeListJoint3)
       nodeListJoint3Len   = len(nodeListJoint3)
       noodeWeightsJoint3  = [1/nodeListJoint3Len]*nodeListJoint3Len
               
       # Boundary condition at pillar
       p12                 = [0, 0,  88e-3]
       p11                 = [0, 0, -88e-3]
       radius6             = 48e-3
       nodeListJoint1T     = feT.GetNodesOnCylinder(p11, p12, radius6, tolerance=1e-4) 
       pJoint1T            = feT.GetNodePositionsMean(nodeListJoint1T)
       nodeListJoint1TLen  = len(nodeListJoint1T)
       noodeWeightsJoint1T = [1/nodeListJoint1TLen]*nodeListJoint1TLen
                   
       # Boundary condition at Piston 1
       p14                 = [-95e-3,243.2e-3,  55.511e-3]
       p13                 = [-95e-3,243.2e-3, -55.511e-3]
       radius7             = 26e-3
       nodeListPist1T      = feT.GetNodesOnCylinder(p13, p14, radius7, tolerance=1e-4)  
       pJoint2T            = feT.GetNodePositionsMean(nodeListPist1T)
       nodeListPist1TLen   = len(nodeListPist1T)
       noodeWeightsPist1T  = [1/nodeListPist1TLen]*nodeListPist1TLen

        # Boundary condition at extension boom
       p16                 = [-415e-3,287e-3, 48.011e-3]
       p15                 = [-415e-3,287e-3, -48.011e-3]
       radius8             = 2.3e-002
       nodeListExtT        = feT.GetNodesOnCylinder(p15, p16, radius8, tolerance=1e-4)  
       pExtT               = feT.GetNodePositionsMean(nodeListExtT)
       nodeListExTLen      = len(nodeListExtT)
       noodeWeightsExt1T   = [1/nodeListExTLen]*nodeListExTLen
           
       # STEP 2: Craig-Bampton Modes
       boundaryListL       = [nodeListJoint1,nodeListPist1, nodeListJoint2,  nodeListJoint3] 
       boundaryListT       = [nodeListJoint1T, nodeListPist1T,nodeListExtT]
       StressNode1         = feL.GetNodeAtPoint(np.array([0.800971925, -0.0428495929,  0.074000001]))
       StressNode2         = feT.GetNodeAtPoint(np.array([0.493444443,  0.347000003, 0.0390110984]))
       LiftTip             = feL.GetNodeAtPoint(np.array([2.829, 0.0151500003,  0.074000001]))
       TiltTip             = feT.GetNodeAtPoint(np.array([1.80499995,  0.347000003, 0.0390110984]))
   
       start_time          = time.time()
       
       if self.loadFromSavedNPY:
               feL.LoadFromFile(f"{self.filePath0}/feL_Modes{self.nModes}", mode='PKL')
               feT.LoadFromFile(f"{self.filePath20}/feT_Modes{self.nModes}", mode='PKL')
       else:
         feL.ComputeHurtyCraigBamptonModes(boundaryNodesList=boundaryListL, nEigenModes=self.nModes, useSparseSolver=True,computationMode = HCBstaticModeSelection.RBE2) 
         feT.ComputeHurtyCraigBamptonModes(boundaryNodesList=boundaryListT, nEigenModes=self.nModes, useSparseSolver=True,computationMode = HCBstaticModeSelection.RBE2) 
         print("ComputePostProcessingModes ... (may take a while)")
         feL.ComputePostProcessingModes(material=self.mat,outputVariableType=self.varType,)
         feT.ComputePostProcessingModes(material=self.mat,outputVariableType=self.varType,)
         
         feL.SaveToFile(f"{self.filePath0}/feL_Modes{self.nModes}", mode='PKL')
         feT.SaveToFile(f"{self.filePath20}/feT_Modes{self.nModes}", mode='PKL')
       if self.verboseMode:
           print("Hurty-Craig Bampton modes... ")
           print("eigen freq.=", feL.GetEigenFrequenciesHz())
           print("eigen freq.=", feT.GetEigenFrequenciesHz())
           print("HCB modes needed %.3f seconds" % (time.time() - start_time))  

       LiftBoom            = ObjectFFRFreducedOrderInterface(feL)
       TiltBoom            = ObjectFFRFreducedOrderInterface(feT)
               
       LiftBoomFFRF        = LiftBoom.AddObjectFFRFreducedOrder(self.mbs, positionRef=np.array(Mark4), initialVelocity=[0,0,0], 
                                        initialAngularVelocity=[0,0,0],rotationMatrixRef  = RotationMatrixZ(mt.radians(self.theta1)),gravity= [0, -9.8066, 0],
                                        #massProportionalDamping = 0, stiffnessProportionalDamping = 1e-5 ,
                                       massProportionalDamping = 0, stiffnessProportionalDamping = 3.35e-3 ,color=color4blue,)
       self.mbs.SetObjectParameter(objectNumber=LiftBoomFFRF['oFFRFreducedOrder'],parameterName='outputVariableTypeModeBasis',value=self.varType)
       Marker7             = self.mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=LiftBoomFFRF['oFFRFreducedOrder'],meshNodeNumbers=np.array(nodeListJoint1), weightingFactors=noodeWeightsJoint1))
       Marker8             = self.mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=LiftBoomFFRF['oFFRFreducedOrder'],meshNodeNumbers=np.array(nodeListPist1),weightingFactors=noodeWeightsPist1))
       Marker9             = self.mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=LiftBoomFFRF['oFFRFreducedOrder'],meshNodeNumbers=np.array(nodeListCyl2),weightingFactors=noodeWeightsCyl2)) 
       Marker10            = self.mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=LiftBoomFFRF['oFFRFreducedOrder'],meshNodeNumbers=np.array(nodeListJoint2), weightingFactors=noodeWeightsJoint2))      
       Marker11            = self.mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=LiftBoomFFRF['oFFRFreducedOrder'],meshNodeNumbers=np.array(nodeListJoint3), weightingFactors=noodeWeightsJoint3))
       
       if self.StaticCase or self.StaticInitialization:
           #compute reference length of distance constraint 
           self.mbs.Assemble()
           TiltP = self.mbs.GetMarkerOutput(Marker11, variableType=exu.OutputVariableType.Position, configuration=exu.ConfigurationType.Initial)
       TiltBoomFFRF        = TiltBoom.AddObjectFFRFreducedOrder(self.mbs, positionRef=TiltP,initialVelocity=[0,0,0],initialAngularVelocity=[0,0,0],
                                                          rotationMatrixRef  = RotationMatrixZ(mt.radians(self.theta2))   ,gravity= [0, -9.8066, 0],
                                                          #massProportionalDamping = 0, stiffnessProportionalDamping = 1e-5 ,
                                                         massProportionalDamping = 0, stiffnessProportionalDamping = 3.35e-3 ,color=color4blue,)
       self.mbs.SetObjectParameter(objectNumber=TiltBoomFFRF['oFFRFreducedOrder'],parameterName='outputVariableTypeModeBasis',value=self.varType)
       Marker13        = self.mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=TiltBoomFFRF['oFFRFreducedOrder'], meshNodeNumbers=np.array(nodeListJoint1T), weightingFactors=noodeWeightsJoint1T))
       Marker14        = self.mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=TiltBoomFFRF['oFFRFreducedOrder'], meshNodeNumbers=np.array(nodeListPist1T),  weightingFactors=noodeWeightsPist1T))  
       MarkerEx        = self.mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=TiltBoomFFRF['oFFRFreducedOrder'], meshNodeNumbers=np.array(nodeListExtT), weightingFactors=noodeWeightsExt1T))  
       
       #Revolute Joint
       self.mbs.AddObject(GenericJoint(markerNumbers=[Marker4, Marker7],constrainedAxes=[1,1,1,1,1,0],visualization=VObjectJointGeneric(axesRadius=0.18*0.263342,axesLength=1.1*0.263342)))
       iCube4          = RigidBodyInertia(mass=m4, com=pMid4, inertiaTensor=Inertia4, inertiaTensorAtCOM=True)
       graphicsCOM4    = GraphicsDataBasis(origin=iCube4.com, length=2*W4)
       iCube5          = RigidBodyInertia(mass=m5, com=pMid5,inertiaTensor=Inertia5,inertiaTensorAtCOM=True)
       graphicsBody5   = GraphicsDataFromSTLfile(fileName5, color4blue,verbose=False, invertNormals=True,invertTriangles=True)
       graphicsBody5   = AddEdgesAndSmoothenNormals(graphicsBody5, edgeAngle=0.25*pi,addEdges=True, smoothNormals=True)
       graphicsCOM5    = GraphicsDataBasis(origin=iCube5.com, length=2*W5)
       ###########################################
       if self.StaticCase or self.StaticInitialization:
           #compute reference length of distance constraint 
           self.mbs.Assemble()
           Bracket1L  = self.mbs.GetMarkerOutput(Marker10, variableType=exu.OutputVariableType.Position, configuration=exu.ConfigurationType.Initial)
           ExtensionP = self.mbs.GetMarkerOutput(MarkerEx, variableType=exu.OutputVariableType.Position, configuration=exu.ConfigurationType.Initial)
           Bracket2T  = self.mbs.GetMarkerOutput(Marker14, variableType=exu.OutputVariableType.Position,  configuration=exu.ConfigurationType.Initial)
           
           #initial assumptions
           theta4_0        = 148.17834497673348
           theta5_0        = 0
           z_dep0          = np.array([theta4_0, theta5_0])
           AbsErr          = 1e-6
           MaxIter         = 100
           Niter           = 0
           ok              = False
           nb              = 4  # bodies in the closed loop
           Rd              = np.zeros((3*nb, nb))
           Ri              = np.zeros((nb, 3))
           Rj              = np.zeros((nb, 3))
           Rot             = np.zeros((nb, 3, 3))
           node1           = self.mbs.GetObjectParameter(LiftBoomFFRF['oFFRFreducedOrder'], 'nodeNumbers')
           node2           = self.mbs.GetObjectParameter(TiltBoomFFRF['oFFRFreducedOrder'], 'nodeNumbers')
           localcom1       = np.array(self.mbs.GetObjectParameter(LiftBoomFFRF['oFFRFreducedOrder'], 'physicsCenterOfMass'))
           localcom2       = np.array(self.mbs.GetObjectParameter(TiltBoomFFRF['oFFRFreducedOrder'], 'physicsCenterOfMass'))
           mass1           = self.mbs.GetObjectParameter(LiftBoomFFRF['oFFRFreducedOrder'], 'physicsMass')
           mass2           = self.mbs.GetObjectParameter(TiltBoomFFRF['oFFRFreducedOrder'], 'physicsMass')
           Rot1            = self.mbs.GetNodeOutput(node1[0],exu.OutputVariableType.Rotation, configuration=exu.ConfigurationType.Reference)
           Rot2            = self.mbs.GetNodeOutput(node2[0],exu.OutputVariableType.Rotation, configuration=exu.ConfigurationType.Reference)
           Rj[0, :]        = LiftP
           Rj[3, :]        = TiltP
           Rot[0, :, :]    = RotXYZ2RotationMatrix(Rot1)
           Rot[3, :, :]    = RotXYZ2RotationMatrix(Rot2)
           Ri[0,:]         = mass1*(Rot[0, :, :] @ localcom1 + Rj[0, :])/mass1
           Ri[3,:]         = mass2*(Rot[3, :, :] @ localcom2 + Rj[3, :])/mass2
           def Rd_Matrix(Rj):
                for i  in range(nb):
                    Rd[3*i + 0, i]  = Rj[i,1]
                    Rd[3*i + 1, i]  = -Rj[i, 0]
                    Rd[3*i + 2, i]  = 1
                return Rd
           def SystemMatrices():
                #if b4 and b5 defined before.
                bodyList = [('LiftBoom', LiftBoomFFRF['oFFRFreducedOrder']),('b4', b4),('b5', b5),('TiltBoom', TiltBoomFFRF['oFFRFreducedOrder'])]
                RD              = np.zeros((3*nb, nb))
                RI              = np.zeros((nb, 3))
                RJ              = np.zeros((nb, 3))
                ROT             = np.zeros((nb, 3, 3))
                for i, (name, objID) in enumerate(bodyList):
                    objParams = self.mbs.GetObject(objID)
                    if 'physicsCenterOfMass' in objParams:
                        localCOM = np.array(objParams['physicsCenterOfMass'])
                        mass     = objParams['physicsMass']
                        if 'nodeNumbers' in objParams:      # Flexible (FFRF)
                            refNode = objParams['nodeNumbers'][0] 
                        else:                                # Rigid body
                            refNode = objParams['nodeNumber']
                        rotVec          = self.mbs.GetNodeOutput(refNode, exu.OutputVariableType.Rotation,configuration=exu.ConfigurationType.Reference)
                        RObject         = RotXYZ2RotationMatrix(rotVec)
                        # Reference node position
                        refPos          = np.array(self.mbs.GetNodeOutput(refNode, exu.OutputVariableType.Position,configuration=exu.ConfigurationType.Initial))
                        # Global COM
                        globalCOM       = mass*(RObject @ localCOM + refPos)
                        # Store in array
                        RI[i, :]        = globalCOM/mass
                        RJ[i, :]        = refPos
                        ROT[i, :, :]    = RObject
                        RD[3*i + 0, i]  = RJ[i,1]
                        RD[3*i + 1, i]  = -RJ[i, 0]
                        RD[3*i + 2, i]  = 1
                return RI, RJ,ROT,RD
           while not ok and Niter < MaxIter:
               Niter += 1
               #if b4 and b5 defined before.
               # self.mbs.Assemble()
               # RI, RJ,ROT,RD = SystemMatrices() 
               Rot[1, :, :] = RotationMatrixZ(mt.radians(z_dep0[0])) #Rotational matrix bracket1 
               Rot[2, :, :] = RotationMatrixZ(mt.radians(z_dep0[1])) #Rotational matrix bracket2
               Rj[1, :]     = Bracket1L                     #Bracket1-Bracket2 connection
               Rj[2, :]     = Rj[1, :]+Rot[1, :, :]@Mark16  #Bracket2 end point
               ContraintP   = Rj[2, :]+Rot[2, :, :]@Mark19  #Bracket2 end point
               Ri[1, :]     = Bracket1L+Rot[1, :, :]@pMid4
               Ri[2, :]     = Rj[2, :]+Rot[2, :, :]@pMid5
               Rd           = Rd_Matrix(Rj)
               #Constraint Eqs 
               # ContraintP   = self.mbs.GetMarkerOutput(Marker19, variableType=exu.OutputVariableType.Position, 
               #                                                           configuration=exu.ConfigurationType.Initial)
               CC           = -ContraintP+Bracket2T
               #Jacobian of constraints
               I2            = np.eye(2)
               I_til         = np.array([[0, -1],[1,  0]])
               D_i_1         = np.eye(3)
               D_i_2         = np.eye(3)
               #Bracket2
               D_i_1[0:2, 2] = I_til@Ri[3,0:2]
               bar_up_1      = ContraintP[0:2]-Ri[2,0:2]
               T_1           = np.array([ [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=float)
               L_1           = np.hstack([I2, Rot[3, 0:2, 0:2] * I_til @ bar_up_1.reshape(2,1)])
               Cz1           = L_1 @ (D_i_1 @ T_1 @ Rd)
               # Tilt boom
               bar_up_2      = Ri[3,0:2]-Bracket2T[0:2]
               T_2           = np.array([[1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],[0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]], dtype=float)
               D_i_2[0:2, 2] = I_til @ Ri[2,0:2]
               L_2           = np.hstack([I2, Rot[2, 0:2, 0:2] * I_til @ bar_up_2.reshape(2,1)])
               Cz2           = L_2 @ (D_i_2 @ T_2 @ Rd)
               # Jacobian part
               Cz            = -Cz2 + Cz1
               Cz_d          = Cz[:, 1:3]
               if np.max(np.abs(CC)) < AbsErr:
                    ok = True
               z_dep = z_dep0 - np.linalg.lstsq(Cz_d, CC[0:2], rcond=None)[0]
               z_dep0=z_dep
           self.theta4 = z_dep[0]
           self.theta5 = z_dep[1]
                

        # Bracket1 
       [n4, b4]        = AddRigidBody(mainSys=self.mbs,inertia=iCube4, nodeType=exu.NodeType.RotationEulerParameters,position=Bracket1L,rotationMatrix=RotationMatrixZ(mt.radians(self.theta4)),
                                                    gravity= [0, -9.8066, 0], graphicsDataList=[graphicsCOM4, graphicsBody4])
       Marker15        = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b4, localPosition=Mark15))                        #With LIft Boom 
       Marker16        = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b4, localPosition=Mark16))  
       if self.StaticCase or self.StaticInitialization:
           # Lets find Bracket1B2 as per initial configuration.
           self.mbs.Assemble()
           ExtensionBoomP = ExtensionP+Rot[3, :, :]@np.array([0, -0.1, 0]) 
           # exu.StartRenderer()
           Bracket1B2       = self.mbs.GetMarkerOutput(Marker16, variableType=exu.OutputVariableType.Position, configuration=exu.ConfigurationType.Initial)
       
       [n5, b5]        = AddRigidBody(mainSys=self.mbs,inertia=iCube5, nodeType=exu.NodeType.RotationEulerParameters, position=Bracket1B2,  rotationMatrix=RotationMatrixZ(mt.radians(self.theta5)) ,   #-5, 140
                                                    gravity= [0, -9.8066, 0],graphicsDataList=[graphicsCOM5, graphicsBody5])
       Marker18        = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b5, localPosition=Mark18))                        #With LIft Boom 
       Marker19        = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b5, localPosition=Mark19))  
       iCube6          = RigidBodyInertia(mass=m6, com=pMid6,inertiaTensor=Inertia6,inertiaTensorAtCOM=True)
       graphicsCOM6    = GraphicsDataBasis(origin=iCube6.com, length=2*W5)
       [n6, b6]        = AddRigidBody(mainSys=self.mbs,inertia=iCube6,nodeType=exu.NodeType.RotationEulerParameters, position=ExtensionBoomP,  # pMid2
                                            rotationMatrix=RotationMatrixZ(mt.radians(self.theta2)) , gravity= [0, -9.8066, 0],graphicsDataList=[graphicsCOM6, graphicsBody6]) 
       Marker20        = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b6, localPosition=[0, 0.09, 0]))
       
       if self.mL != 0:
             TipLoadMarker     = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=b6, localPosition=[2.45, 0.05, 0]))
             pos                = self.mbs.GetMarkerOutput(TipLoadMarker, variableType=exu.OutputVariableType.Position,configuration=exu.ConfigurationType.Reference)
             #print('pos=', pos)
             bMass = self.mbs.CreateMassPoint(physicsMass=self.mL, referencePosition=pos,gravity=[0, -9.8066, 0], show=True, graphicsDataList=[GraphicsDataSphere(radius=0.1, color=color4red)])
             mMass = self.mbs.AddMarker(MarkerBodyPosition(bodyNumber=bMass))
             self.mbs.AddObject(SphericalJoint(markerNumbers=[TipLoadMarker, mMass], visualization=VSphericalJoint(show=False)))
             #self.mbs.Assemble()
             #exu.StartRenderer()
             
       # Joints: Fixed 
       self.mbs.AddObject(GenericJoint(markerNumbers=[MarkerEx, Marker20],constrainedAxes=[1,1,1,1,1,1],visualization=VObjectJointGeneric(axesRadius=0.18*W2,axesLength=0.18)))   
       #Revolute 
       self.mbs.AddObject(GenericJoint(markerNumbers=[Marker11, Marker13],constrainedAxes=[1,1,1,1,1,0], visualization=VObjectJointGeneric(axesRadius=0.22*W3,axesLength=0.16)))     
       #Revolute 
       self.mbs.AddObject(GenericJoint(markerNumbers=[Marker10, Marker15],constrainedAxes=[1,1,1,1,1,0],visualization=VObjectJointGeneric(axesRadius=0.32*W4,axesLength=0.20)))   
       #Revolute
       self.mbs.AddObject(GenericJoint(markerNumbers=[Marker16, Marker18],constrainedAxes=[1,1,1,1,1,0],visualization=VObjectJointGeneric(axesRadius=0.28*W5,axesLength=2.0*W5)))  
       #Revolute 
       self.mbs.AddObject(GenericJoint(markerNumbers=[Marker19, Marker14],constrainedAxes=[1,1,0,0,0,0],visualization=VObjectJointGeneric(axesRadius=0.23*W5,axesLength=0.20)))  

       #ODE1 for pressures:
       nODE1               = self.mbs.AddNode(NodeGenericODE1(referenceCoordinates=[0, 0], initialCoordinates=[self.p1, self.p2], numberOfODE1Coordinates=2))
       nODE2               = self.mbs.AddNode(NodeGenericODE1(referenceCoordinates=[0, 0], initialCoordinates=[self.p3, self.p4],numberOfODE1Coordinates=2))
            
      
       oFriction1       = self.mbs.AddObject(ObjectConnectorSpringDamper(markerNumbers=[Marker5, Marker8], referenceLength=0,stiffness=0, damping=0, force=0, velocityOffset = 0., activeConnector = True,
                                                                 springForceUserFunction=self.CylinderFriction1, visualization=VSpringDamper(show=False) ))
       oFriction2       = self.mbs.AddObject(ObjectConnectorSpringDamper(markerNumbers=[Marker9, Marker16], referenceLength=0,stiffness=0, damping=0, force=0, velocityOffset = 0, activeConnector = True,
                                                                   springForceUserFunction=self.CylinderFriction1, visualization=VSpringDamper(show=False) ))
       
       oHA1 = oHA2 = None
      
       if True:
           oHA1                = self.mbs.AddObject(HydraulicActuatorSimple(name='LiftCylinder', markerNumbers=[ Marker5, Marker8], nodeNumbers=[nODE1], offsetLength=L_Cyl1, strokeLength=L_Pis1, chamberCrossSection0=A[0], 
                                                   chamberCrossSection1=A[1], hoseVolume0=V1, hoseVolume1=V2, valveOpening0=0, valveOpening1=0, actuatorDamping=3e5, oilBulkModulus=Bo, cylinderBulkModulus=Bc, 
                                                   hoseBulkModulus=Bh, nominalFlow=Qn1, systemPressure=pS, tankPressure=pT,  useChamberVolumeChange=True, activeConnector=True, 
                                                   visualization={'show': True, 'cylinderRadius': 50e-3, 'rodRadius': 28e-3, 'pistonRadius': 0.04, 'pistonLength': 0.001, 'rodMountRadius': 0.0, 
                                                                   'baseMountRadius': 20.0e-3, 'baseMountLength': 20.0e-3, 'colorCylinder': color4orange,'colorPiston': color4grey}))
           oHA2                = self.mbs.AddObject(HydraulicActuatorSimple(name='TiltCylinder', markerNumbers=[Marker9, Marker16], nodeNumbers=[nODE2], offsetLength=L_Cyl2, strokeLength=L_Pis2, chamberCrossSection0=A[0], 
                                                     chamberCrossSection1=A[1], hoseVolume0=V1, hoseVolume1=V2, valveOpening0=0, valveOpening1=0, actuatorDamping=5e4, oilBulkModulus=Bo, cylinderBulkModulus=Bc, 
                                                     hoseBulkModulus=Bh, nominalFlow=Qn2, systemPressure=pS, tankPressure=pT, useChamberVolumeChange=True, activeConnector=True, 
                                                     visualization={'show': True, 'cylinderRadius': 50e-3, 'rodRadius': 28e-3, 'pistonRadius': 0.04, 'pistonLength': 0.001, 'rodMountRadius': 0.0, 
                                                    'baseMountRadius': 0.0, 'baseMountLength': 0.0, 'colorCylinder': color4orange, 'colorPiston': color4grey}))
           self.oHA1 = oHA1
           self.oHA2 = oHA2
           
       if self.StaticCase or self.StaticInitialization:
              #compute reference length of distance constraint 
              self.mbs.Assemble()
              mGHposition = self.mbs.GetMarkerOutput(Marker5, variableType=exu.OutputVariableType.Position, configuration=exu.ConfigurationType.Initial)
              mRHposition = self.mbs.GetMarkerOutput(Marker8, variableType=exu.OutputVariableType.Position,  configuration=exu.ConfigurationType.Initial)
              mGHpositionT = self.mbs.GetMarkerOutput(Marker9, variableType=exu.OutputVariableType.Position,  configuration=exu.ConfigurationType.Initial)
              mRHpositionT = self.mbs.GetMarkerOutput(Marker16, variableType=exu.OutputVariableType.Position, configuration=exu.ConfigurationType.Initial)
              dLH0 = NormL2(mGHposition - mRHposition)
              dLH1 = NormL2(mGHpositionT - mRHpositionT)
              if self.verboseMode:
                  print('dLH0=', dLH0)
                  print('dLH1=', dLH1)
              oDC = self.mbs.AddObject(DistanceConstraint(markerNumbers=[Marker5, Marker8], distance=dLH0))
              oDCT = self.mbs.AddObject(DistanceConstraint(markerNumbers=[Marker9, Marker16], distance=dLH1))
       
       self.mbs.variables['isStatics'] = False
       def PreStepUserFunction(mbs, t):
           if not mbs.variables['isStatics']: 
               Av0 = GetInterpolatedSignalValue(t, mbs.variables['inputTimeU1'], timeArray= [], dataArrayIndex= 1, timeArrayIndex= 0, rangeWarning= False)
               Av2 = GetInterpolatedSignalValue(t, mbs.variables['inputTimeU2'], timeArray= [], dataArrayIndex= 1, timeArrayIndex= 0, rangeWarning= False)

               Av1 = -Av0
               Av3 = -Av2
               if oHA1 and oHA2 != None:
                  mbs.SetObjectParameter(oHA1, "valveOpening0", Av0)
                  mbs.SetObjectParameter(oHA1, "valveOpening1", Av1)
                  mbs.SetObjectParameter(oHA2, "valveOpening0", Av2)
                  mbs.SetObjectParameter(oHA2, "valveOpening1", Av3)
           return True
       self.mbs.SetPreStepUserFunction(PreStepUserFunction)  
       if self.verboseMode:
           print('#joint nodes=',len(nodeListJoint3))
           
       
       theta1       = self.mbs.AddSensor(SensorNode( nodeNumber=LiftBoomFFRF['nRigidBody'], storeInternal=True, outputVariableType=exu.OutputVariableType.Rotation))
       theta2       = self.mbs.AddSensor(SensorNode(nodeNumber=TiltBoomFFRF['nRigidBody'], storeInternal=True, outputVariableType=exu.OutputVariableType.Rotation))
       dtheta1      = self.mbs.AddSensor(SensorNode( nodeNumber=LiftBoomFFRF['nRigidBody'], storeInternal=True, outputVariableType=exu.OutputVariableType.AngularVelocity))
       dtheta2      = self.mbs.AddSensor(SensorNode( nodeNumber=TiltBoomFFRF['nRigidBody'], storeInternal=True, outputVariableType=exu.OutputVariableType.AngularVelocity))
       ddtheta1     = self.mbs.AddSensor(SensorNode( nodeNumber=LiftBoomFFRF['nRigidBody'], storeInternal=True, outputVariableType=exu.OutputVariableType.AngularAcceleration))
       ddtheta2     = self.mbs.AddSensor(SensorNode( nodeNumber=TiltBoomFFRF['nRigidBody'], storeInternal=True, outputVariableType=exu.OutputVariableType.AngularAcceleration))
              
       #self.mbs.AddSensor(SensorSuperElement(bodyNumber=LiftBoomFFRF['oFFRFreducedOrder'], meshNodeNumber=LiftTip,  storeInternal=True,outputVariableType=exu.OutputVariableType.DisplacementLocal ))
       deltaY      = self.mbs.AddSensor(SensorSuperElement(bodyNumber=TiltBoomFFRF['oFFRFreducedOrder'], meshNodeNumber=TiltTip,  storeInternal=True, outputVariableType=exu.OutputVariableType.DisplacementLocal ))
       eps1        = self.mbs.AddSensor(SensorSuperElement(bodyNumber=LiftBoomFFRF['oFFRFreducedOrder'], meshNodeNumber=StressNode1,storeInternal=True, outputVariableType=self.varType))
       sig1        = self.mbs.AddSensor(SensorUserFunction(sensorNumbers=[eps1], storeInternal=True, sensorUserFunction=self.UFStressData))  
       eps2        = self.mbs.AddSensor(SensorSuperElement(bodyNumber=TiltBoomFFRF['oFFRFreducedOrder'], meshNodeNumber=StressNode2,storeInternal=True,outputVariableType=self.varType ))
       sig2        = self.mbs.AddSensor(SensorUserFunction(sensorNumbers=[eps2], storeInternal=True, sensorUserFunction=self.UFStressData)) 
   
       self.dictSensors['deltaY']    = deltaY
       self.dictSensors['eps1']      = eps1
       self.dictSensors['eps2']      = eps2
       self.dictSensors['sig1']      = sig1 
       self.dictSensors['sig2']      = sig2
       
       self.dictSensors['theta1']    = theta1 
       self.dictSensors['theta2']    = theta2  
       self.dictSensors['dtheta1']   = dtheta1 
       self.dictSensors['dtheta2']   = dtheta2  
       self.dictSensors['ddtheta1']  = ddtheta1 
       self.dictSensors['ddtheta2']  = ddtheta2  
       
       if oHA1 and oHA2 != None:
           sForce1          = self.mbs.AddSensor(SensorObject(objectNumber=oHA1, storeInternal=True,outputVariableType=exu.OutputVariableType.Force))
           sForce2          = self.mbs.AddSensor(SensorObject(objectNumber=oHA2, storeInternal=True, outputVariableType=exu.OutputVariableType.Force))
           self.dictSensors['sForce1']=sForce1
           self.dictSensors['sForce2']=sForce2
           sDistance1       = self.mbs.AddSensor(SensorObject(objectNumber=oHA1, storeInternal=True, outputVariableType=exu.OutputVariableType.Distance))
           sDistance2       = self.mbs.AddSensor(SensorObject(objectNumber=oHA2, storeInternal=True, outputVariableType=exu.OutputVariableType.Distance))
           self.dictSensors['sDistance1']=sDistance1
           self.dictSensors['sDistance2']=sDistance2
           sVelocity1       = self.mbs.AddSensor(SensorObject(objectNumber=oHA1, storeInternal=True, outputVariableType=exu.OutputVariableType.VelocityLocal))
           sVelocity2       = self.mbs.AddSensor(SensorObject(objectNumber=oHA2, storeInternal=True, outputVariableType=exu.OutputVariableType.VelocityLocal))
           self.dictSensors['sVelocity1']=sVelocity1
           self.dictSensors['sVelocity2']=sVelocity2
           sPressures1      = self.mbs.AddSensor(SensorNode(nodeNumber=nODE1, storeInternal=True, outputVariableType=exu.OutputVariableType.Coordinates))   
           sPressures2      = self.mbs.AddSensor(SensorNode(nodeNumber=nODE2, storeInternal=True, outputVariableType=exu.OutputVariableType.Coordinates))   
           self.dictSensors['sPressures1']=sPressures1
           self.dictSensors['sPressures2']=sPressures2

       #+++++++++++++++++++++++++++++++++++++++++++++++++++
       #assemble and solve    
       self.mbs.Assemble()
       self.simulationSettings = exu.SimulationSettings()   
       self.simulationSettings.solutionSettings.sensorsWritePeriod              = self.endTime / (self.nStepsTotal)
       self.simulationSettings.timeIntegration.numberOfSteps                    = self.GetNSimulationSteps()
       self.simulationSettings.timeIntegration.endTime                          = self.endTime
       self.simulationSettings.timeIntegration.verboseModeFile                  = 0
       self.simulationSettings.timeIntegration.verboseMode                      = self.verboseMode
       self.simulationSettings.timeIntegration.newton.useModifiedNewton         = True
       self.simulationSettings.linearSolverType                                 = exu.LinearSolverType.EigenSparse
       self.simulationSettings.timeIntegration.stepInformation                 += 8
       self.simulationSettings.displayStatistics                                = True
       self.simulationSettings.displayComputationTime                           = True
       self.simulationSettings.linearSolverSettings.ignoreSingularJacobian      = True
       self.simulationSettings.timeIntegration.generalizedAlpha.spectralRadius  = 0.7
       self.SC.visualizationSettings.nodes.show                                 = False
       self.SC.visualizationSettings.contour.outputVariable                     = self.varType
       
       if self.Visualization:
           self.SC.visualizationSettings.window.renderWindowSize                = [1600, 1200]        
           self.SC.visualizationSettings.openGL.multiSampling                   = 4        
           self.SC.visualizationSettings.openGL.lineWidth                       = 3  
           self.SC.visualizationSettings.general.autoFitScene                   = False      
           self.SC.visualizationSettings.nodes.drawNodesAsPoint                 = False        
           self.SC.visualizationSettings.nodes.showBasis                        = True 
           #self.SC.visualizationSettings.markers.                              = True
           exu.StartRenderer()
       
       if self.StaticCase or self.StaticInitialization:
           self.mbs.variables['isStatics'] = True
           self.simulationSettings.staticSolver.newton.relativeTolerance = 1e-4
           # self.simulationSettings.staticSolver.stabilizerODE2term = 2
           self.simulationSettings.staticSolver.verboseMode = self.verboseMode
           self.simulationSettings.staticSolver.numberOfLoadSteps = 1
           self.simulationSettings.staticSolver.constrainODE1coordinates = True #constrain pressures to initial values
       
           # exu.SuppressWarnings(True)
           exu.config.suppressWarnings = True
           self.mbs.SolveStatic(self.simulationSettings, updateInitialValues=True) 
           exu.config.suppressWarnings = False
           #use solution as new initial values for next simulation
           # exu.SuppressWarnings(False)
           
           force1 = self.mbs.GetObjectOutput(oDC, variableType=exu.OutputVariableType.Force)
           force2 = self.mbs.GetObjectOutput(oDCT, variableType=exu.OutputVariableType.Force)
           
           t_dVec, eVal = EOM_SLIDE(self, A_d, nValues=5)
           
           n_etd   = round(np.max(t_dVec)/self.TimeStep)
           self.dictSensors['n_etd']    = n_etd 
            
           
           #print(self.n_etd)

           #now deactivate distance constraint:
           # self.mbs.SetObjectParameter(oDC, 'activeConnector', False)
           #now deactivate distance constraint:
           
           
           if self.verboseMode:
               print('initial force=', force)
           
           #deactivate distance constraint
           self.mbs.SetObjectParameter(oDC, 'activeConnector', False)
           self.mbs.SetObjectParameter(oDCT, 'activeConnector', False)
           
           #overwrite pressures:
           if  oHA1 and oHA2 != None:
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
              self.mbs.systemData.SetODE1Coordinates(coordinates=sysODE1,configuration=exu.ConfigurationType.Initial)
              if self.verboseMode:
                  print('new p0,p1=',sysODE1[nODE1index],sysODE1[nODE1index+1])
                  print('new p2,p3=',sysODE1[nODE2index],sysODE1[nODE2index+1])
           self.mbs.variables['isStatics'] = False

       if not self.StaticCase:
           exu.SolveDynamic(self.mbs, simulationSettings=self.simulationSettings,solverType=exu.DynamicSolverType.TrapezoidalIndex2)

       if self.Visualization:
           # self.SC.WaitForRenderEngineStopFlag()
           exu.StopRenderer()
       
       return self.dictSensors
   
    