
import numpy as np
from models.simulations import ExudynFlexible
from models.compute_SLIDE import *
from models.control import *
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class SLIDEModel():

    #initialize class 
    def __init__(self, nStepsTotal=100, nnType='FFN', endTime=0.5,nModes = 2,loadFromSavedNPY=True, 
                 mL= 50, material='Steel', visualization = False,system = True, verboseMode = 0):
        
        self.CreateModel()
        self.Case               = system
        self.nStepsTotal        = nStepsTotal
        self.endTime            = endTime
        self.TimeStep           = self.endTime / (self.nStepsTotal)        
        self.nnType             = nnType        
        self.angleMinDeg1       = 0
        self.angleMaxDeg1       = 30
        self.n_td               = 0 
        self.n_etd              = 0
        self.p1Init             = 1e7
        self.p2Init             = 1e7
        self.p3Init             = 1e7
        self.p4Init             = 1e7
        self.mL                 = mL
        self.modelName          = 'hydraulics'
        self.modelNameShort     = 'hyd'
        self.Visualization      = visualization
        self.timeVecOut         = np.arange(1,self.nStepsTotal+1)/self.nStepsTotal*self.endTime
        self.dictSensors        = {}
        self.model              = ExudynFlexible(parent=self,nStepsTotal=self.nStepsTotal, endTime=self.endTime,
                                                nModes=nModes,loadFromSavedNPY=loadFromSavedNPY,
                                                mL=mL,material=material,visualization=visualization,
                                                dictSensors=self.dictSensors)
        
    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def NNtype(self):
        return self.nnType
    
    def GetModelName(self):
        self.modelName          = 'hydraulics'
        return self.modelName
    
    #get short model name
    def GetModelNameShort(self):
        return self.modelNameShort

    def IsRNN(self):
        return self.nnType == 'RNN'
    
    def IsFFN(self):
        return self.nnType == 'FFN'
    
    def GetNumberofSensors(self):
       return self.numSensors
   
    def SLIDEWindow(self):
        return self.n_td    
    
    def CreateHiddenInit(self, isTest):
        return np.array([])
    
    def GetInputScaling(self):
        if self.Case == 'Patu':
            self.inputScaling = np.ones(17*self.nStepsTotal)
        else:
            self.inputScaling = np.ones(9*self.nStepsTotal)
        return self.inputScaling
    
    
    
    #return a numpy array with scaling factors for output data
    #also used to determine output dimensions
    def GetOutputScaling(self):
        if self.Case == 'Patu':
            self.outputScaling= np.ones(11*self.nStepsTotal)
        else:
            self.outputScaling = np.ones(6*self.nStepsTotal)
        return self.outputScaling
    
    def GetInputOutputSizeNN(self):
       if self.nnType == 'FFN':
           return [self.inputScaling.size, (self.outputScaling.size,)]
       else:
           return [self.inputScaling.shape[-1], self.outputScaling.shape]
       

    def CreateModel(self):
        self.SC  = exu.SystemContainer()
        self.mbs = self.SC.AddSystem()
        
    #get time vector according to output data
    def GetOutputXAxisVector(self):
        return self.timeVecOut
    
    
    def CreateInputVector(self, relCnt = 0, theta1=0, theta2=0, isTest=False,SLIDEwindow=False,
                                     Evaluate=False, Plotting= False):
        
        if not Evaluate:
            angleInit1  =  np.random.rand()*(self.angleMaxDeg1-self.angleMinDeg1)+self.angleMinDeg1
        else:
            angleInit1  = theta1
        
        if self.Case == 'Patu':
            vec         = np.zeros(17*self.nStepsTotal)
            U1          = np.zeros(self.nStepsTotal)
            U2          = np.zeros(self.nStepsTotal)
            if not Evaluate:
                if angleInit1 > 25:
                    self.angleMinDeg2 =  -22.5
                    self.angleMaxDeg2 = -20
                    
                U1      =  RandomSignal(self,SLIDEwindow, Evaluate)
                U2      = -RandomSignal(self,SLIDEwindow, Evaluate)
            else:
                self.angleMinDeg2 = 0
                self.angleMaxDeg2 = 10
                
                # for i in range(self.nStepsTotal):
                #     U1[i] =uref_1(self.timeVecOut[i])
                #     U2[i] =uref_2(self.timeVecOut[i])
                U1   =  RandomSignal(self,SLIDEwindow, Evaluate)
                U2   =  RandomSignal(self,SLIDEwindow, Evaluate)
            
      
            angleInit2            = np.random.rand()*(self.angleMaxDeg2-self.angleMinDeg2)+self.angleMinDeg2
            angleInit2 = theta2
            
            vec[0:self.nStepsTotal]                     = self.timeVecOut
            vec[self.nStepsTotal:2*self.nStepsTotal]    = U1
            vec[2*self.nStepsTotal:3*self.nStepsTotal]  = U2 
            vec[self.nStepsTotal*4]                     = angleInit1 
            vec[self.nStepsTotal*5]                     = angleInit2
            if Plotting: 
                PlottingFunc2( self.timeVecOut, U1, U2)
            
        else:
            vec                                         = np.zeros(9*self.nStepsTotal)
            U1                                          = np.zeros(self.nStepsTotal)
            if not Evaluate:
                U1      =  RandomSignal(self,SLIDEwindow, Evaluate)
            else:
                U1      =  RandomSignal(self,SLIDEwindow, Evaluate)
                # for i in range(self.nStepsTotal):
                #     U1[i] =uref(self.timeVecOut[i])
           
            vec[0:self.nStepsTotal]                     = self.timeVecOut
            vec[self.nStepsTotal:2*self.nStepsTotal]    = U1
            vec[3*(self.nStepsTotal)]                   = angleInit1
            if Plotting: 
                PlottingFunc( self.timeVecOut, U1)
                
        return vec

    #split input data into initial values, forces or other inputs
    def SplitInputData(self, inputData):
         rv = {}
         if self.Case == 'Patu':
             rv['t']          = inputData[0*(self.nStepsTotal):1*(self.nStepsTotal)]
             rv['U1']         = inputData[1*(self.nStepsTotal):2*(self.nStepsTotal)]      
             rv['U2']         = inputData[2*(self.nStepsTotal):3*(self.nStepsTotal)]   
             rv['s1']         = inputData[3*(self.nStepsTotal):4*(self.nStepsTotal)]    
             rv['s2']         = inputData[4*(self.nStepsTotal):5*(self.nStepsTotal)]    
             rv['ds1']        = inputData[5*(self.nStepsTotal):6*(self.nStepsTotal)]    
             rv['ds2']        = inputData[6*(self.nStepsTotal):7*(self.nStepsTotal)]    
             rv['p1']         = inputData[7*(self.nStepsTotal):8*(self.nStepsTotal)]    
             rv['p2']         = inputData[8*(self.nStepsTotal):9*(self.nStepsTotal)]    
             rv['p3']         = inputData[9*(self.nStepsTotal):10*(self.nStepsTotal)]    
             rv['p4']         = inputData[10*(self.nStepsTotal):11*(self.nStepsTotal)]    
             rv['theta1']     = inputData[11*(self.nStepsTotal):12*(self.nStepsTotal)]    
             rv['dtheta1']    = inputData[12*(self.nStepsTotal):13*(self.nStepsTotal)]    
             rv['ddtheta1']   = inputData[13*(self.nStepsTotal):14*(self.nStepsTotal)]    
             rv['theta2']     = inputData[14*(self.nStepsTotal):15*(self.nStepsTotal)]    
             rv['dtheta2']    = inputData[15*(self.nStepsTotal):16*(self.nStepsTotal)]    
             rv['ddtheta2']   = inputData[16*(self.nStepsTotal):17*(self.nStepsTotal)]    

         else:
             rv['t']          = inputData[0:self.nStepsTotal] 
             rv['U1']         = inputData[1*(self.nStepsTotal):2*(self.nStepsTotal)]
             rv['s1']         = inputData[2*(self.nStepsTotal):3*(self.nStepsTotal)]  
             rv['ds1']        = inputData[3*(self.nStepsTotal):4*(self.nStepsTotal)]    
             rv['p1']         = inputData[4*(self.nStepsTotal):5*(self.nStepsTotal)]    
             rv['p2']         = inputData[5*(self.nStepsTotal):6*(self.nStepsTotal)] 
             rv['theta1']     = inputData[6*(self.nStepsTotal):7*(self.nStepsTotal)]    
             rv['dtheta1']    = inputData[7*(self.nStepsTotal):8*(self.nStepsTotal)]    
             rv['ddtheta1']   = inputData[8*(self.nStepsTotal):9*(self.nStepsTotal)] 
         return rv

    #split input data into initial values, forces or other inputs
    def SplitOutputData(self):
       rv   = {}
       data = []
       if self.Case == 'Patu':
           rv['deltaY']     = data[0*(self.nStepsTotal):1*(self.nStepsTotal)]      
           rv['eps1']       = data[1*(self.nStepsTotal):2*(self.nStepsTotal)]   
           rv['sig1']       = data[2*(self.nStepsTotal):3*(self.nStepsTotal)]      
           rv['eps2']       = data[3*(self.nStepsTotal):4*(self.nStepsTotal)]  
           rv['sig2']       = data[4*(self.nStepsTotal):5*(self.nStepsTotal)]      
           rv['F1']         = data[5*(self.nStepsTotal):8*(self.nStepsTotal)]      
           rv['F2']         = data[8*(self.nStepsTotal):11*(self.nStepsTotal)]  
       else:
           rv['deltaY']     = data[0*(self.nStepsTotal):1*(self.nStepsTotal)]      
           rv['eps1']       = data[1*(self.nStepsTotal):2*(self.nStepsTotal)]   
           rv['sig1']       = data[2*(self.nStepsTotal):3*(self.nStepsTotal)] 
           rv['F1']         = data[3*(self.nStepsTotal):6*(self.nStepsTotal)]      

       return rv
     
    #initialState contains position and velocity states as list of two np.arrays 
    def ComputeModel(self, inputData,hiddenData=None, SLIDEwindow=None,verboseMode = 0, 
                     solutionViewer = False):
        self.CreateModel()        
        self.verboseMode = verboseMode
        
        #++++++++++++++++++++++++++++++++++++++++++
        outputDict                          = self.SplitOutputData()
        inputDict                           = self.SplitInputData(np.array(inputData))
        self.inputTimeU1                    = np.zeros((self.nStepsTotal,2))
        self.inputTimeU1[:,0]               = self.timeVecOut
        self.inputTimeU1[:,1]               = inputDict['U1']
        self.mbs.variables['inputTimeU1']   = self.inputTimeU1      

        if self.Case == 'Patu':
            self.inputTimeU2                    = np.zeros((self.nStepsTotal,2))
            self.inputTimeU2[:,0]               = self.timeVecOut
            self.inputTimeU2[:,1]               = inputDict['U2']       
            self.mbs.variables['inputTimeU2']   = self.inputTimeU2            
            self.mbs.variables['theta1']        = inputData[self.nStepsTotal*4]
            self.mbs.variables['theta2']        = inputData[self.nStepsTotal*5]
            self.dictSensors = self.model.PatuCrane(self.mbs, self.SC, self.mbs.variables['theta1'],self.mbs.variables['theta2'], 
                                                  self.p1Init, self.p2Init, self.p3Init, self.p4Init)
        else:           
            self.mbs.variables['theta1'] = inputData[self.nStepsTotal*3]
            self.dictSensors = self.model.LiftBoom(self.mbs, self.SC,self.mbs.variables['theta1'], self.p1Init, self.p2Init)
        
        if SLIDEwindow:
            #print('Computing SLIDE window')
            DS = self.dictSensors
            
            tdVec     = Statistic_SLIDE(self,A_d)
            self.n_td = round(float(tdVec)/self.TimeStep)
            
            slideSteps   = np.array([(self.n_td), DS['n_etd']])
            
        else:     
            
            slideSteps   = np.array([0, 0])
            DS = self.dictSensors
            # NN input 
            inputDict['t']          = inputData[0:self.nStepsTotal] 
            inputDict['U1']         = inputData[1*(self.nStepsTotal):2*(self.nStepsTotal)]
            inputDict['s1']         = self.mbs.GetSensorStoredData(DS['sDistance1'])[0:1*self.nStepsTotal,1]  
            inputDict['ds1']        = self.mbs.GetSensorStoredData(DS['sVelocity1'])[0:1*self.nStepsTotal,1]   
            inputDict['p1']         = self.mbs.GetSensorStoredData(DS['sPressures1'])[0:self.nStepsTotal,1]   
            inputDict['p2']         = self.mbs.GetSensorStoredData(DS['sPressures1'])[0:self.nStepsTotal,2] 
            inputDict['theta1']     = self.mbs.GetSensorStoredData(DS['theta1'])[0:self.nStepsTotal,3]    
            inputDict['dtheta1']    = self.mbs.GetSensorStoredData(DS['dtheta1'])[0:self.nStepsTotal,3]   
            inputDict['ddtheta1']   = self.mbs.GetSensorStoredData(DS['ddtheta1'])[0:self.nStepsTotal,3]
                
            outputDict['deltaY']    = self.mbs.GetSensorStoredData(DS['deltaY'])[0:1*self.nStepsTotal,2]     
            outputDict['eps1']      = self.mbs.GetSensorStoredData(DS['eps1'])[0:1*self.nStepsTotal,6]  
            outputDict['sig1']      = self.mbs.GetSensorStoredData(DS['sig1'])[0:1*self.nStepsTotal,1]
            outputDict['F1']        = self.mbs.GetSensorStoredData(DS['sForce1'])[0:1*self.nStepsTotal,1:4]      
            
            if self.Case == 'Patu':
                inputDict['U2']          = inputData[2*self.nStepsTotal:3*self.nStepsTotal]
                inputDict['s2']         = self.mbs.GetSensorStoredData(DS['sDistance2'])[0:1*self.nStepsTotal,1]
                inputDict['ds2']         = self.mbs.GetSensorStoredData(DS['sVelocity2'])[0:1*self.nStepsTotal,1]
                inputDict['p3']          = self.mbs.GetSensorStoredData(DS['sPressures2'])[0:self.nStepsTotal,1]   
                inputDict['p4']          = self.mbs.GetSensorStoredData(DS['sPressures2'])[0:self.nStepsTotal,2]
                inputDict['theta2']      = self.mbs.GetSensorStoredData(DS['theta1'])[0:self.nStepsTotal,3]    
                inputDict['dtheta2']     = self.mbs.GetSensorStoredData(DS['dtheta1'])[0:self.nStepsTotal,3]   
                inputDict['ddtheta2']    = self.mbs.GetSensorStoredData(DS['ddtheta1'])[0:self.nStepsTotal,3]
                
                outputDict['eps2']      = self.mbs.GetSensorStoredData(DS['eps2'])[0:1*self.nStepsTotal,6]  
                outputDict['sig2']      = self.mbs.GetSensorStoredData(DS['sig2'])[0:1*self.nStepsTotal,1]
                outputDict['F2']        = self.mbs.GetSensorStoredData(DS['sForce2'])[0:1*self.nStepsTotal,1:4]
                
        if solutionViewer:
           self.mbs.SolutionViewer()
           
        return [inputDict, outputDict, slideSteps] 
    



   
            
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__': #include this to enable parallel processing
    model = SLIDEModel(nStepsTotal=250, endTime=1, verboseMode=1)
    inputData = model.CreateInputVector(0)
    [inputData, output, slideSteps] = model.ComputeModel(inputData, verboseMode=True, solutionViewer=False)
    