
import numpy as np
from models.simulations import ExudynFlexible
from models.ComputeSLIDE import *
from models.Control import *
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class SLIDEModel():

    #initialize class 
    def __init__(self, nStepsTotal=100, endTime=0.5,nModes = 2,loadFromSavedNPY=True, 
                 mL= 50, material='Steel', nnType='FFN', visualization = False,system = True, verboseMode = 0):
        
        self.CreateModel()
        self.Case               = system
        self.nStepsTotal        = nStepsTotal
        self.endTime            = endTime
        self.TimeStep           = self.endTime / (self.nStepsTotal)        
                
        self.angleMinDeg1       = 0
        self.angleMaxDeg1       = 50
        self.n_td               = 0 
        self.n_etd              = 0
        self.p1Init             = 1e7
        self.p2Init             = 1e7
        self.p3Init             = 1e7
        self.p4Init             = 1e7
        self.mL                 = mL
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

    def IsRNN(self):
        return self.nnType == 'RNN'
    
    def IsFFN(self):
        return self.nnType == 'FFN'
    
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

    def CreateModel(self):
        self.SC  = exu.SystemContainer()
        self.mbs = self.SC.AddSystem()
        
    #get time vector according to output data
    def GetOutputXAxisVector(self):
        return self.timeVecOut
    
    
    def CreateInputVector(self, relCnt = 0, theta1=0, isTest=False,SLIDEwindow=False,
                                     Evaluate=False, Plotting= False):
        if self.Case == 'Patu':
            vec         = np.zeros(17*self.nStepsTotal)
            U1          = np.zeros(self.nStepsTotal)
            U2          = np.zeros(self.nStepsTotal)
            U1          =  RandomSignal(self,SLIDEwindow, Evaluate)
            U2          = -RandomSignal(self,SLIDEwindow, Evaluate)
            angleInit1  =  np.random.rand()*(self.angleMaxDeg1-self.angleMinDeg1)+self.angleMinDeg1
            
            if angleInit1 > 25:
                self.angleMinDeg2 =  -22.5
                self.angleMaxDeg2 = -20
            else:
                self.angleMinDeg2 = 0
                self.angleMaxDeg2 = 10
                
            angleInit2                                  = np.random.rand()*(self.angleMaxDeg2-self.angleMinDeg2)+self.angleMinDeg2
            vec[0:self.nStepsTotal]                     = self.timeVecOut
            vec[self.nStepsTotal:2*self.nStepsTotal]    = U1
            vec[2*self.nStepsTotal:3*self.nStepsTotal]  = U2 
            vec[self.nStepsTotal*4]                     = angleInit1 
            vec[self.nStepsTotal*5]                     = angleInit2
            if Plotting: 
                PlottingFunc2( self.timeVecOut, U1, U2)
            
        else:
            vec                                         = np.zeros(9*self.nStepsTotal)
            angleInit1                                  =  np.random.rand()*(self.angleMaxDeg1-self.angleMinDeg1)+self.angleMinDeg1
            U1                                          = np.zeros(self.nStepsTotal)
            U1                                          =  RandomSignal(self,SLIDEwindow, Evaluate)
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
            
        if SLIDEwindow:
            #print('Computing SLIDE window')
            tdVec     = Statistic_SLIDE(self,A_d)
            self.n_td = round(float(tdVec)/self.TimeStep)
            
        slideSteps   = np.array([(self.n_td), (self.n_etd)])


        if self.Visualization:
           self.mbs.SolutionViewer()
           
        return [inputDict, outputDict, slideSteps] 
    



    def SLIDE_Network_Evaluate(self,td,StepPredictor,Liftload, Yprediction, outputVecLong, string=None, system= None):
        
        
        #Control.PlottingFunc ( self.timeVecOut, self.inputTimeU1[:,1] )
        Predeltay       = np.zeros((self.nStepsTotal,1))
        PreEpsxy        = np.zeros((self.nStepsTotal,1))
        Presigmaxx      = np.zeros((self.nStepsTotal,1))
        
        deltay          = np.zeros((self.nStepsTotal,1))
        epsxy           = np.zeros((self.nStepsTotal,1))
        sigmaxx         = np.zeros((self.nStepsTotal,1))
        ns              = self.nStepsTotal
       
        
       # Fig02: Deflection estimation
        deltay               = self.ScaleBackMinusOneToOne(xmin=self.scalDefMin, 
                                            xmax=self.scalDefMax,scaled_x=outputVecLong[0:ns])*1000 #in mm
        epsxy                = self.ScaleBackMinusOneToOne(xmin=self.scalStrainMin, 
                                            xmax=self.scalStrainMax,scaled_x=outputVecLong[ns:2*ns])*1e6 # micron
        
        sigmaxx              = self.ScaleBackMinusOneToOne(xmin=self.scalStressMin, 
                                            xmax=self.scalStressMax,scaled_x=outputVecLong[2*ns:3*ns])/1e6 #MPa
        
        Predeltay[td+StepPredictor:ns]   = self.ScaleBackMinusOneToOne(xmin=self.scalDefMin, 
                                            xmax=self.scalDefMax,scaled_x=Yprediction[td+StepPredictor:ns])*1000
        PreEpsxy[td+StepPredictor:ns]    = self.ScaleBackMinusOneToOne(xmin=self.scalStrainMin, 
                                            xmax=self.scalStrainMax,scaled_x=Yprediction[ns+td+StepPredictor:2*ns])*1e6 # micron
        
        Presigmaxx[td+StepPredictor:ns]  = self.ScaleBackMinusOneToOne(xmin=self.scalStressMin, 
                                            xmax=self.scalStressMax,scaled_x=Yprediction[2*ns+td+StepPredictor:3*ns])/1e6 #MPa
      
        # Structural deflection estimation
        error_percentage    = [mean_absolute_error([true_val], [pred_val]) for true_val, pred_val in 
                               zip(np.abs(deltay[td+StepPredictor:ns]), np.abs(Predeltay[td+StepPredictor:ns]))]
        delyMax             = 2.4 #round(float(np.max(deltay)) +float(np.max(deltay))*0.9)
        erryMax             = round(float(np.max(error_percentage))) 
        eroryP              = round(0.75*delyMax)
        
        fontSize2  = 10
        
        if not system:
            
            fig                 = plt.figure(figsize=(a4_width_inches, a4_height_inches), constrained_layout=True)
            gs                  = gridspec.GridSpec(2, 2,figure=fig, width_ratios=[2, 1], height_ratios=[1, 1], wspace=0.2, hspace=0.3)
            # Plot 1: Main plot
            ax_main = fig.add_subplot(gs[:, 0])
            ax_main.plot(self.timeVecOut,  deltay, color='red', linestyle='-', label='Reference solution')
            ax_main.plot(self.timeVecOut, Predeltay, color='blue', linestyle=':', label='SLIDE estimations')
            ax_main.legend(fontsize=12)
            ax_main.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': fontSize1})
            ax_main.set_ylabel(r'$\delta_\mathrm{y}$, mm', fontdict={'family': 'Times New Roman', 'size': fontSize1})
            ax_main.set_xlim(0, 10)
            ax_main.set_ylim(-delyMax, delyMax) #(-3.5, 3.5) [0 kg, Lift boom]
            x_ticks = [0,     5,10]
            y_ticks = [-delyMax,-delyMax/2,0,delyMax/2, delyMax]
            ax_main.set_xticks(x_ticks)
            ax_main.set_yticks(y_ticks)
            ax_main.set_xticklabels([r'$0$',r'$5$',  r'$10$'])
            ax_main.set_yticklabels([f'$-{delyMax}$',f'$-{delyMax/2}$',r'$0$',f'${delyMax/2}$',f'${delyMax}$'])
            ax_main.yaxis.set_label_coords(-0.1, 0.5)
            ax_main.xaxis.set_label_coords(0.5, -0.07)
            ax_main.grid(color='lightgray', linestyle='--', linewidth=0.5)  
            ax_main.tick_params(axis='both', labelsize=fontSize2)
            ax_main.grid(True)
            ax_main.set_facecolor('#f0f8ff')  # Light blue background
            # Add background color and annotations
            mape = mean_absolute_percentage_error(deltay[td+StepPredictor:self.nStepsTotal], 
                                                  Predeltay[td+StepPredictor:self.nStepsTotal])
            ax_main.annotate(f'Error: {mape:.2f}%', xy=(7.5, -eroryP), fontsize=12, backgroundcolor='lightgrey')
            ax_main.axvline(x=self.timeVecOut[td - 1], color='gray', linestyle='--', linewidth=1)
            ax_main.text(self.timeVecOut[td - 1]+0.08, -0.15, r'$t_d$', fontdict={'family': 'Times New Roman', 'size': fontSize1})
            # Plot 2: Error subplot
            ax_error = fig.add_subplot(gs[0, 1])
            ax_error.plot(self.timeVecOut[td+StepPredictor:self.nStepsTotal], error_percentage, color='green')
            ax_error.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': fontSize1},labelpad=5)
            ax_error.set_ylabel('Error, mm', fontdict={'family': 'Times New Roman', 'size': fontSize1},labelpad=5)
            ax_error.set_xlim(0, 10)
            ax_error.set_ylim(0, erryMax)
            x_ticks = [0, 10]
            y_ticks = [0, erryMax]
            ax_error.set_xticks(x_ticks)
            ax_error.set_yticks(y_ticks)
            ax_error.set_xticklabels([r'$0$', r'$10$'])
            ax_error.set_yticklabels([r'$0$', rf'${erryMax}$' ])
            ax_error.yaxis.set_label_coords(-0.1, 0.5)
            ax_error.xaxis.set_label_coords(0.5, -0.125)
            ax_error.tick_params(axis='both', labelsize=fontSize2)
            ax_error.grid(True)
            ax_error.set_facecolor('#e6ffe6')  # Light green background
            ax_error.grid(color='lightgray', linestyle='--', linewidth=0.5)
            # Plot 3: Zoomed-in subplot
            ax_zoom = fig.add_subplot(gs[1, 1])
            ax_zoom.plot(self.timeVecOut, deltay, color='red', linestyle='-')
            ax_zoom.plot(self.timeVecOut, Predeltay, color='blue', linestyle=':')
            ax_zoom.set_xlim(2, 4)
            ax_zoom.set_ylim(-delyMax, delyMax)
            x_ticks = [2,4]
            y_ticks = [-delyMax, delyMax]
            ax_zoom.set_xticks(x_ticks)
            ax_zoom.set_yticks(y_ticks)
            ax_zoom.set_xticklabels([r'$2$', r'$4$'])
            ax_zoom.set_yticklabels([rf'$-{delyMax}$',rf'${delyMax}$'])
            ax_zoom.tick_params(axis='both', labelsize=fontSize2)
            ax_zoom.set_facecolor('#fffacd')  # Light yellow background
            ax_zoom.grid(color='lightgray', linestyle='--', linewidth=0.5)
            ax_zoom.yaxis.set_label_coords(-0.1, 0.5)
            ax_zoom.xaxis.set_label_coords(0.5, -0.125)
            ax_zoom.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': fontSize1}, labelpad=5)
            ax_zoom.set_ylabel(r'$\delta_\mathrm{y}$, mm', fontdict={'family': 'Times New Roman', 'size': fontSize1},labelpad=5)
            ax_zoom.grid(True)
            # Adjust the layout
            plt.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.125)
            plt.tight_layout()
            plt.savefig(f'solution/Figures/Results/LiftBoom/{Liftload} kg/Evaluate_Deflectiony_{string}.pdf', format='pdf',
                                        bbox_inches='tight')
            plt.show()
        
            
            # Tip strain
            error_percentage1   = [mean_absolute_error([true_val], [pred_val]) for true_val, pred_val 
                                   in zip(epsxy[td+StepPredictor:self.nStepsTotal], 
                                          PreEpsxy[td+StepPredictor:self.nStepsTotal])]        
            epsxyMax            = 16#round(float(np.max(epsxy)) +float(np.max(epsxy)*0.6))
            ErrorEpsxy          = round(float(np.max(error_percentage1)))
            epsxyP              = round(0.8*epsxyMax)
            
            fig1                = plt.figure(figsize=(a4_width_inches, a4_height_inches), constrained_layout=True)
            gs1                 = gridspec.GridSpec(2, 2,figure=fig1, width_ratios=[2, 1], height_ratios=[1, 1], wspace=0.2, hspace=0.3)
            # Plot 1: Main plot
            ax_main1 = fig1.add_subplot(gs1[:, 0])
            ax_main1.plot(self.timeVecOut, epsxy, color='red', linestyle='-', label='Reference solution')
            ax_main1.plot(self.timeVecOut,  PreEpsxy, color='blue', linestyle=':', label='SLIDE estimations')
            ax_main1.legend(fontsize=12)
            ax_main1.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': fontSize1})
            ax_main1.set_ylabel(r'$\epsilon_\mathrm{xy}, \mu$', fontdict={'family': 'Times New Roman', 'size': fontSize1})
            ax_main1.set_xlim(0, 10)
            ax_main1.set_ylim(-epsxyMax, epsxyMax) #(-3.5, 3.5) [0 kg, Lift boom]
            x_ticks = [0,     5,10]
            y_ticks = [-epsxyMax,-epsxyMax/2 ,0,epsxyMax/2 ,epsxyMax]
            ax_main1.set_xticks(x_ticks)
            ax_main1.set_yticks(y_ticks)
            ax_main1.set_xticklabels([r'$0$',r'$5$',  r'$10$'])
            ax_main1.set_yticklabels([f'$-{epsxyMax}$',f'$-{epsxyMax/2:.0f}$',
                                      r'$0$',f'${epsxyMax/2:.0f}$',f'${epsxyMax}$'])
            ax_main1.yaxis.set_label_coords(-0.1, 0.5)
            ax_main1.xaxis.set_label_coords(0.5, -0.07)
            ax_main1.grid(color='lightgray', linestyle='--', linewidth=0.5)  
            ax_main1.tick_params(axis='both', labelsize=fontSize2)
            ax_main1.grid(True)
            ax_main1.set_facecolor('#f0f8ff')  # Light blue background
            # Add background color and annotations
            mape = mean_absolute_percentage_error(epsxy[td+StepPredictor:self.nStepsTotal], 
                                                  PreEpsxy[td+StepPredictor:self.nStepsTotal])
            ax_main1.annotate(f'Error: {mape:.2f}%', xy=(7.5, -12), fontsize=12, backgroundcolor='lightgrey')
            ax_main1.axvline(x=self.timeVecOut[td - 1], color='gray', linestyle='--', linewidth=1)
            ax_main1.text(self.timeVecOut[td - 1]+0.05, -1, r'$t_d$', fontdict={'family': 'Times New Roman', 'size': fontSize1})
            # Plot 2: Error subplot
            ax_error1 = fig1.add_subplot(gs1[0, 1])
            ax_error1.plot(self.timeVecOut[td+StepPredictor:self.nStepsTotal], error_percentage1, color='green')
            ax_error1.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': fontSize1},labelpad=5)
            ax_error1.set_ylabel('Error, $\mu$', fontdict={'family': 'Times New Roman', 'size': fontSize1},labelpad=5)
            ax_error1.set_xlim(0, 10)
            ax_error1.set_ylim(0, ErrorEpsxy)
            x_ticks = [0, 10]
            y_ticks = [0, ErrorEpsxy]
            ax_error1.set_xticks(x_ticks)
            ax_error1.set_yticks(y_ticks)
            ax_error1.set_xticklabels([r'$0$', r'$10$'])
            ax_error1.set_yticklabels([r'$0$', rf'${ErrorEpsxy}$' ])
            ax_error1.yaxis.set_label_coords(-0.1, 0.5)
            ax_error1.xaxis.set_label_coords(0.5, -0.125)
            ax_error1.tick_params(axis='both', labelsize=fontSize2)
            ax_error1.grid(True)
            ax_error1.set_facecolor('#e6ffe6')  # Light green background
            ax_error1.grid(color='lightgray', linestyle='--', linewidth=0.5)
            # Plot 3: Zoomed-in subplot
            ax_zoom1 = fig1.add_subplot(gs1[1, 1])
            ax_zoom1.plot(self.timeVecOut, epsxy, color='red', linestyle='-')
            ax_zoom1.plot(self.timeVecOut, PreEpsxy, color='blue', linestyle=':')
            ax_zoom1.set_xlim(2, 4)
            ax_zoom1.set_ylim(-epsxyMax, epsxyMax)
            x_ticks = [2,4]
            y_ticks = [-epsxyMax, epsxyMax]
            ax_zoom1.set_xticks(x_ticks)
            ax_zoom1.set_yticks(y_ticks)
            ax_zoom1.set_xticklabels([r'$2$', r'$4$'])
            ax_zoom1.set_yticklabels([rf'$-{epsxyMax}$',   rf'${epsxyMax}$'])
            ax_zoom1.tick_params(axis='both', labelsize=fontSize2)
            ax_zoom1.set_facecolor('#fffacd')  # Light yellow background
            ax_zoom1.grid(color='lightgray', linestyle='--', linewidth=0.5)
            ax_zoom1.yaxis.set_label_coords(-0.1, 0.5)
            ax_zoom1.xaxis.set_label_coords(0.5, -0.125)
            ax_zoom1.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': fontSize1}, labelpad=5)
            ax_zoom1.set_ylabel(r'$\epsilon_\mathrm{xy}, \mu$', fontdict={'family': 'Times New Roman', 'size': fontSize1},labelpad=5)
            ax_zoom1.grid(True)
            # Adjust the layout
            plt.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.125)
            plt.tight_layout()
            plt.savefig(f'solution/Figures/Results/LiftBoom/{Liftload} kg/Evaluate_Epsxy_{string}.pdf', format='pdf',
                                bbox_inches='tight')
            plt.show()
            
            
            
            # Structural deflection estimation
            error_percentage2   = [mean_absolute_error([true_val], [pred_val]) for true_val, pred_val in zip(sigmaxx, Presigmaxx)]
            stressMax           = round(float(np.max(sigmaxx)) +float(np.max(sigmaxx)*0.2))
            ErrorStress         = round(float(np.max(error_percentage2)))
            stressP             = round(0.8*stressMax)
            stressMax0          = 35   
            fig2                 = plt.figure(figsize=(a4_width_inches, a4_height_inches), constrained_layout=True)
            gs2                  = gridspec.GridSpec(2, 2, figure=fig2, width_ratios=[2, 1], height_ratios=[1, 1], wspace=0.2, hspace=0.3)
            # Plot 1: Main plot
            ax_main2 = fig2.add_subplot(gs2[:, 0])
            ax_main2.plot(self.timeVecOut, sigmaxx, color='red', linestyle='-', label='Reference solution')
            ax_main2.plot(self.timeVecOut, Presigmaxx, color='blue', linestyle=':', label='SLIDE estimations')
            ax_main2.legend(fontsize=12)
            ax_main2.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': fontSize1})
            ax_main2.set_ylabel(r'$\sigma_\mathrm{xx}$, MPa', fontdict={'family': 'Times New Roman', 'size': fontSize1})
            ax_main2.set_xlim(0, 10)
            ax_main2.set_ylim(-stressMax0, stressMax) #(-3.5, 3.5) [0 kg, Lift boom]
            x_ticks = [0,     5,10]
            y_ticks = [-stressMax0, -20,0,20,40 ,stressMax]
            ax_main2.set_xticks(x_ticks)
            ax_main2.set_yticks(y_ticks)
            ax_main2.set_xticklabels([r'$0$',r'$5$',  r'$10$'])
            ax_main2.set_yticklabels([rf'$-{stressMax0}$', r'$-20$', r'$0$', r'$20$', r'$40$',rf'${stressMax}$'])
            ax_main2.yaxis.set_label_coords(-0.1, 0.5)
            ax_main2.xaxis.set_label_coords(0.5, -0.07)
            ax_main2.grid(color='lightgray', linestyle='--', linewidth=0.5)  
            ax_main2.tick_params(axis='both', labelsize=fontSize2)
            ax_main2.grid(True)
            ax_main2.set_facecolor('#f0f8ff')  # Light blue background
            # Add background color and annotations
            mape = mean_absolute_percentage_error(sigmaxx, Presigmaxx)
            ax_main2.annotate(f'Error: {mape:.2f}%', xy=(7.5, -stressP/2), fontsize=12, backgroundcolor='lightgrey')
            ax_main2.axvline(x=self.timeVecOut[td - 1], color='gray', linestyle='--', linewidth=1)
            ax_main2.text(self.timeVecOut[td - 1]-0.4, -15, r'$t_d$', fontdict={'family': 'Times New Roman', 'size': fontSize1})
            # Plot 2: Error subplot
            ax_error2 = fig2.add_subplot(gs2[0, 1])
            ax_error2.plot(self.timeVecOut, error_percentage2, color='green')
            ax_error2.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': fontSize1},labelpad=5)
            ax_error2.set_ylabel('Error, MPa', fontdict={'family': 'Times New Roman', 'size': fontSize1},labelpad=5)
            ax_error2.set_xlim(0, 10)
            ax_error2.set_ylim(0, ErrorStress)
            x_ticks = [0, 10]
            y_ticks = [0, ErrorStress]
            ax_error2.set_xticks(x_ticks)
            ax_error2.set_yticks(y_ticks)
            ax_error2.set_xticklabels([r'$0$', r'$10$'])
            ax_error2.set_yticklabels([r'$0$', rf'${ErrorStress}$' ])
            ax_error2.yaxis.set_label_coords(-0.1, 0.5)
            ax_error2.xaxis.set_label_coords(0.5, -0.125)
            ax_error2.tick_params(axis='both', labelsize=fontSize2)
            ax_error2.grid(True)
            ax_error2.set_facecolor('#e6ffe6')  # Light green background
            ax_error2.grid(color='lightgray', linestyle='--', linewidth=0.5)
            # Plot 3: Zoomed-in subplot
            ax_zoom2 = fig2.add_subplot(gs2[1, 1])
            ax_zoom2.plot(self.timeVecOut, sigmaxx, color='red', linestyle='-')
            ax_zoom2.plot(self.timeVecOut, Presigmaxx, color='blue', linestyle=':')
            ax_zoom2.set_xlim(2, 4)
            ax_zoom2.set_ylim(-stressMax0, stressMax-2)
            x_ticks = [2,4]
            y_ticks = [-stressMax0, stressMax]
            ax_zoom2.set_xticks(x_ticks)
            ax_zoom2.set_yticks(y_ticks)
            ax_zoom2.set_xticklabels([r'$2$', r'$4$'])
            ax_zoom2.set_yticklabels([
                        rf'$-{stressMax0}$',   
                        rf'${stressMax}$'             
                            ])
            ax_zoom2.tick_params(axis='both', labelsize=fontSize2)
            ax_zoom2.set_facecolor('#fffacd')  # Light yellow background
            ax_zoom2.grid(color='lightgray', linestyle='--', linewidth=0.5)
            ax_zoom2.yaxis.set_label_coords(-0.1, 0.5)
            ax_zoom2.xaxis.set_label_coords(0.5, -0.125)
            ax_zoom2.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': fontSize1}, labelpad=5)
            ax_zoom2.set_ylabel(r'$\sigma_\mathrm{xx}$, MPa', fontdict={'family': 'Times New Roman', 'size': fontSize1},labelpad=5)
            ax_zoom2.grid(True)
            # Adjust the layout
            plt.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.125)
            plt.tight_layout()
            plt.savefig(f'solution/Figures/Results/LiftBoom/{Liftload} kg/Evaluate_sigmaxx_{string}.pdf', format='pdf',
                bbox_inches='tight')
            plt.show()
        else:
            
            fontsize2 = 11
            # # Tip strain
            error_percentage2  = [mean_absolute_error([true_val], [pred_val]) for true_val, pred_val in zip(deltay, Predeltay)]
            defMax              = round(float(np.max(sigmaxx)) +float(np.max(sigmaxx)*0.2))
            Errordef            = round(float(np.max(error_percentage2)))
            defP                = 2 #-0.16#round(0.8*stressMax)
            def0                = 0.5   
            
            a40_width_inches, a40_height_inches = 8.3 / 2, (11.7 / 3.2)   
            fig, ax = plt.subplots(figsize=(a40_width_inches, a40_height_inches))
            ax.plot(self.timeVecOut, 10*deltay, color='red', linestyle='-', label='Reference solution')
            ax.plot(self.timeVecOut,  10*Predeltay, color='blue', linestyle=':', label='SLIDE estimations')
            ax.legend(loc='upper right', fontsize=fontsize2)            
            ax.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': fontsize2})
            ax.set_ylabel(r'$\delta_\mathrm{y}$, mm', fontdict={'family': 'Times New Roman', 'size': fontsize2})
            mape = mean_absolute_percentage_error(deltay[td+StepPredictor:self.nStepsTotal], 
                                                  Predeltay[td+StepPredictor:self.nStepsTotal])
            ax.annotate(f'Error: {mape:.2f}%', xy=(6.5, -1.6), fontsize=fontsize2, backgroundcolor='lightgrey')
            ax.axvline(x=self.timeVecOut[td - 1], color='gray', linestyle='--', linewidth=1)
            ax.text(self.timeVecOut[td]+0.25, -0.1,r'$t_d$', fontdict={'family': 'Times New Roman', 'size': fontsize2})
            ax.set_xlim(0, 10)
            ax.set_ylim(-2, 0.5) #(-3.5, 3.5) [0 kg, Lift boom]
            x_ticks = [0,     5,10]
            y_ticks = [-2, -1,0, 0.5]
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.set_xticklabels([r'$0$',r'$5$',  r'$10$'])
            ax.set_yticklabels([rf'${-2}$',rf'${-1}$',rf'${0}$',rf'${0.5}$'])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            ax.xaxis.set_label_coords(0.5, -0.07)
            ax.grid(color='lightgray', linestyle='--', linewidth=0.5)  
            ax.tick_params(axis='both', labelsize=fontSize2)
            ax.grid(True)
            ax.set_facecolor('#f0f8ff') 
            plt.tight_layout()
            plt.savefig(f'solution/Figures/Results/Patu/{Liftload} kg/Evaluate_deltay_{string}.pdf', format='pdf',
                                bbox_inches='tight')
            plt.show()
            
            
            # Tip strain
            error_percentage1   = [mean_absolute_error([true_val], [pred_val]) for true_val, pred_val in zip(epsxy, PreEpsxy)]        
            epsxyMax            = 40 #round(float(np.max(epsxy)) +float(np.max(epsxy)*0.6))
            ErrorEpsxy          = round(float(np.max(error_percentage1)))
            epsxyP              = round(0.9*epsxyMax)
            
            fig, ax = plt.subplots(figsize=(a40_width_inches, a40_height_inches))
            ax.plot(self.timeVecOut, epsxy, color='red', linestyle='-', label='Reference solution')
            ax.plot(self.timeVecOut,  PreEpsxy, color='blue', linestyle=':', label='SLIDE estimations')
            ax.legend(loc='upper right', fontsize=fontsize2)            
            ax.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': fontSize2})
            ax.set_ylabel(r'$\epsilon_\mathrm{xy}, \mu$', fontdict={'family': 'Times New Roman', 'size': fontsize2})
            mape = mean_absolute_percentage_error(epsxy[td+StepPredictor:self.nStepsTotal], PreEpsxy[td+StepPredictor:self.nStepsTotal])
            ax.annotate(f'Error: {mape:.2f}%', xy=(6.5, 6), fontsize=fontsize2, backgroundcolor='lightgrey')
            ax.axvline(x=self.timeVecOut[td], color='gray', linestyle='--', linewidth=1)
            ax.text(self.timeVecOut[td - 1]-0.5, epsxyMax/4, r'$t_d$', fontdict={'family': 'Times New Roman', 'size': fontsize2})
            ax.set_xlim(0, 10)
            ax.set_ylim(0, epsxyMax) #(-3.5, 3.5) [0 kg, Lift boom]
            x_ticks = [0,     5,10]
            y_ticks = [0,epsxyMax/4, epsxyMax/2, 3*(epsxyMax/4), epsxyMax]
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.set_xticklabels([r'$0$',r'$5$',  r'$10$'])
            ax.set_yticklabels([ r'$0$', r'$10$',r'$20$', r'$30$',rf'${epsxyMax}$'])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            ax.xaxis.set_label_coords(0.5, -0.07)
            ax.grid(color='lightgray', linestyle='--', linewidth=0.5)  
            ax.tick_params(axis='both', labelsize=fontSize2)
            ax.grid(True)
            ax.set_facecolor('#f0f8ff') 
            plt.tight_layout()
            plt.savefig(f'solution/Figures/Results/Patu/{Liftload} kg/Evaluate_Epsxy_{string}.pdf', format='pdf',
                                bbox_inches='tight')
            plt.show()
            
            # Tip strain
            error_percentage2   = [mean_absolute_error([true_val], [pred_val]) for true_val, pred_val in zip(sigmaxx, Presigmaxx)]
            stressMax           = round(float(np.max(sigmaxx)) +float(np.max(sigmaxx)*0.2))
            ErrorStress         = round(float(np.max(error_percentage2)))
            stressP             = 5#round(0.8*stressMax)
            stressMax0          = 5   
            
            fig, ax = plt.subplots(figsize=(a40_width_inches, a40_height_inches))
            ax.plot(self.timeVecOut, sigmaxx, color='red', linestyle='-', label='Reference solution')
            ax.plot(self.timeVecOut,  Presigmaxx, color='blue', linestyle=':', label='SLIDE estimations')
            ax.legend(loc='upper right', fontsize=10)            
            ax.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': 10})
            ax.set_ylabel(r'$\sigma_\mathrm{xx}$, MPa', fontdict={'family': 'Times New Roman', 'size': 10})
            mape = mean_absolute_percentage_error(sigmaxx[td+StepPredictor:self.nStepsTotal], 
                                                  Presigmaxx[td+StepPredictor:self.nStepsTotal])
            ax.annotate(f'Error: {mape:.2f}%', xy=(6.5, (stressP/5)), fontsize=10, backgroundcolor='lightgrey')
            ax.axvline(x=self.timeVecOut[td - 1], color='gray', linestyle='--', linewidth=1)
            ax.text(self.timeVecOut[td]-0.5, (stressP/5),r'$t_d$', fontdict={'family': 'Times New Roman', 'size': 10})
            ax.set_xlim(0, 10)
            ax.set_ylim(0, stressP) #(-3.5, 3.5) [0 kg, Lift boom]
            x_ticks = [0,     5,10]
            y_ticks = [0,stressP/5, 2*(stressP/5),3*(stressP/5),4*(stressP/5),5*(stressP/5)]
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.set_xticklabels([r'$0$',r'$5$',  r'$10$'])
            ax.set_yticklabels([rf'${0}$',rf'${1 * (stressP/5):.0f}$',rf'${2 * (stressP/5):.0f}$',rf'${3 * (stressP/5):.0f}$',
                               rf'${4 * (stressP/5):.0f}$',rf'${5 * (stressP/5):.0f}$'])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            ax.xaxis.set_label_coords(0.5, -0.07)
            ax.grid(color='lightgray', linestyle='--', linewidth=0.5)  
            ax.tick_params(axis='both', labelsize=fontSize2)
            ax.grid(True)
            ax.set_facecolor('#f0f8ff') 
            plt.tight_layout()
            plt.savefig(f'solution/Figures/Results/Patu/{Liftload} kg/Evaluate_sigmaxx_{string}.pdf', format='pdf',
                                bbox_inches='tight')
            plt.show()
        return         
            
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__': #include this to enable parallel processing
    model = SLIDEModel(nStepsTotal=250, endTime=1, verboseMode=1)
    inputData = model.CreateInputVector(0)
    [inputData, output, slideSteps] = model.ComputeModel(inputData, verboseMode=True, solutionViewer=False)
    