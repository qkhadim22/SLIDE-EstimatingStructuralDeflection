
from models.container import *

def roundSignificant(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags

def Statistic_SLIDE(self,A_d, Patu=None):
        
        # Inputs and outputs from the simulation.
        DS                      = self.dictSensors
        DeflectionTip           = self.dictSensors['deltaY']
        StrainTip               = self.dictSensors['eps1']
        StressPoint             = self.dictSensors['sig1']
        
        dampedSteps             = np.array([-1])
        data1                   =  self.inputTimeU1
        Time                    = self.timeVecOut
        self.deflection         = np.zeros((self.nStepsTotal,2))
        self.strain             = np.zeros((self.nStepsTotal,2))
        self.stress             = np.zeros((self.nStepsTotal,2))
        
        self.deflection[:,0]    = Time
        self.strain[:,0]        = Time
        self.stress[:,0]        = Time
        self.deflection[:,1]    = self.mbs.GetSensorStoredData(DeflectionTip)[0:self.nStepsTotal,2]
        self.strain[:,1]        = self.mbs.GetSensorStoredData(StrainTip)[0:1*self.nStepsTotal,6]
        self.stress[:,1]        = self.mbs.GetSensorStoredData(StressPoint)[0:1*self.nStepsTotal,1]

        
        
        fontsize = 12
        if not  self.Case == 'Patu':
            
           #Assumptions
           #threshold_deflection    = -np.log(A_d)*np.max(np.abs(self.deflection[0:self.nStepsTotal,1]))
           threshold_deflection    = (1-2*np.log(A_d)/100)*np.mean((self.deflection[0:self.nStepsTotal,1])) #(1-2*np.log(A_d)/100)*
           #threshold_deflection    = 1.10 *np.mean((self.deflection[0:self.nStepsTotal,1]))
           min_consecutive_steps   = round(0.10*self.nStepsTotal) #0.2
           
           ##################################################################
                               #SLIDE Computing#
           ##################################################################
           
           def Compute(self, threshold_deflection, min_consecutive_steps): 
               
                   consecutive_count       = 0
                   SLIDE_time              = 0


                   for i in range(len(self.deflection[:, 1])):
                       if self.deflection[i, 1] < threshold_deflection:
                           consecutive_count += 1

                           if consecutive_count == 1:
                               SLIDE_time = self.deflection[i, 0]

                           if consecutive_count >= min_consecutive_steps:
                               # Optionally do something more
                               pass
                       else:
                           consecutive_count = 0

                   return SLIDE_time
    
           SLIDE_time    =   Compute(self, threshold_deflection, min_consecutive_steps)  
           
           #print(f'SLIDE window:{SLIDE_time} s')
           #print(f'Steps in SLIDE window:{self.n_td}')
            # Control signal
           a40_width_inches, a40_height_inches = 8.3 / 2, 11.7 / 4  
           fig, ax = plt.subplots(figsize=(a40_width_inches, a40_height_inches))
           
           ax.plot(data1[:,0], data1[:,1], color='k')
           ax.set_xlabel(r'Time, s', fontdict={'family': 'Times New Roman', 'size': fontsize})
           ax.set_ylabel(r'Control signal, V', fontdict={'family': 'Times New Roman', 'size': fontsize})
           
           ax.axvspan(0, 0.101, ymin = 0.5, ymax = 0.925, color='lightgreen', 
                      alpha=0.2, label='Spool open')
           
           ax.text(0.18, -0.4, 'Spool open', horizontalalignment='center', color='black', fontsize=fontsize, 
                       bbox=dict(facecolor='white', edgecolor='black', pad=4.0))
           
           # ax.annotate('',                      # no text
           #             xy=(0.203, -0.3),         # head of the arrow (end point)
           #             xytext=(-0.003, -0.3),        # tail of the arrow (start point)
           #             arrowprops=dict(arrowstyle="<->", color='black', lw=2))
           
           ax.set_xlim(0, 1)
           ax.set_ylim(-1.2, 1.2)
           x_ticks = [0, 0.1, 0.5, 1]
           y_ticks = [-1,0,1]
           ax.set_xticks(x_ticks)
           ax.set_yticks(y_ticks)
           ax.set_xticklabels([r'$0$', r'$0.1$', r'$0.5$', r'$1$'])
           ax.set_yticklabels([r'$U_{min}$', r'$0$', r'$U_{max}$'])    
           ax.tick_params(axis='both', labelsize=fontsize)  # Change 8 to the desired font size
           # Enable grid and legend
           ax.grid(True)
           # Adjust layout and save the figure
           #plt.tight_layout()
           plt.savefig('solution/Figures/Fig7/LiftBoom/TestControl_Plot.pdf', format='pdf', bbox_inches='tight')
           #plt.show()
           plt.close()

            
           # Deflection
           MaxVal = 7
           fig, ax = plt.subplots(figsize=(a40_width_inches, a40_height_inches))
           ax.plot(Time, 1000*self.deflection[:,1], color='gray')
           ax.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': fontsize})
           ax.set_ylabel(r'${\delta}_{\mathrm{y}}$, mm', fontdict={'family': 'Times New Roman', 'size': fontsize})
           ax.text(SLIDE_time + 0.16, MaxVal/4, r'${{\delta}_{\mathrm{y}}}^{*}$', horizontalalignment='center', color='black', fontsize=fontsize, 
                     bbox=dict(facecolor='white', edgecolor='black', pad=4.0))
           ax.axvline(SLIDE_time, color='red', alpha=0.5, linestyle='-.', linewidth=2, label='Midpoint of Open')
           ax.axvspan(0, SLIDE_time, ymin = 0, ymax = 1, color='lightgreen', 
                       alpha=0.2, label='Spool open')            
           ax.text(SLIDE_time/2, MaxVal*0.75, r'${{\mathrm{t}}_{\mathrm{d}}}^{*}$', horizontalalignment='center', color='black', fontsize=fontsize, 
                        bbox=dict(facecolor='white', edgecolor='black', pad=4.0))
           ax.annotate('',                      # no text
                        xy=(SLIDE_time+0.005, MaxVal*0.58),         # head of the arrow (end point)
                        xytext=(-0.003, MaxVal*0.58),        # tail of the arrow (start point)
                        arrowprops=dict(arrowstyle="<->", color='black', lw=2))
           center_x, center_y = SLIDE_time, threshold_deflection*1000 
           width, height = 0.02, 0.18  # Size of the rectangle, adjust as necessary
           rectangle = Rectangle((center_x - width / 2, center_y - height / 2), width, height,
                                  edgecolor='black', facecolor='black', fill=True)
           ax.add_patch(rectangle)
           ax.annotate('',  # No text, just the arrow
                        xy=(center_x, center_y),  # Arrow points to the center of the rectangle
                        xytext=(SLIDE_time+0.118, 1.525),  # Starting point of the arrow at text location
                        arrowprops=dict(arrowstyle="->", color='black'))
           ax.set_xlim(0, 1)
           ax.set_ylim(-MaxVal, MaxVal)
           x_ticks = [0, SLIDE_time,0.5, 0.75, 1]
           y_ticks = [-MaxVal,-MaxVal/2,    0, MaxVal/2,     MaxVal]
           ax.set_xticks(x_ticks)
           ax.set_yticks(y_ticks)
           ax.set_xticklabels([r'$0$', f'${SLIDE_time}$',r'$0.5$', r'$0.75$', r'$1$'])
           ax.set_yticklabels([f'${-MaxVal}$', f'${-MaxVal/2}$', r'$0$',f'${MaxVal/2}$', f'${MaxVal}$']) 
           ax.tick_params(axis='both', labelsize=fontsize)  # Change 8 to the desired font size
           # Enable grid and legend
           ax.grid(True)
           # Adjust layout and save the figure
           #plt.tight_layout()
           plt.savefig(f'solution/Figures/Fig7/LiftBoom/TestDef_Plot_{self.mL}.pdf', format='pdf', bbox_inches='tight')
           #plt.show()
           plt.close()

           
           
           fig, ax = plt.subplots(figsize=(a40_width_inches, a40_height_inches))
           ax.plot(Time, self.strain[:,1], color='gray')
           ax.set_xlabel('Time [s]', fontdict={'family': 'Times New Roman', 'size': fontsize})
           ax.set_ylabel(r'${\epsilon}_{{xy}}$', fontdict={'family': 'Times New Roman', 'size': fontsize})

           ax.text(SLIDE_time + 0.16, 1.525, r'${{\epsilon}_{xy}}^{*}$', horizontalalignment='center', color='black', fontsize=fontsize, 
                     bbox=dict(facecolor='white', edgecolor='black', pad=4.0))
           ax.axvline(SLIDE_time, color='red', alpha=0.5, linestyle='-.', linewidth=2, label='Midpoint of Open')
           ax.axvspan(0, SLIDE_time, ymin = 0, ymax = 1, color='lightgreen', 
                       alpha=0.2, label='Spool open')            
           ax.text(SLIDE_time/2, 10*0.72, r'${{\mathrm{t}}_{\mathrm{d}}}^{*}$', horizontalalignment='center', color='black', fontsize=fontsize, 
                        bbox=dict(facecolor='white', edgecolor='black', pad=4.0))
           ax.annotate('',                      # no text
                        xy=(SLIDE_time+0.005, 10*0.58),         # head of the arrow (end point)
                        xytext=(-0.003, 10*0.58),        # tail of the arrow (start point)
                        arrowprops=dict(arrowstyle="<->", color='black', lw=2))
           center_x, center_y = SLIDE_time, threshold_deflection*1000 
           width, height = 0.02, 0.18  # Size of the rectangle, adjust as necessary
           rectangle = Rectangle((center_x - width / 2, center_y - height / 2), width, height,
                                  edgecolor='black', facecolor='black', fill=True)
           ax.add_patch(rectangle)
           ax.annotate('',  # No text, just the arrow
                        xy=(center_x, center_y),  # Arrow points to the center of the rectangle
                        xytext=(SLIDE_time+0.118, 1.525),  # Starting point of the arrow at text location
                        arrowprops=dict(arrowstyle="->", color='black'))
           ax.set_xlim(0, 1)
           ax.set_ylim(-1.5e-5, 1.5e-5)
           x_ticks = [0, SLIDE_time,0.75, 1]
           y_ticks = [-1.5e-5,    0,      1.5e-5]
           ax.set_xticks(x_ticks)
           ax.set_yticks(y_ticks)
           ax.set_xticklabels([r'$0$', f'${SLIDE_time}$', r'$0.75$', r'$1$'])
           ax.set_yticklabels([r'$-1.5e-5$', r'$0$', r'$1.5e-5$']) 
           ax.tick_params(axis='both', labelsize=fontsize)  # Change 8 to the desired font size
           # Enable grid and legend
           ax.grid(True)
           # Adjust layout and save the figure
           #plt.tight_layout()
           plt.savefig(f'solution/Figures/Fig7/LiftBoom/TestStrain_Plot_{self.mL}.pdf', format='pdf', bbox_inches='tight')
           #plt.show()           
           plt.close()

           
           fig, ax = plt.subplots(figsize=(a40_width_inches, a40_height_inches))
           ax.plot(Time, self.stress[:,1], color='gray')
           ax.set_xlabel('Time [s]', fontdict={'family': 'Times New Roman', 'size': fontsize})
           ax.set_ylabel(r'${\sigma}_{{xx}}$', fontdict={'family': 'Times New Roman', 'size': fontsize})

           ax.text(SLIDE_time + 0.16, 2e7, r'${{\sigma}_{{xx}}}^{*}$', horizontalalignment='center', color='black', fontsize=fontsize, 
                     bbox=dict(facecolor='white', edgecolor='black', pad=4.0))
           ax.axvline(SLIDE_time, color='red', alpha=0.5, linestyle='-.', linewidth=2, label='Midpoint of Open')
           ax.axvspan(0, SLIDE_time, ymin = 0, ymax = 1, color='lightgreen', 
                       alpha=0.2, label='Spool open')            
           ax.text(SLIDE_time/2+0.1, -2.2e7, r'${{\mathrm{t}}_{\mathrm{d}}}^{*}$', horizontalalignment='center', color='black', fontsize=fontsize, 
                        bbox=dict(facecolor='white', edgecolor='black', pad=4.0))
           ax.annotate('',                      # no text
                        xy=(SLIDE_time+0.005, -3e7),         # head of the arrow (end point)
                        xytext=(-0.003, -3e7),        # tail of the arrow (start point)
                        arrowprops=dict(arrowstyle="<->", color='black', lw=2))
           center_x, center_y = SLIDE_time, threshold_deflection*1000 
           width, height = 0.02, 0.18  # Size of the rectangle, adjust as necessary
           rectangle = Rectangle((center_x - width / 2, center_y - height / 2), width, height,
                                  edgecolor='black', facecolor='black', fill=True)
           ax.add_patch(rectangle)
           ax.annotate('',  # No text, just the arrow
                        xy=(center_x, center_y),  # Arrow points to the center of the rectangle
                        xytext=(SLIDE_time+0.118, 2e7),  # Starting point of the arrow at text location
                        arrowprops=dict(arrowstyle="->", color='black'))
           ax.set_xlim(0, 1)
           ax.set_ylim(-4e7, 4e7)
           x_ticks = [0, SLIDE_time,0.75, 1]
           y_ticks = [-4e7,    0,      4e7]
           ax.set_xticks(x_ticks)
           ax.set_yticks(y_ticks)
           ax.set_xticklabels([r'$0$', f'${SLIDE_time}$', r'$0.75$', r'$1$'])
           ax.set_yticklabels([r'$-4e7$', r'$0$', r'$4e7$']) 
           ax.tick_params(axis='both', labelsize=fontsize)  # Change 8 to the desired font size
           # Enable grid and legend
           ax.grid(True)
           # Adjust layout and save the figure
           #plt.tight_layout()
           plt.savefig(f'solution/Figures/Fig7/LiftBoom/TestStress_Plot_{self.mL}.pdf', format='pdf', bbox_inches='tight')
          # plt.show()
           plt.close()
           
           # Deflection
           # fig, ax = plt.subplots(figsize=(a40_width_inches, a40_height_inches))
           # ax.plot(Time, 1000*self.deflection[:,1], color='blue')
           # ax.set_xlim(0, 1)
           # ax.set_ylim(-10, 10)
           # x_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
           # y_ticks = [-15, -7.5,      0,7.5,      15]
           # ax.set_xticks(x_ticks)
           # ax.set_yticks(y_ticks)
           # ax.set_xticklabels([])
           # ax.set_yticklabels([])
           # ax.tick_params(axis='both', labelsize=11)  # Change 8 to the desired font size
           # # Enable grid and legend
           # ax.grid(True)
           # # Adjust layout and save the figure
           # plt.tight_layout()
           # plt.savefig(f'SLIDE/LiftBoom/TestDef_Plot.png', format='png', dpi=300)
           # plt.show()           
            
        else:
            
            #Assumptions
            #threshold_deflection    = -np.log(A_d)*np.max(np.abs(self.deflection[0:self.nStepsTotal,1]))
            threshold_deflection    = (1-1*np.log(A_d)/100)*np.mean((self.deflection[0:self.nStepsTotal,1])) #(1-2*np.log(A_d)/100)*
            #threshold_deflection    = 1.10 *np.mean((self.deflection[0:self.nStepsTotal,1]))
            min_consecutive_steps   = round(0.10*self.nStepsTotal) #0.2
            
            ##################################################################
                                #SLIDE Computing#
            ##################################################################
            
            def Compute(self, threshold_deflection, min_consecutive_steps): 
                
                    consecutive_count       = 0
                    SLIDE_time              = 0


                    for i in range(len(self.deflection[:, 1])):
                        if self.deflection[i, 1] < threshold_deflection:
                            consecutive_count += 1

                            if consecutive_count == 1:
                                SLIDE_time = self.deflection[i, 0]

                            if consecutive_count >= min_consecutive_steps:
                                # Optionally do something more
                                pass
                        else:
                            consecutive_count = 0

                    return SLIDE_time
     
            SLIDE_time    =   Compute(self, threshold_deflection, min_consecutive_steps)  
            
            #print(f'SLIDE window:{SLIDE_time} s')
            #print(f'Steps in SLIDE window:{self.n_td}')
            data2  =  self.inputTimeU2
           
            # For publication
            a40_width_inches, a40_height_inches = 8.3 / 2, 11.7 / 4  
            fig, ax = plt.subplots(figsize=(a40_width_inches, a40_height_inches))           
            ax.plot(Time, data1[:,1], color='k')
            ax.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': 10})
            ax.set_ylabel(r'Control signal, V', fontdict={'family': 'Times New Roman', 'size': 10})            
            ax.axvspan(0, Time[-1]*0.1015, ymin = 0.5, ymax = 0.90, color='lightgreen', 
                       alpha=0.2, label='Spool open')            
            ax.text(Time[-1]*0.15, -0.60, 'Spool open', horizontalalignment='center', color='black', fontsize=8, 
                        bbox=dict(facecolor='white', edgecolor='black', pad=4.0))           
            ax.annotate('',                      # no text
                        xy=(Time[-1]*0.2015, -0.25),         # head of the arrow (end point)
                        xytext=(-0.003, -0.25),        # tail of the arrow (start point)
                        arrowprops=dict(arrowstyle="<->", color='black', lw=2))
            # Set ticks
            ax.set_xlim(0, Time[-1])
            ax.set_ylim(-1.2, 1.2)
            x_ticks = [0, 0.2, 1, 2]
            y_ticks = [-1,0,1]
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.set_xticklabels([r'$0$', r'$0.2$', r'$1$', r'$2$'])
            ax.set_yticklabels([r'$U_{min}$', r'$0$', r'$U_{max}$'])   
            ax.tick_params(axis='both', labelsize=8)  # Change 8 to the desired font size
            # Enable grid and legend
            ax.grid(True)
            # Adjust layout and save the figure
            plt.tight_layout()
            plt.savefig('solution/Figures/Fig7/PATU/SLIDE_Liftcontrol_Plot.pdf', format='pdf', bbox_inches='tight')
            plt.close()
            
            
            # For publication
            a40_width_inches, a40_height_inches = 8.3 / 2, 11.7 / 4  
            fig, ax = plt.subplots(figsize=(a40_width_inches, a40_height_inches))           
            ax.plot(Time, data2[:,1], color='k')
            ax.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': 10})
            ax.set_ylabel(r'Control signal, V', fontdict={'family': 'Times New Roman', 'size': 10})            
            ax.axvspan(0, Time[-1]*0.1015, ymin = 0.1, ymax = 0.50, color='lightgreen', 
                       alpha=0.2, label='Spool open')            
            ax.text(Time[-1]*0.15, 0.60, 'Spool open', horizontalalignment='center', color='black', fontsize=8, 
                        bbox=dict(facecolor='white', edgecolor='black', pad=4.0))           
            ax.annotate('',                      # no text
                        xy=(Time[-1]*0.2015, 0.25),         # head of the arrow (end point)
                        xytext=(-0.003, 0.25),        # tail of the arrow (start point)
                        arrowprops=dict(arrowstyle="<->", color='black', lw=2))
            ax.set_xlim(0, Time[-1])
            ax.set_ylim(-1.2, 1.2)
            x_ticks = [0, 0.2, 1, 2]
            y_ticks = [-1,0,1]
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.set_xticklabels([r'$0$', r'$0.2$', r'$1$', r'$2$'])
            ax.set_yticklabels([r'$U_{min}$', r'$0$', r'$U_{max}$'])  
            ax.tick_params(axis='both', labelsize=8)  # Change 8 to the desired font size
            # Enable grid and legend
            ax.grid(True)
            # Adjust layout and save the figure
            plt.tight_layout()
            plt.savefig('solution/Figures/Fig7/PATU/Tilt_control_Plot.pdf', format='pdf',bbox_inches='tight')
            plt.close()
            

            # Deflection
            fig, ax = plt.subplots(figsize=(a40_width_inches, a40_height_inches))
            ax.plot(Time, 1000*self.deflection[:,1], color='gray')
            ax.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': 10})
            ax.set_ylabel(r'${\delta}_{\mathrm{y}}$, mm', fontdict={'family': 'Times New Roman', 'size': 10})
            ax.text(SLIDE_time + 0.16, 1.525, r'${{\delta}_{\mathrm{y}}}^{*}$', horizontalalignment='center', color='black', fontsize=10, 
                        bbox=dict(facecolor='white', edgecolor='black', pad=4.0))
            ax.axvline(SLIDE_time, color='red', alpha=0.5, linestyle='-.', linewidth=2, label='Midpoint of Open')
            ax.axvspan(0, SLIDE_time, ymin = 0, ymax = 1, color='lightgreen', 
                          alpha=0.2, label='Spool open')            
            ax.text(SLIDE_time/2, 10*0.75, r'$\mathrm{t}_\mathrm{d}$', horizontalalignment='center', color='black', fontsize=10, 
                           bbox=dict(facecolor='white', edgecolor='black', pad=4.0))
            ax.annotate('',                      # no text
                           xy=(SLIDE_time+0.005, 10*0.6),         # head of the arrow (end point)
                           xytext=(-0.003, 10*0.6),        # tail of the arrow (start point)
                           arrowprops=dict(arrowstyle="<->", color='black', lw=2))
            center_x, center_y = SLIDE_time, threshold_deflection*1000  # These replace your damped_time -0.52
            width, height = 0.02, 0.18  # Size of the rectangle, adjust as necessary
            rectangle = Rectangle((center_x - width / 2, center_y - height / 2), width, height,
                                     edgecolor='black', facecolor='black', fill=True)
            ax.add_patch(rectangle)
            ax.annotate('',  # No text, just the arrow
                           xy=(center_x, center_y),  # Arrow points to the center of the rectangle
                           xytext=(SLIDE_time+0.118, 1.525),  # Starting point of the arrow at text location
                           arrowprops=dict(arrowstyle="->", color='black'))
            ax.set_xlim(0, 2)
            # Set ticks
            ax.set_xticks(np.linspace(0, 2, 6))
            ax.set_yticks(np.linspace(-10, 10, 6))
            ax.tick_params(axis='both', labelsize=8)  # Change 8 to the desired font size
            # Enable grid and legend
            ax.grid(True)
            # Adjust layout and save the figure
            plt.tight_layout()
            plt.savefig(f'solution/Figures/Fig7/PATU/Def_Plot_{self.mL}.pdf', format='pdf', bbox_inches='tight')
            plt.close()
                
        return SLIDE_time
    
    
def EOM_SLIDE(self,  A_d=0.01, nValues=5, flagDebug=True, computeComplexEigenvalues=True): 
    
            # exu.SuppressWarnings(True)
            exu.config.suppressWarnings = True
            [eigenValues, eVectors] = self.mbs.ComputeODE2Eigenvalues(computeComplexEigenvalues=True,
                                                      useAbsoluteValues=False)
            exu.config.suppressWarnings = False
            # exu.SuppressWarnings(False)
            
            # here the sqrt is already taken! 
            w, D, eVal = [], [], []
            for i in range(len(eigenValues)): 
                if eigenValues[i].imag == 0: 
                    pass
                else: 
                    eigenFreq = abs(eigenValues[i].imag)
                    eigenD = abs(eigenValues[i].real)/abs(eigenValues[i].imag)
                    if not(eigenFreq in w): 
                        w += [eigenFreq]
                        D += [eigenD] # note peter: this is not exact for spring-damper... 
                        
                
                eVal += [abs(np.real(eigenValues[i]))]
                if not(computeComplexEigenvalues):
                    eVal[-1] = np.sqrt(eVal[-1]) # when not calculating the complex eigenvalues we get the squared eigenvalues... 
                # else: 
                #     print('not adding eigenValue: ', abs(np.real(eigenValues[i])))
                #     print('current list: ', eVal[-1])
        
            # if not(self.endTime is None): 
            #     print('after t={} solution is damped down: '.format(self.endTime))
            for i in range(len(eVal)): 
                # t_d = - np.log(A_d)/(w[i] * D[i])
                if self.endTime is None: 
                    t_d = -np.log(A_d)/(eVal[i])
                    
                else: 
                    A_rel = np.exp(-eVal[i]*self.endTime)
                    
            # eigenvalues are already sorted.
            t_dMax = - np.log(A_d)/(eVal[-nValues::])
            if not(self.endTime is None): 
                A_dMax = np.exp(-np.array(eVal[-nValues::])*self.endTime)
            else: 
                A_dMax = None
            
            return t_dMax, eVal[-nValues::]# t_dMax, A_dMax    



def DataStat():
    data  = np.load("solution/data/LiftBoom/FFNT1024-256s200t1E2.1e+11Density7.8e+03Load0.npy", allow_pickle=True)[()]


    X_train = torch.tensor(data['inputsTraining'])  # Extracting the first training sample as a torch tensor
    Y_train = torch.tensor(data['targetsTraining'])  # Extracting training targets and squeezing dimensions
    X_test = torch.tensor(data['inputsTest'])  # Extracting the first test sample as a torch tensor
    Y_test = torch.tensor(data['targetsTest'])  # Extracting test targets and squeezing dimensions
    X_InitTrain = torch.tensor(data['hiddenInitTraining'])  # Extracting the first test sample as a torch tensor
    X_InitTest = torch.tensor(data['hiddenInitTest'])  # Extracting test targets and squeezing dimensions

    # Get the shape of the variable
    X_train       = np.array(X_train)
    Y_train       = np.array(Y_train)
    X_test        = np.array(X_test)
    Y_test        = np.array(Y_test)


    U             = X_train[0:1023, 0, 0:200].flatten()
    s             = X_train[0:1023, 0, 200:400].flatten()
    dots          = X_train[0:1023, 0, 400:600].flatten()
    p1            = X_train[0:1023, 0, 600:800].flatten()
    p2            = X_train[0:1023, 0, 800:1000].flatten()
    deltaY        = Y_train[0:1023, 0:200].flatten()
    strainXY      = Y_train[0:1023, 200:400].flatten()
    stressXX      = Y_train[0:1023, 400:600].flatten()
    nTrainings    = 1024

    mu1, std1     = U.mean(), U.std()
    mu2, std2     = s.mean(), s.std()
    mu3, std3     = dots.mean(), dots.std()
    mu4, std4     = p1.mean(), p1.std()
    mu5, std5     = p2.mean(), p2.std()
    mu6, std6     = deltaY.mean(), deltaY.std()
    mu7, std7     = strainXY.mean(), strainXY.std()
    mu8, std8     = stressXX.mean(), stressXX.std()

    x1            = np.linspace(U.min(), U.max(), nTrainings)
    y1            = norm.pdf(x1, mu1, std1)
    x2            = np.linspace(s.min(), s.max(), nTrainings)
    y2            = norm.pdf(x2, mu2, std2)
    x3            = np.linspace(dots.min(), dots.max(), nTrainings)
    y3            = norm.pdf(x3, mu3, std3)
    x4            = np.linspace(p1.min(), p1.max(), nTrainings)
    y4            = norm.pdf(x4, mu4, std4)
    x5            = np.linspace(p2.min(), p2.max(), nTrainings)
    y5            = norm.pdf(x5, mu5, std5)
    x6            = np.linspace(deltaY.min(), deltaY.max(), nTrainings)
    y6            = norm.pdf(x6, mu6, std6)
    x7            = np.linspace(strainXY.min(), strainXY.max(), nTrainings)
    y7            = norm.pdf(x7, mu7, std7) 
    x8            = np.linspace(stressXX.min(), stressXX.max(), nTrainings)
    y8            = norm.pdf(x8, mu8, std8)

    
    fontSize1 = 12
    
    data_list = [
        (U, mu1, std1, {"title": "Input, U", "fontdict": {"family": "Times New Roman", "size": fontSize1, "weight": "bold"}}),
        (s, mu2, std2, {"title": "Input, s", "fontdict": {"family": "Times New Roman", "size": fontSize1, "weight": "bold"}}),
        (dots, mu3, std3, {"title": "Input, dots", "fontdict": {"family": "Times New Roman", "size": fontSize1, "weight": "bold"}}),
        (p1, mu4, std4, {"title": "Input, p1", "fontdict": {"family": "Times New Roman", "size": fontSize1, "weight": "bold"}}),
        (p2, mu5, std5, {"title": "Input, p2", "fontdict": {"family": "Times New Roman", "size": fontSize1, "weight": "bold"}}),
        (deltaY, mu6, std6, {"title": "Target, deltay", "fontdict": {"family": "Times New Roman", "size": fontSize1, "weight": "bold"}}),
        (strainXY, mu7, std7, {"title": "Target, deltay", "fontdict": {"family": "Times New Roman", "size": fontSize1, "weight": "bold"}}),
        (stressXX, mu8, std8, {"title": "Target, deltay", "fontdict": {"family": "Times New Roman", "size": fontSize1, "weight": "bold"}})
    ]


    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(11.7, 8.3 / 2))  # A4 landscape
    

    # Adjustments for all plots
    for i in range(2):
        for j in range(4):
            axes[i, j].tick_params(axis='both', which='both', length=4, color='gray', direction='in', labelsize=10)
            

    # Plot 01
    axes[0, 0].plot(x1, y1, color='black', label=r'$U$', linewidth=2)
    axes[0, 0].fill_between(x1, y1, color='blue', alpha=0.2)  
    axes[0, 0].vlines(x=-1, ymin=0, ymax=0.2, linestyle='--', color='black')
    axes[0, 0].vlines(x=1, ymin=0, ymax=0.2, linestyle='--', color='black')
    axes[0, 0].set_xlim(-1.25, 1.25)
    axes[0, 0].set_ylim(0, 0.6)
    axes[0, 0].set_xticks([-1, 1])  
    axes[0, 0].set_yticks([0, 0.6])  
    axes[0, 0].set_xticklabels([r'$-1$', r'$1$'])
    axes[0, 0].set_yticklabels([r'$0$', r'$0.6$'])
    axes[0, 0].minorticks_on()
    axes[0, 0].grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)  
    axes[0, 0].grid(visible=True, which='minor', linestyle='--', linewidth=0.5, alpha=0.7)  
    axes[0, 0].legend(loc='upper right')

    # Plot 02
    axes[0, 1].plot(x2, y2, color='black', label=r'$s$', linewidth=2)
    axes[0, 1].fill_between(x2, y2, color='blue', alpha=0.2)  
    axes[0, 1].vlines(x=0.22, ymin=0, ymax=0.384, linestyle='--', color='black')
    axes[0, 1].vlines(x=0.75, ymin=0, ymax=0.784, linestyle='--', color='black')
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 3)
    axes[0, 1].set_xticks([0.22, 0.75])
    axes[0, 1].set_yticks([0, 3])
    axes[0, 1].set_xticklabels(['0.22', '0.75'])
    axes[0, 1].minorticks_on()
    axes[0, 1].grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    axes[0, 1].grid(visible=True, which='minor', linestyle='--', linewidth=0.5, alpha=0.7)
    axes[0, 1].legend(loc='upper right')

    # Plot 03
    axes[0, 2].plot(x3, y3, color='black', label=r'$\dot{s}$', linewidth=2)
    axes[0, 2].fill_between(x3, y3, color='blue', alpha=0.2)  
    axes[0, 2].vlines(x=-0.932, ymin=0, ymax=0.1458, linestyle='--', color='black')
    axes[0, 2].vlines(x=0.942, ymin=0, ymax=0.113, linestyle='--', color='black')
    axes[0, 2].set_xlim(-1, 1)
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].set_xticks([-0.76, 0.85])
    axes[0, 2].set_yticks([0, 1])
    axes[0, 2].set_xticklabels(['-0.93', '0.94'])
    axes[0, 2].set_yticklabels(['0', '1'])
    axes[0, 2].minorticks_on()
    axes[0, 2].grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    axes[0, 2].grid(visible=True, which='minor', linestyle='--', linewidth=0.5, alpha=0.7)
    axes[0, 2].legend(loc='upper right')

    # Plot 04
    axes[0, 3].plot(x4, y4, color='black', label=r'$p_1$', linewidth=2)
    axes[0, 3].fill_between(x4, y4, color='blue', alpha=0.2)  
    axes[0, 3].vlines(x=0.08, ymin=0, ymax=0.085, linestyle='--', color='black')
    axes[0, 3].vlines(x=0.969, ymin=0, ymax=0.0392, linestyle='--', color='black')
    axes[0, 3].set_xlim(0, 1)
    axes[0, 3].set_ylim(0, 3)
    axes[0, 3].set_xticks([0.08, 0.969])
    axes[0, 3].set_yticks([0, 3])
    axes[0, 3].set_xticklabels(['0.1', '0.97'])
    axes[0, 3].set_yticklabels(['0', '3'])
    axes[0, 3].minorticks_on()
    axes[0, 3].grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    axes[0, 3].grid(visible=True, which='minor', linestyle='--', linewidth=0.5, alpha=0.7)
    axes[0, 3].legend(loc='upper right')

    # Plot 05
    axes[1, 0].plot(x5, y5, color='black', label=r'$p_2$', linewidth=2)
    axes[1, 0].fill_between(x5, y5, color='blue', alpha=0.2)  
    axes[1, 0].vlines(x=0.068, ymin=0, ymax=0.21, linestyle='--', color='black')
    axes[1, 0].vlines(x=0.973, ymin=0, ymax=0.323, linestyle='--', color='black')
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 2.5)
    axes[1, 0].set_xticks([0.068, 0.973])
    axes[1, 0].set_yticks([0, 2.5])
    axes[1, 0].set_xticklabels(['0.1', '0.97'])
    axes[1, 0].set_yticklabels(['0', '2.5'])
    axes[1, 0].minorticks_on()
    axes[1, 0].grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    axes[1, 0].grid(visible=True, which='minor', linestyle='--', linewidth=0.5, alpha=0.7)
    axes[1, 0].legend(loc='upper right')

    # Plot 06
    axes[1, 1].plot(x6, y6, alpha=0.2, color='black', label=r'$\delta_{\mathrm{y}}$', linewidth=2)
    axes[1, 1].fill_between(x6, y6, color='black', alpha=0.2)  
    axes[1, 1].vlines(x=-0.97, ymin=0, ymax=0.0, linestyle='--', color='black')
    axes[1, 1].vlines(x=0.90, ymin=0, ymax=0, linestyle='--', color='black')
    axes[1, 1].set_xlim(-1, 1)
    axes[1, 1].set_ylim(0, 3)
    axes[1, 1].set_xticks([-0.97, 0.9])
    axes[1, 1].set_yticks([0, 3])
    axes[1, 1].set_xticklabels(['-0.97', '0.9'])
    axes[1, 1].set_yticklabels(['0', '3'])
    axes[1, 1].minorticks_on()
    axes[1, 1].grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    axes[1, 1].grid(visible=True, which='minor', linestyle='--', linewidth=0.5, alpha=0.7)
    axes[1, 1].legend(loc='upper right')
    
    # Plot 07
    axes[1, 2].plot(x7, y7, alpha=0.2, color='black', label=r'$\epsilon_{\mathrm{xy}}$', linewidth=2)
    axes[1, 2].fill_between(x7, y7, color='black', alpha=0.2)  
    axes[1, 2].vlines(x=x7[0], ymin=0, ymax=y7[1023], linestyle='--', color='black')
    axes[1, 2].vlines(x=x7[1023], ymin=0, ymax=y7[1023], linestyle='--', color='black')
    axes[1, 2].set_xlim(-1, 1)
    axes[1, 2].set_ylim(0, 3)
    axes[1, 2].set_xticks([x7[0], x7[1023]])
    axes[1, 2].set_yticks([0, 3])
    axes[1, 2].set_xticklabels(['-0.98', '0.95'])
    axes[1, 2].set_yticklabels(['0', '3'])
    axes[1, 2].minorticks_on()
    axes[1, 2].grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    axes[1, 2].grid(visible=True, which='minor', linestyle='--', linewidth=0.5, alpha=0.7)
    axes[1, 2].legend(loc='upper right')
    
    # Plot 08
    axes[1, 3].plot(x8, y8, alpha=0.2, color='black', label=r'$\sigma_{\mathrm{xx}}$', linewidth=2)
    axes[1, 3].fill_between(x8, y8, color='black', alpha=0.2)  
    axes[1, 3].vlines(x=x8[0], ymin=0, ymax=y8[1023], linestyle='--', color='black')
    axes[1, 3].vlines(x=x8[1023], ymin=0, ymax=y8[1023], linestyle='--', color='black')
    axes[1, 3].set_xlim(-1,1)
    axes[1, 3].set_ylim(0, 3)
    axes[1, 3].set_xticks([x8[0], x8[1023]])
    axes[1, 3].set_yticks([0, 3])
    axes[1, 3].set_xticklabels(['-0.98', '0.99'])
    axes[1, 3].set_yticklabels(['0', '3'])
    axes[1, 3].minorticks_on()
    axes[1, 3].grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    axes[1, 3].grid(visible=True, which='minor', linestyle='--', linewidth=0.5, alpha=0.7)
    axes[1, 3].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('solution/Figures/Fig8/DataStats.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
    return