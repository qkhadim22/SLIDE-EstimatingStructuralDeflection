from models.container import *

# Control signal 1
def RandomSignal(self, SLIDEwindow=False, Evaluate=False): 
    
        if not Evaluate:
                ten_percent            = int(0.1 * self.nStepsTotal)  
                segment1               = np.ones(2*ten_percent)               # 20% of nStepsTotal at 1  
                segment2               = -1 * np.ones(2*ten_percent)          # 20% of nStepsTotal at -1
                segment3               = np.zeros(2*ten_percent)
                
                if SLIDEwindow==False: 
                    num_segments = np.random.randint(5, 1*ten_percent) #randomly selecting the number of segments 
                    segment_lengths = np.zeros(num_segments, dtype=int)
                    remaining_length = 2*ten_percent - num_segments
        
                    for i in range(num_segments - 1): 
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
        
                    signal = np.concatenate(segments)
                    
                    def custom_shuffle(array, frequency):
                                            
                        for _ in range(frequency):
                            # Randomly select two indices to swap
                            i, j = np.random.randint(0, self.nStepsTotal , size=2)
                            # Swap elements
                            array[i], array[j] = array[j], array[i]
                            
                    #Swapping frequency
                    num_swaps = np.random.randint(1, 0.1*self.nStepsTotal)  #self.nStepsTotal
                    custom_shuffle(signal, num_swaps)
                else:
                    
                    seg1     = int(0.1 * self.nStepsTotal)
                    seg2     = int(0.9 * self.nStepsTotal)

                    segments = [np.ones(seg1), np.zeros(seg2)]
                    signal         = np.concatenate(segments)
        else:
                ten_percent  = int(0.1 * self.nStepsTotal)
                five_percent = int(0.05 * self.nStepsTotal)
                two_percent  = int(0.02 * self.nStepsTotal)
                one_percent  = int(0.01 * self.nStepsTotal)
                            
                segment1     = np.zeros(ten_percent)        # 2% steps at zero.
                segment2     = np.ones(ten_percent)        # 5% steps at 1.
                segment3     = np.zeros(ten_percent)       # 5% steps at zero.
                segment4     = -1*np.ones(ten_percent+one_percent)     # 5% steps at -1.
                            
                ramp_up_1    = np.linspace(0, 0.6, five_percent)  # 5% ramp between 0 and 0.6.
                ramp_down_1  = np.linspace(0.6, 0, five_percent)  # 5% ramp between 0.6 and 0.
                static1      = 0.6*np.ones(2*two_percent)         # 4% steps at 0.6.
                segment5     = np.concatenate((ramp_up_1,static1, ramp_down_1)) 
                            
                segment6     = np.random.uniform(low=-0.8, high=0.8, size=two_percent)
                segment6     = np.repeat(segment6, 5)
                segment6     = segment6[(segment6 != -1) & (segment6 != 0) & (segment6 != 1)]
                ramp_up_2    = np.linspace(0, -1, five_percent) # 5% ramp between 0 and -0.8.
                static2      = -1*np.ones(2*two_percent)        # 4% steps at 0.6.
                ramp_down_2  = np.linspace(-1, 0, five_percent) # 5% ramp between -0.8 and 0.
                segment7     = np.concatenate((ramp_up_2,static2, ramp_down_2)) 
                segment8     = np.zeros(ten_percent)    # 10% steps at zero.
                segment9     = np.ones(five_percent)     # 10% steps at 1.
                segment10    = -1*np.ones(ten_percent)  # 10% steps at -1.
                segment11    = np.zeros(five_percent+one_percent)    # 10% steps at zero.
                segments    = [segment3,-segment9, segment11, segment9,segment11,-segment9,
                                            segment1,segment9, segment3, segment7, segment8, -segment7]
                signal      = np.concatenate(segments)
                
        return signal
    


# Control signal 1
def uref(t):
    
    
    #Lets comment this part and call 
    Lifting_Time_Start_1    = 1.00          # Start of lifting mass, m
    Lifting_Time_End_1      = 1.5          # End of lifting mass, m
    Lowering_Time_Start_1   = 2.1           # Start of lowering mass, m
    Lowering_Time_End_1     = 2.6           # End of lowering mass, m
    Lowering_Time_Start_2   = 3.2          # Start of lowering mass, m
    Lowering_Time_End_2     = 3.7            # End of lowering mass, m
    # Lowering_Time_Start_3   = 4.7         # Start of lowering mass, m
    # Lowering_Time_End_3     = 5.2           # End of lowering mass, m
    # Lowering_Time_Start_4   = 6.2         # Start of lowering mass, m
    # Lowering_Time_End_4     = 19           # End of lowering mass, m

    if Lifting_Time_Start_1 <= t < Lifting_Time_End_1:
        u = -1
    elif Lowering_Time_Start_1 <= t < Lowering_Time_End_1:
        u = 1
    elif Lowering_Time_Start_2 <= t < Lowering_Time_End_2:
         u = -1
    # elif Lowering_Time_Start_3 <= t < Lowering_Time_End_3:
    #      u = 1
    # elif Lowering_Time_Start_4 <= t < Lowering_Time_End_4:
    #      u = -1
    else:
        u = 0
    
    return u

# Control signal 1
def uref_1(t):
    
    
    #Lets comment this part and call 
    Lifting_Time_Start_1    = 2          # Start of lifting mass, m
    Lifting_Time_End_1      = 6          # End of lifting mass, m
    Lowering_Time_Start_1   = 8           # Start of lowering mass, m
    Lowering_Time_End_1     = 10.7     # End of lowering mass, m
    Lowering_Time_Start_2   = 13          # Start of lowering mass, m
    Lowering_Time_End_2     = 16            # End of lowering mass, m
    Lowering_Time_Start_3   = 17        # Start of lowering mass, m
    Lowering_Time_End_3     = 18          # End of lowering mass, m
    Lowering_Time_Start_4   = 18.5         # Start of lowering mass, m
    Lowering_Time_End_4     = 19           # End of lowering mass, m

    if Lifting_Time_Start_1 <= t < Lifting_Time_End_1:
        u = 1
    elif Lowering_Time_Start_1 <= t < Lowering_Time_End_1:
        u = -1
    elif Lowering_Time_Start_2 <= t < Lowering_Time_End_2:
         u = 1
    elif Lowering_Time_Start_3 <= t < Lowering_Time_End_3:
         u = -1
    # elif Lowering_Time_Start_4 <= t < Lowering_Time_End_4:
    #      u = -10
    else:
        u = 0
    
    return u

# Control signal 2
def uref_2(t):
    
   Lifting_Time_Start_1  = 1.0          # Start of lifting mass, m
   Lifting_Time_End_1    = 3.0            # End of lifting mass, m
   Lowering_Time_Start_1 = 4.0         # Start of lowering mass, m
   Lowering_Time_End_1   = 6.6           # End of lowering mass, m
   Lowering_Time_Start_2 = 8         # Start of lowering mass, m
   Lowering_Time_End_2   = 10           # End of lowering mass, m
   Lowering_Time_Start_3 = 12         # Start of lowering mass, m
   Lowering_Time_End_3 = 15          # End of lowering mass, m
   Lowering_Time_Start_4 = 16         # Start of lowering mass, m
   Lowering_Time_End_4 = 18          # End of lowering mass, m

   if Lifting_Time_Start_1 <= t < Lifting_Time_End_1:
       u = 1
   elif Lowering_Time_Start_1 <= t < Lowering_Time_End_1:
       u = -1
   elif Lowering_Time_Start_2 <= t < Lowering_Time_End_2:
       u = 1
   elif Lowering_Time_Start_3 <= t < Lowering_Time_End_3:
       u = -1
   elif Lowering_Time_Start_4 <= t < Lowering_Time_End_4:
         u = 1
   else:
       u = 0
    #u = ExpData[t,5]

   return u

    
def PlottingFunc ( timeVecLong, ControlSig):
    
    
    #timeVecLong,ControlSig
    # Fig01: Control signal
    a40_width_inches, a40_height_inches = 8.3, 11.7/2 
    fig, ax = plt.subplots(figsize=(a40_width_inches, a40_height_inches))
    
    # Plot control signal
    ax.plot(timeVecLong, ControlSig, color='k')
    ax.axvspan(1, 1.5, ymin=0, ymax=1, color='thistle', alpha=0.2, label='Valve spool open')
    ax.axvspan(2.1, 2.6, ymin=0, ymax=1, color='lawngreen', alpha=0.2, label='Valve spool open')
    ax.axvspan(3.2, 3.7, ymin=0, ymax=1, color='thistle', alpha=0.2, label='Valve spool open')
    ax.axvspan(4.7, 5.2, ymin=0, ymax=1, color='lawngreen', alpha=0.2, label='Valve spool open')
    ax.axvspan(6.7, 7.1, ymin=0, ymax=1, color='thistle', alpha=0.2, label='Valve spool open')
    ax.axvspan(9.1, 9.5, ymin=0, ymax=1, color='lawngreen', alpha=0.2, label='Valve spool open')
    ax.annotate('Boom up', xy=(5.95, 1.0565), horizontalalignment='center', color='black', fontsize=12, 
                        bbox=dict(facecolor='white', edgecolor='black', pad=4.0))
    
    ax.annotate('Boom down', xy=(5.08, -1.12), horizontalalignment='center', color='black', fontsize=12, 
                        bbox=dict(facecolor='white', edgecolor='black', pad=4.0))
    ax.annotate('Boom still', xy=(7.9, 0.45), horizontalalignment='center', color='black', fontsize=12, 
                        bbox=dict(facecolor='white', edgecolor='black', pad=4.0))
    ax.annotate('',  # No text, just the arrow
                            xy=( 4.25, -1.12),  # Arrow points to the center of the rectangle
                        xytext=(3.3, -0.99),  # Starting point of the arrow at text location
                        arrowprops=dict(arrowstyle="->", color='black'))
    ax.annotate('',  # No text, just the arrow
                        xy=(5.32, 1.07),  # Arrow points to the center of the rectangle
                        xytext=(4.95,1),  # Starting point of the arrow at text location
                        arrowprops=dict(arrowstyle="->", color='black'))
    ax.annotate('',  # No text, just the arrow
                        xy=(7.9, 0.39),  # Arrow points to the center of the rectangle
                        xytext=(7.9, 0),  # Starting point of the arrow at text location
                        arrowprops=dict(arrowstyle="->", color='black'))
    ax.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': 20})
    ax.set_ylabel(r'Control signal, V', fontdict={'family': 'Times New Roman', 'size': 20})
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.2, 1.2)
    x_ticks = [0, 5, 10]
    y_ticks = [-1,0,1]
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([r'$0$',r'$5$', r'$10$'])
    ax.set_yticklabels([r'$U_{min}$', r'$0$', r'$U_{max}$'])  
    ax.tick_params(axis='both', labelsize=20)  # Change 8 to the desired font size
    # Enable grid and legend
    ax.grid( linestyle='--', linewidth=0.5)
    # Adjust layout and save the figure
    plt.tight_layout()
    # if Patu:
    #     plt.savefig(f'Figures/Results/Patu/Evaluation_controlsignal_{LiftLoad}kg.svg', format='png', dpi=300)
    # else:
    plt.savefig(f'solution/Figures/Results/LiftBoom/Evaluation_Control.pdf', format='pdf', bbox_inches='tight')
    plt.show()
                
    return 

def PlottingFunc2 ( timeVecLong, ControlSig1, ControlSig2):
    
    
    #timeVecLong,ControlSig
    # Fig01: Control signal
    a40_width_inches, a40_height_inches = 8.3, 11.7/2 
    fig, ax = plt.subplots(figsize=(a40_width_inches, a40_height_inches))    
    # Plot control signal
    ax.plot(timeVecLong, ControlSig1, color='k')
    ax.axvspan(1, 1.5, ymin=0, ymax=1, color='thistle', alpha=0.2, label='Valve spool open')
    ax.axvspan(2.1, 2.6, ymin=0, ymax=1, color='lawngreen', alpha=0.2, label='Valve spool open')
    ax.axvspan(3.2, 3.7, ymin=0, ymax=1, color='thistle', alpha=0.2, label='Valve spool open')
    ax.axvspan(4.7, 5.2, ymin=0, ymax=1, color='lawngreen', alpha=0.2, label='Valve spool open')
    ax.axvspan(6.7, 7.1, ymin=0, ymax=1, color='thistle', alpha=0.2, label='Valve spool open')
    ax.axvspan(9.1, 9.5, ymin=0, ymax=1, color='lawngreen', alpha=0.2, label='Valve spool open')
    # ax.annotate('Boom up', xy=(5.95, 1.0565), horizontalalignment='center', color='black', fontsize=12, 
    #                     bbox=dict(facecolor='white', edgecolor='black', pad=4.0))
    
    # ax.annotate('Boom down', xy=(5.08, -1.12), horizontalalignment='center', color='black', fontsize=12, 
    #                     bbox=dict(facecolor='white', edgecolor='black', pad=4.0))
    # ax.annotate('Boom still', xy=(7.9, 0.45), horizontalalignment='center', color='black', fontsize=12, 
    #                     bbox=dict(facecolor='white', edgecolor='black', pad=4.0))
    # ax.annotate('',  # No text, just the arrow
    #                         xy=( 4.25, -1.12),  # Arrow points to the center of the rectangle
    #                     xytext=(3.3, -0.99),  # Starting point of the arrow at text location
    #                     arrowprops=dict(arrowstyle="->", color='black'))
    # ax.annotate('',  # No text, just the arrow
    #                     xy=(5.32, 1.07),  # Arrow points to the center of the rectangle
    #                     xytext=(4.95,1),  # Starting point of the arrow at text location
    #                     arrowprops=dict(arrowstyle="->", color='black'))
    # ax.annotate('',  # No text, just the arrow
    #                     xy=(7.9, 0.39),  # Arrow points to the center of the rectangle
    #                     xytext=(7.9, 0),  # Starting point of the arrow at text location
    #                     arrowprops=dict(arrowstyle="->", color='black'))
    ax.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': 20})
    ax.set_ylabel(r'Control signal, V', fontdict={'family': 'Times New Roman', 'size': 20})
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.2, 1.2)
    x_ticks = [0, 5, 10]
    y_ticks = [-1,0,1]
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([r'$0$',r'$5$', r'$10$'])
    ax.set_yticklabels([r'$U_{min}$', r'$0$', r'$U_{max}$'])  
    ax.tick_params(axis='both', labelsize=20)  # Change 8 to the desired font size
    # Enable grid and legend
    ax.grid( linestyle='--', linewidth=0.5)
    # Adjust layout and save the figure
    plt.tight_layout()
    # if Patu:
    #     plt.savefig(f'Figures/Results/Patu/Evaluation_controlsignal_{LiftLoad}kg.svg', format='png', dpi=300)
    # else:
    plt.savefig(f'solution/Figures/Results/Patu/Evaluation_Control1.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    # Plot control signal
    fig1, ax1 = plt.subplots(figsize=(a40_width_inches, a40_height_inches))
    ax1.plot(timeVecLong, ControlSig2, color='k')
    ax1.axvspan(1, 1.5, ymin=0, ymax=1, color='lawngreen', alpha=0.2, label='Valve spool open') #thistle,lawngreen
    ax1.axvspan(2.1, 2.6, ymin=0, ymax=1, color='thistle', alpha=0.2, label='Valve spool open')
    ax1.axvspan(3.2, 3.7, ymin=0, ymax=1, color='lawngreen', alpha=0.2, label='Valve spool open')
    ax1.axvspan(4.7, 5.2, ymin=0, ymax=1, color='thistle', alpha=0.2, label='Valve spool open')
    ax1.axvspan(6.7, 7.1, ymin=0, ymax=1, color='lawngreen', alpha=0.2, label='Valve spool open')
    ax1.axvspan(9.1, 9.5, ymin=0, ymax=1, color='thistle', alpha=0.2, label='Valve spool open')
    # ax1.annotate('Boom up', xy=(5.95, 1.0565), horizontalalignment='center', color='black', fontsize=12, 
    #                     bbox=dict(facecolor='white', edgecolor='black', pad=4.0))
    
    # ax1.annotate('Boom down', xy=(5.08, -1.12), horizontalalignment='center', color='black', fontsize=12, 
    #                     bbox=dict(facecolor='white', edgecolor='black', pad=4.0))
    # ax1.annotate('Boom still', xy=(7.9, 0.45), horizontalalignment='center', color='black', fontsize=12, 
    #                     bbox=dict(facecolor='white', edgecolor='black', pad=4.0))
    # ax1.annotate('',  # No text, just the arrow
    #                         xy=( 4.25, -1.12),  # Arrow points to the center of the rectangle
    #                     xytext=(3.3, -0.99),  # Starting point of the arrow at text location
    #                     arrowprops=dict(arrowstyle="->", color='black'))
    # ax1.annotate('',  # No text, just the arrow
    #                     xy=(4.62, 1.07),  # Arrow points to the center of the rectangle
    #                     xytext=(3.3,1),  # Starting point of the arrow at text location
    #                     arrowprops=dict(arrowstyle="->", color='black'))
    # ax1.annotate('',  # No text, just the arrow
    #                     xy=(7.9, 0.39),  # Arrow points to the center of the rectangle
    #                     xytext=(7.9, 0),  # Starting point of the arrow at text location
    #                     arrowprops=dict(arrowstyle="->", color='black'))
    ax1.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': 20})
    ax1.set_ylabel(r'Control signal, V', fontdict={'family': 'Times New Roman', 'size': 20})
    ax1.set_xlim(0, 10)
    ax1.set_ylim(-1.2, 1.2)
    x_ticks = [0, 5, 10]
    y_ticks = [-1,0,1]
    ax1.set_xticks(x_ticks)
    ax1.set_yticks(y_ticks)
    ax1.set_xticklabels([r'$0$',r'$5$', r'$10$'])
    ax1.set_yticklabels([r'$U_{min}$', r'$0$', r'$U_{max}$'])  
    ax1.tick_params(axis='both', labelsize=20)  # Change 8 to the desired font size
    # Enable grid and legend
    ax1.grid( linestyle='--', linewidth=0.5)
    # Adjust layout and save the figure
    plt.tight_layout()
    # if Patu:
    #     plt.savefig(f'Figures/Results/Patu/Evaluation_controlsignal_{LiftLoad}kg.svg', format='png', dpi=300)
    # else:
    plt.savefig(f'solution/Figures/Results/Patu/Evaluation_Control2.pdf', format='pdf', bbox_inches='tight')
    plt.show()
                
    return 
   

def Plot():
    data  = np.load("solution/data/LiftBoom/FFNT32-8s200t1E2.1e+11Density7.8e+03Load0.npy", allow_pickle=True)[()]


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

    Time = np.linspace(0,1,200)

    U1   = X_train[0, 0, 0:200]
    U2   = X_train[1, 0, 0:200]
    U3   = X_train[2, 0, 0:200]
    U4   = X_train[3, 0, 0:200]
    U5   = X_train[4, 0, 0:200]
    U6   = X_train[5, 0, 0:200]
    U7   = X_train[6, 0, 0:200]
    U8   = X_train[7, 0, 0:200]


    fontSize1 = 22
    fontSize2 = 20

    # Fig01: Control signal
    a40_width_inches, a40_height_inches = 8.3 / 2, (11.7 / 3)   
    fig, ax = plt.subplots(figsize=(a40_width_inches, a40_height_inches))
    ax.plot(Time, U1, color='blue')
    ax.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': fontSize1})
    ax.set_ylabel(r'Control signal, V', fontdict={'family': 'Times New Roman', 'size': fontSize1})
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.2, 1.2)
    x_ticks = [0, 0.5, 1]
    y_ticks = [-1,0,1]
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([r'$0$', r'$0.5$',  r'$1$'])
    ax.set_yticklabels([r'$U_{min}$', r'$0$', r'$U_{max}$'])                   
    ax.tick_params(axis='both', labelsize=fontSize2)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('solution/Figures/Fig6/controlsignal1.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    # Plot control signal2
    # Fig01: Control signal
    a40_width_inches, a40_height_inches = 8.3 / 2, (11.7 / 3)   
    fig1, ax1= plt.subplots(figsize=(a40_width_inches, a40_height_inches))
    ax1.plot(Time, U2, color='red')
    ax1.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': fontSize1})
    ax1.set_ylabel(r'Control signal, V', fontdict={'family': 'Times New Roman', 'size': fontSize1})
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-1.2, 1.2)
    x_ticks = [0, 0.5, 1]
    y_ticks = [-1,0,1]
    ax1.set_xticks(x_ticks)
    ax1.set_yticks(y_ticks)
    ax1.set_xticklabels([r'$0$', r'$0.5$',  r'$1$'])
    ax1.set_yticklabels([r'$U_{min}$', r'$0$', r'$U_{max}$'])                   
    ax1.tick_params(axis='both', labelsize=fontSize2)
    ax1.grid(True)
    plt.tight_layout()
    plt.savefig('solution/Figures/Fig6/controlsignal2.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
    # Plot control signal3
    fig3, ax3= plt.subplots(figsize=(a40_width_inches, a40_height_inches))
    ax3.plot(Time, U3, color='green')
    ax3.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': fontSize1})
    ax3.set_ylabel(r'Control signal, V', fontdict={'family': 'Times New Roman', 'size': fontSize1})
    ax3.set_xlim(0, 1)
    ax3.set_ylim(-1.2, 1.2)
    x_ticks = [0, 0.5, 1]
    y_ticks = [-1,0,1]
    ax3.set_xticks(x_ticks)
    ax3.set_yticks(y_ticks)
    ax3.set_xticklabels([r'$0$', r'$0.5$',  r'$1$'])
    ax3.set_yticklabels([r'$U_{min}$', r'$0$', r'$U_{max}$'])                   
    ax3.tick_params(axis='both', labelsize=fontSize2)
    ax3.grid(True)
    plt.tight_layout()
    plt.savefig('solution/Figures/Fig6/controlsignal3.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    # Plot control signal4
    fig4, ax4= plt.subplots(figsize=(a40_width_inches, a40_height_inches))
    ax4.plot(Time, U4, color='magenta')
    ax4.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': fontSize1})
    ax4.set_ylabel(r'Control signal, V', fontdict={'family': 'Times New Roman', 'size': fontSize1})
    ax4.set_xlim(0, 1)
    ax4.set_ylim(-1.2, 1.2)
    x_ticks = [0, 0.5, 1]
    y_ticks = [-1,0,1]
    ax4.set_xticks(x_ticks)
    ax4.set_yticks(y_ticks)
    ax4.set_xticklabels([r'$0$', r'$0.5$',  r'$1$'])
    ax4.set_yticklabels([r'$U_{min}$', r'$0$', r'$U_{max}$'])                   
    ax4.tick_params(axis='both', labelsize=fontSize2)
    ax4.grid(True)
    plt.tight_layout()
    plt.savefig('solution/Figures/Fig6/controlsignal4.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    # Plot control signal5
    fig5, ax5= plt.subplots(figsize=(a40_width_inches, a40_height_inches))
    ax5.plot(Time, U5, color='lightsalmon')
    ax5.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': fontSize1})
    ax5.set_ylabel(r'Control signal, V', fontdict={'family': 'Times New Roman', 'size': fontSize1})
    ax5.set_xlim(0, 1)
    ax5.set_ylim(-1.2, 1.2)
    x_ticks = [0, 0.5, 1]
    y_ticks = [-1,0,1]
    ax5.set_xticks(x_ticks)
    ax5.set_yticks(y_ticks)
    ax5.set_xticklabels([r'$0$', r'$0.5$',  r'$1$'])
    ax5.set_yticklabels([r'$U_{min}$', r'$0$', r'$U_{max}$'])                   
    ax5.tick_params(axis='both', labelsize=fontSize2)
    ax5.grid(True)
    plt.tight_layout()
    plt.savefig('solution/Figures/Fig6/controlsignal5.pdf', format='pdf', bbox_inches='tight')
    plt.show()


    # Plot control signal6
    fig6, ax6= plt.subplots(figsize=(a40_width_inches, a40_height_inches))
    ax6.plot(Time, U6, color='cyan')
    ax6.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': fontSize1})
    ax6.set_ylabel(r'Control signal, V', fontdict={'family': 'Times New Roman', 'size': fontSize1})
    ax6.set_xlim(0, 1)
    ax6.set_ylim(-1.2, 1.2)
    x_ticks = [0, 0.5, 1]
    y_ticks = [-1,0,1]
    ax6.set_xticks(x_ticks)
    ax6.set_yticks(y_ticks)
    ax6.set_xticklabels([r'$0$', r'$0.5$',  r'$1$'])
    ax6.set_yticklabels([r'$U_{min}$', r'$0$', r'$U_{max}$'])                   
    ax6.tick_params(axis='both', labelsize=fontSize2)
    ax6.grid(True)
    plt.tight_layout()
    plt.savefig('solution/Figures/Fig6/controlsignal6.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    # Plot control signal7
    fig7, ax7= plt.subplots(figsize=(a40_width_inches, a40_height_inches))
    ax7.plot(Time, U7, color='orange')
    ax7.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': fontSize1})
    ax7.set_ylabel(r'Control signal, V', fontdict={'family': 'Times New Roman', 'size': fontSize1})
    ax7.set_xlim(0, 1)
    ax7.set_ylim(-1.2, 1.2)
    x_ticks = [0, 0.5, 1]
    y_ticks = [-1,0,1]
    ax7.set_xticks(x_ticks)
    ax7.set_yticks(y_ticks)
    ax7.set_xticklabels([r'$0$', r'$0.5$',  r'$1$'])
    ax7.set_yticklabels([r'$U_{min}$', r'$0$', r'$U_{max}$'])                   
    ax7.tick_params(axis='both', labelsize=fontSize2)
    ax7.grid(True)
    plt.tight_layout()
    plt.savefig('solution/Figures/Fig6/controlsignal7.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    # Plot control signal8
    fig8, ax8= plt.subplots(figsize=(a40_width_inches, a40_height_inches))
    ax8.plot(Time, U8, color='black')
    ax8.set_xlabel('Time, s', fontdict={'family': 'Times New Roman', 'size': fontSize1})
    ax8.set_ylabel(r'Control signal, V', fontdict={'family': 'Times New Roman', 'size': fontSize1})
    ax8.set_xlim(0, 1)
    ax8.set_ylim(-1.2, 1.2)
    x_ticks = [0, 0.5, 1]
    y_ticks = [-1,0,1]
    ax8.set_xticks(x_ticks)
    ax8.set_yticks(y_ticks)
    ax8.set_xticklabels([r'$0$', r'$0.5$',  r'$1$'])
    ax8.set_yticklabels([r'$U_{min}$', r'$0$', r'$U_{max}$'])                   
    ax8.tick_params(axis='both', labelsize=fontSize2)
    ax8.grid(True)
    plt.tight_layout()
    plt.savefig('solution/Figures/Fig6/controlsignal8.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    # num_samples = X_train.shape[0]

    # # Define the number of subplot columns and rows
    # # %%
    # cols = 3  # Example: 3 columns

    # rows = num_samples // cols + (num_samples % cols > 0)  # Calculate rows needed based on number of samples

    # # Create a large figure to hold all subplots
    # plt.figure(figsize=(cols * 6, rows * 4))  # Adjust size as needed

    # for i in range(num_samples - 1, -1, -1):
    #     ax = plt.subplot(rows, cols, num_samples - i)  # Create subplot for each sample
    #     ax.plot(Time, X_train[i, 0, :200], 'k')  # Plot the first 200 data points of the first feature
    #     ax.set_ylabel("Spool position, V")
    #     ax.set_xlabel("Time, s")
    #     ax.grid(True)

    #     ax.set_xlim(0, 1)  # Set the x-axis limits
    #     ax.set_ylim(-1.2, 1.2)  # Set the y-axis limits

    # # Adjust layout to prevent overlap
    # plt.tight_layout()

    # # Display all plots
    # plt.show()


    #++++++++++++Fig01++++++++++++++++++++++++++
    # Assuming Time, Y_train, damped_time, and damped_Steps are defined

    g, ax = plt.subplots(figsize=(8, 6))

    # Plotting data
    ax.plot(Time, 2.5*Y_train[0, 0:200], color='grey', linewidth=2)

    # Setting labels, limits, and grids
    ax.set_xlabel('Time, s', fontsize=24)
    ax.set_ylabel(r'Deflection, mm', fontsize=24)
    ax.set_xlim([0, 1])
    ax.set_ylim([-1.25, 1.25])
    ax.grid(True)

    # Correctly setting x-ticks and x-tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.show()


    #++++++++++++++++++++++++++

    from matplotlib.patches import Rectangle
    damped_time = 0.43
    damped_Steps = 86

    a40_width_inches, a40_height_inches = 8.3 / 2, (11.7 / 4)   
    fig, ax = plt.subplots(figsize=(a40_width_inches, a40_height_inches))

    # Plotting data
    ax.plot(Time, 1.5*Y_train[0, 0:200], color='gray', linewidth=2)
    ax.axvspan(0, damped_time, ymin=0, ymax=1, color='lightgreen', alpha=0.2, label='Valve spool open')
    ax.axvline(damped_time, color='red', alpha=0.5, linestyle='-.', linewidth=2, label='Midpoint of Open')
    # Text annotations with raw strings for LaTeX
    ax.text(0.12, 0.525, r'$\mathbf{\mathcal{X}}$', horizontalalignment='center', color='black', fontsize=11, 
                 bbox=dict(facecolor='white', edgecolor='black', pad=4.0))
    ax.annotate('',                      # no text
                xy=(0.431, 0.765),         # head of the arrow (end point)
                xytext=(-0.001, 0.765),        # tail of the arrow (start point)
                arrowprops=dict(arrowstyle="<->", color='black', lw=2))
    # Text close to the rectangle
    ax.text(damped_time + 0.08, -0.825, r'$\mathbf{\mathcal{Y}}$', horizontalalignment='center', color='black', fontsize=11, 
             bbox=dict(facecolor='white', edgecolor='black', pad=4.0))

    # Adding a rectangle
    center_x, center_y = 0.43, -0.48  # These replace your damped_time, -0.52
    width, height = 0.015, 0.06  # Size of the rectangle, adjust as necessary

    rectangle = Rectangle((center_x - width / 2, center_y - height / 2), width, height,
                          edgecolor='black', facecolor='black', fill=True)
    ax.add_patch(rectangle)

    # Arrow pointing from text to rectangle
    ax.annotate('',  # No text, just the arrow
                xy=(center_x, center_y),  # Arrow points to the center of the rectangle
                xytext=(damped_time + 0.08, -0.705),  # Starting point of the arrow at text location
                arrowprops=dict(arrowstyle="->", color='black'))

    # Setting labels, limits, and grids
    ax.set_xlabel('Time, s', fontsize=11)
    ax.set_ylabel(r'${\delta}_\mathrm{y}$, mm', fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([-1.0, 1.0])
    ax.grid(True)
    # Tick settings
    ax.set_xticks([0, 0.25, 0.43, 0.75, 1])
    ax.set_yticks([-1, -0.5,   0,  0.5,    1])
    ax.tick_params(axis='both', which='both', labelsize=11)
    ax.set_xticklabels([r'$0$', r'$0.25$', r'${\mathrm{t}}_{\mathrm{d}}$', r'$0.75$', r'$1$'])
    ax.set_yticklabels([r'$-1$', r'$-0.5$', r'$0$', r'$0.5$', r'$1$']) 
    plt.tight_layout()  
    # Save and display the plot
    plt.savefig('solution/Controldata/SingleStep_Predictor.png', format='png', dpi=300)
    plt.show()



    # Create the plot using the object-oriented API
    fig, ax = plt.subplots(figsize=(a40_width_inches, a40_height_inches))
    ax.plot(Time, 1.5*Y_train[0, 0:200], color='gray', linewidth=2)
    ax.axvspan(0, damped_time+39*5e-3, ymin=0, ymax=1, color='lightgreen', alpha=0.2, label='Valve spool open')
    ax.axvline(damped_time, color='red', alpha=0.5, linestyle='-.', linewidth=2, label='Midpoint of Open')
    # Text annotations with raw strings for LaTeX
    ax.text(0.25, 0.525, r'$\mathbf{\mathcal{X}}$', horizontalalignment='center', color='black', fontsize=10, 
                 bbox=dict(facecolor='white', edgecolor='black', pad=4.0))
    ax.annotate('',                      # no text
                xy=(0.64, 0.765),         # head of the arrow (end point)
                xytext=(-0.001, 0.765),        # tail of the arrow (start point)
                arrowprops=dict(arrowstyle="<->", color='black', lw=2))

    ax.text(damped_time + 0.11, -0.87, r'$\mathbf{\mathcal{Y}}$', horizontalalignment='center', color='black', fontsize=10, 
             bbox=dict(facecolor='white', edgecolor='black', pad=4.0))

    start_index = damped_Steps
    end_index = damped_Steps+39
    ax.plot(Time[start_index:end_index], 1.5*Y_train[0,start_index:end_index], color='fuchsia', label='Highlighted Segment')

    ax.axvline(0.6331658291457286, color='red', alpha=0.5, linestyle='-.', linewidth=2, label='Midpoint of Open')

    # Adding a rectangle
    center_x, center_y = 0.43, -0.48
    center_x1, center_y1 = damped_time - 39 * 5e-3, -0.0383559916247686-0.3
    width, height = 0.015, 0.06  # Size of the rectangle, adjust as necessary

    rectangle1 = Rectangle((center_x - width / 2, center_y - height / 2), width, height,
                          edgecolor='black', facecolor='black', fill=True)

    rectangle2 = Rectangle((center_x1 - width / 2, center_y1 - height / 2), width, height,
                          edgecolor='black', facecolor='black', fill=True)

    ax.add_patch(rectangle1)
    ax.add_patch(rectangle2)
    ax.annotate('',                      # no text
                xy=(0.43-0.005, -0.62),         # head of the arrow (end point)
                xytext=(0.64, -0.62),        # tail of the arrow (start point)
                arrowprops=dict(arrowstyle="<->", color='black', lw=2))

    # Setting labels, limits, and grids
    ax.set_xlabel('Time, s', fontsize=10)
    ax.set_ylabel(r'${\delta}_\mathrm{y}$, mm', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([-1.0, 1.0])
    ax.grid(True)
    # Tick settings
    ax.set_xticks([0, 0.25, 0.43,0.64 , 1])
    ax.set_yticks([-1, -0.5,   0,  0.5,    1])
    ax.tick_params(axis='both', which='both', labelsize=10)
    ax.set_xticklabels([r'$0$', r'$0.25$', r'${\mathrm{t}}_{\mathrm{d}}$',  r'${\mathrm{t}}_{\mathrm{d}}+{\mathrm{k}}$', r'$1$'])
    ax.set_yticklabels([r'$-1$', r'$-0.5$', r'$0$', r'$0.5$', r'$1$']) 
    plt.tight_layout()  
    plt.savefig('solution/Controldata/MultiStep_Predictor.png', format='png', dpi=300)
    plt.show()
    
    return


