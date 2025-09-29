# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 17:33:52 2025

@author: Peter Manzl, Universität Innsbruck
"""
import exudyn as exu
from exudyn.utilities import *
import sys
import numpy as np
import time
import os
import copy

if not ('../' in sys.path): 
    sys.path.append('../')

from Models.Container import *
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
from exudyn.processing import ProcessParameterList

#%% 
os.environ['OMP_NUM_THREADS'] = '1' 
flagRealData = True

timeStep    = 5e-3              #Simulation time step: Change it as desired.
T           = 10               #Time period
ns          = int(T/timeStep)       
angleInit1  = np.deg2rad(14.6)  #Lift boom angle               
angleInit2  = np.deg2rad(-58.8) #Tilt boom angle 

Plotting    =  True

useCUDA             = torch.cuda.is_available()
useCUDA             = True              #CUDA support helps for fully connected networks > 256

if __name__ == '__main__':              #include this to enable parallel processing
    print('pytorch cuda=',useCUDA)

dataPath    = 'solution/data' + str(T) + '-' + 's' + str(ns) + 'Steps' 

npyFile = 'solution/data/LiftBoom/FFNT32-8s200t1E2.1e+11Density7.8e+03Load0'

data    = np.load(npyFile+'.npy', allow_pickle=True).item()

nTotal = 200 # steps
nDamped = 48 # Peter: assumption, not sure ?? --> todo
# calculated: 83

# unflattened_array = inputsTest[0].reshape((200,5), order='C')
input_size = 5 # Number of input features (5 channels)


for key in ['inputsTraining', 'inputsTest']: 
    data[key] = data[key].reshape((-1, 200, 5), order='F')

for key in ['targetsTraining', 'targetsTest']: 
    data[key] = data[key].reshape((-1, 200,3), order='F')

# order F seems to be the correct order
# with order C learning does not work at all 
locals().update(data) # add keys of data to (local) accessible variables




#%%
# Define the LSTM-based sequence-to-sequence regression model
class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Seq2SeqLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.type = 'LSTM'

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Activation functions
        self.tanh1 = nn.Tanh()
        self.linear1 = nn.Identity()  # Linear = no change
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Identity()  # Linear = no change
        self.tanh2 = nn.Tanh()
        
        # Final FC layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM
        out, _ = self.lstm(x, (h0, c0))  # Shape: (batch_size, seq_len, hidden_size)

        # TLSLT activations
        out = self.tanh1(out)      # T
        out = self.linear1(out)    # L
        out = self.sigmoid(out)    # S
        out = self.linear2(out)    # L
        out = self.tanh2(out)      # T

        # Output layer
        out = self.fc(out)         # Final linear map
        return out

    
# Define the RNN-based sequence-to-sequence regression model
class Seq2SeqRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Seq2SeqRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.type = 'RNN'

        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # Activation functions (TLSLT)
        self.tanh1 = nn.Tanh()       # T
        self.linear1 = nn.Identity() # L
        self.sigmoid = nn.Sigmoid()  # S
        self.linear2 = nn.Identity() # L
        self.tanh2 = nn.Tanh()       # T

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pass through RNN layer
        out, _ = self.rnn(x, h0)  # (batch_size, seq_len, hidden_size)

        # Apply TLSLT activations
        out = self.tanh1(out)      # T
        out = self.linear1(out)    # L
        out = self.sigmoid(out)    # S
        out = self.linear2(out)    # L
        out = self.tanh2(out)      # T

        # Fully connected output layer
        out = self.fc(out)         # (batch_size, seq_len, output_size)
        return out


# Define the CNN-based sequence-to-sequence regression model
class Seq2SeqCNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, kernel_size=3):
        super(Seq2SeqCNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.type = 'CNN'
        
        # Convolutional layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = input_size if i == 0 else hidden_size
            self.convs.append(
                nn.Conv1d(in_channels, hidden_size, kernel_size, padding=kernel_size // 2)
            )
        
        # Activation functions
        self.tanh1 = nn.Tanh()
        self.linear1 = nn.Identity()   # Linear = no change
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Identity()   # Linear = no change
        self.tanh2 = nn.Tanh()
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_size)
        x = x.permute(0, 2, 1)  # (batch_size, input_size, sequence_length)
        
        for conv in self.convs:
            x = conv(x)  # (batch_size, hidden_size, sequence_length)
            
            # Apply TLSLT activations
            x = self.tanh1(x)      # T
            x = self.linear1(x)    # L
            x = self.sigmoid(x)    # S
            x = self.linear2(x)    # L
            x = self.tanh2(x)      # T
        
        x = x.permute(0, 2, 1)  # (batch_size, sequence_length, hidden_size)
        out = self.fc(x)        # (batch_size, sequence_length, output_size)
        return out
    

# Define the FNN-based sequence-to-sequence regression model
class Seq2SeqFNN(nn.Module):
    def __init__(self, input_size, sequence_length_in, sequence_length_out, hidden_size, output_size):
        super(Seq2SeqFNN, self).__init__()
        self.input_size = input_size
        self.n_in= sequence_length_in
        self.n_Out = sequence_length_out
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.type = 'FNN'
        # Fully connected layers
        self.flatten = nn.Flatten(start_dim=1) 
        self.fc1 = nn.Linear(input_size * self.n_in, hidden_size)  # Input layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)                  # Hidden layer
        self.fc3 = nn.Linear(hidden_size, output_size * self.n_Out)  # Output layer
    
    def forward(self, data_in):
        # Flatten the input: (batch_size, sequence_length, input_size) -> (batch_size, sequence_length * input_size)
        data_in = self.flatten(data_in)
        # todo: add flatten! 
        # Pass through fully connected layers
        
        data_in = torch.relu(self.fc1(data_in))
        data_in = torch.relu(self.fc2(data_in))
        data_in = self.fc3(data_in)
        
        # Reshape the output: (batch_size, sequence_length * output_size) -> (batch_size, sequence_length, output_size)
        data_in = data_in.view(-1, self.n_Out, self.output_size)
        return data_in
    
# Define the FNN-based sequence-to-sequence regression model
class Seq2SeqFNNNew(nn.Module):
    def __init__(self, input_size, sequence_length_in, sequence_length_out, hidden_size, output_size):
        super(Seq2SeqFNNNew, self).__init__()
        self.input_size = input_size
        self.n_in= sequence_length_in
        self.n_out = sequence_length_out
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.type = 'FNN'
        # Fully connected layers
        self.myNN = nn.Sequential(nn.Flatten(start_dim=1) ,
                                  nn.Linear(input_size * self.n_in, hidden_size),
                                  nn.Linear(hidden_size, hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size, hidden_size), 
                                  nn.ReLU(),
                                  nn.Linear(hidden_size, hidden_size),
                                  nn.Linear(hidden_size, output_size * self.n_out), 
                                  nn.Unflatten(1, (self.n_out, output_size))
                                  )
    def forward(self, data_in): 
        return self.myNN(data_in)
    
    

def training(model, num_epochs, optimizer, criterion, loaderTraining, loaderVal, flagVerbose=True): 
    
    losses, losses_val = [], []
    epoch_val = []
    if flagVerbose: 
        print('training of {} start'.format(model.type))
    t1 = time.time()
    
    for epoch in range(num_epochs):
        # training
        for inputs, targets in loaderTraining: 
            outputs = model(inputs)
            
            if model.type == 'RNN' or model.type == 'CNN' or model.type == 'LSTM': 
                outputs_cut = outputs[:,(nTotal-nDamped):,:]
                targets_data_cut  = targets[:,(nTotal-nDamped):,:]
            else: 
                outputs_cut = outputs
                targets_data_cut = targets[:,(nTotal-nDamped):,:]
                
            loss = criterion(outputs_cut, targets_data_cut )
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            for x_data, y_data in loaderVal: 
                outputsVal = model(x_data)
                if model.type == 'RNN' or model.type == 'CNN' or model.type == 'LSTM': 
                    outputs_cutVal = outputsVal[:,(nTotal-nDamped):,:]
                    y_data_cutVal = y_data[:,(nTotal-nDamped):,:]
                else: 
                    outputs_cutVal = outputsVal
                    y_data_cutVal = y_data[:,(nTotal-nDamped):,:]
                    
                loss = criterion(outputs_cutVal, y_data_cutVal)
                epoch_val += [epoch]
                losses_val.append(loss.item())
            if flagVerbose: 
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss training: {losses[-1]:.4f} | val: {losses_val[-1]:.4f}')
            model.train()
    t_Training = time.time() - t1
    if flagVerbose: 
        print('training of {} epochs ({}) took {}s\n'.format(num_epochs, model.type, round(t_Training, 1)))
    
    return losses, losses_val, epoch_val, t_Training

def plotTraining(losses, losses_val, epoch_val, figNumberLabel = None): 
    if figNumberLabel is None: 
        plt.figure()
    else: 
        plt.figure(figNumberLabel)
    
    plt.semilogy(losses, label='training' + model.type)
    plt.semilogy(epoch_val, losses_val, label='val' + model.type)
    # plt.title('training loss ' + model.type)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
    plt.legend()
    
    return

def prepareData(data, mask_data = [0,1,2,3,4], cutOutputSteps= 0):
    
    locals().update(data) 
    x_data, y_data = torch.tensor(inputsTraining, dtype=torch.float32), torch.tensor(targetsTraining, dtype=torch.float32)
    x_val, y_val = torch.tensor(inputsTest, dtype=torch.float32), torch.tensor(targetsTest, dtype=torch.float32)
    
    if not (mask_data is [0,1,2,3,4]): 
        x_data = x_data[:,:,mask_data]
        x_val = x_val[:,:,mask_data]
    
    if cutOutputSteps > 0: # number of output steps which are removed -> can be used to switch between multi-step and single-step
        x_data, y_data = x_data[:,0:-cutOutputSteps,:], y_data[:,0:-cutOutputSteps,:]
        x_val, y_val = x_val[:,0:-cutOutputSteps,:], y_val[:,0:-cutOutputSteps,:]
    
    dataTraining = TensorDataset(x_data, y_data)
    dataVal = TensorDataset(x_val, y_val)
    
    loaderTraining = DataLoader(dataTraining, batch_size=len(x_data))
    loaderVal = DataLoader(dataVal, batch_size=len(x_val))
    return loaderTraining, loaderVal 
  
#%% 
# Hyperparameters

if __name__ == '__main__': 
    hidden_size = nDamped           # Number of hidden units
    num_layers = 5             # Number of LSTM layers
    output_size = 3            # Output size (1-dimensional regression)
    learning_rate = 1e-3
    num_epochs = int(400*2.5)
    mask = [0,1,2,3,4]
    input_size = len(mask)
    cutOutputSteps = 99*0
    # Create the model
    # model = RegressionLSTM(input_size, hidden_size, num_layers, output_size)
    # model = Seq2SeqLSTM(input_size, hidden_size, num_layers, output_size)
    model_FNN = Seq2SeqFNN(len(mask), nTotal-cutOutputSteps, nTotal-nDamped-cutOutputSteps, hidden_size, output_size)
    model_RNN = Seq2SeqRNN(len(mask), hidden_size, num_layers, output_size)
    model_LSTM = Seq2SeqLSTM(len(mask), hidden_size, num_layers, output_size)
    model_FNN_new = Seq2SeqFNNNew(len(mask), nTotal, nTotal-nDamped, hidden_size, output_size)
    model_CNN = Seq2SeqCNN(input_size, hidden_size, num_layers, output_size, kernel_size=5)
    loaderTraining, loaderVal = prepareData(data, mask, cutOutputSteps=cutOutputSteps)
    criterion = nn.MSELoss() # Loss for training
    
    def init_weights(m):
        if isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    model_LSTM.apply(init_weights)

    models = []
    for model in [model_CNN]: #model_RNN, model_LSTM,model_CNN, model_FNN

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        losses, losses_val, epoch_val, tTraining = training(model, num_epochs, optimizer, criterion, 
                                                            loaderTraining, loaderVal, flagVerbose=True)
        
        plotTraining(losses, losses_val, epoch_val)
    
        models += [copy.deepcopy(model)]
    
    #%%
    plt.figure()
    import matplotlib.colors as mcolors
    myColList = list(mcolors.TABLEAU_COLORS)
    N_ = 5
    for i in range(N_): 
        inputs, targets = loaderVal.dataset[i]
        outputs = model(inputs.reshape([1, -1, input_size]))
        iCol = i%len(myColList)
        if model.type=='FNN': 
            i_out = np.linspace(nDamped, nTotal-1-cutOutputSteps, nTotal-nDamped-cutOutputSteps)
        else: 
            i_out = np.linspace(0, nTotal-1-cutOutputSteps, nTotal-cutOutputSteps)
        plt.plot(i_out, outputs[0].detach().numpy(), color=myColList[iCol], linewidth=0.5)
        plt.plot(targets.detach().numpy(), '--', color=myColList[iCol], alpha=0.5)
    
    plt.plot([nDamped]*2, [-1,1], 'k:')
