#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Test model for 3D FFRF reduced order model with 2 flexible bodies
#
# Author:   Qasim Khadim and Johannes Gerstmayr
# Contact       : qasim.khadim@outlook.com,qkhadim22 (Github)
# Date:     2024-06-18
#
# Copyright:This file is part of Exudyn. Exudyn is free software. You can redistribute it and/or modify it under the terms of the Exudyn license. See 'LICENSE.txt' for more details.
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
# %%
# %%
import exudyn as exu
from exudyn.utilities import *
import sys
import numpy as np
import time
import os

from Models.FlexibleMultibody import NNHydraulics

os.environ['OMP_NUM_THREADS'] = '4' 

timeStep    = 5e-3              #Simulation time step: Change it as desired.
T           = 10               #Time period
ns          = int(T/timeStep)       
angleInit1  = np.deg2rad(14.6)  #Lift boom angle               
angleInit2  = np.deg2rad(-58.8) #Tilt boom angle 

Plotting    =  True

dataPath    = 'solution/data' + str(T) + '-' + 's' + str(ns) + 'Steps' 


model       = NNHydraulics(nStepsTotal=ns, endTime=T, Flexible=True, 
                      nModes=2, #Change it, as desired.
                      loadFromSavedNPY=True, 
                      visualization  = False,
                      verboseMode=1)

inputVec    =model.CreateInputVector( ns,  angleInit1,angleInit2 )
data = model.ComputeModel(inputVec, solutionViewer = True) #solutionViewer: for visualization

data_array = np.array(data, dtype=object)
np.save(dataPath, data_array)

if Plotting:
   model.Plotting(data_array)
