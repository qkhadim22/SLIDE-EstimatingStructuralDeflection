#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                            #PARAMETERS
#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
from math import sin, cos, sqrt, pi, tanh, atan2, degrees
import exudyn as exu
from exudyn.itemInterface import *
from exudyn.utilities import *
from exudyn.plot import PlotSensor, listMarkerStyles
from exudyn.signalProcessing import GetInterpolatedSignalValue
from exudyn.physics import StribeckFunction
from exudyn.processing import ParameterVariation
from exudyn.FEM import *

fileName1       = 'AbaqusMesh/Pillar.stl'
fileName4       = 'AbaqusMesh/Bracket1.stl'
fileName5       = 'AbaqusMesh/Bracket2.stl'
fileName6       = 'AbaqusMesh/ExtensionBoom.stl'

plane           = graphics.CheckerBoard(point=[2,0,-2], normal=[0,0,1], size=6, nTiles=16, color=color4lightgrey,alternatingColor=color4lightgrey2)

# physical parameters
Gravity         = [0, -9.8066, 0]  # Gravity
# Cylinder and piston parameters
L_Cyl1          = 820e-3                            # Cylinder length
D_Cyl1          = 100e-3                             # Cylinder dia
A_1             = (pi/4)*(D_Cyl1)**2                # Area of cylinder side
L_Pis1          = 535e-3                             # Piston length, also equals to stroke length
d_pis1          = 56e-3                             # Piston dia
A_2             = A_1-(pi/4)*(d_pis1)**2            # Area on piston-rod side
L_Cyl2          = 1050e-3                            # Cylinder length
L_Pis2          = 780e-3                             # Piston length, also equals to stroke length
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
Qn               = 1.667*10*2.1597e-08                     # Nominal flow rate of valve at 18 l/min under
Qn1             = (24/60000)/((9.9)*sqrt(35e5))                      # Nominal flow rate of valve at 18 l/min under
Qn2             = (45/60000)/((9.9)*sqrt(35e5))                     # Nominal flow rate of valve at 18 l/min under


graphicsBody1   = GraphicsDataFromSTLfile(fileName1, color4black,verbose=False, invertNormals=True,invertTriangles=True)
graphicsBody1   = AddEdgesAndSmoothenNormals(graphicsBody1, edgeAngle=0.25*pi,addEdges=True, smoothNormals=True)

#Pillar
PillarP         = np.array([0, 0, 0])
L1              = 0.365    # Length in x-direction
H1              = 1.4769      # Height in y-direction
W1              = 0.25    # Width in z-direction
bodyDim1        = [L1, H1, W1]  # body dimensions
m1              = 93.26
pMid1           = np.array([-0.017403, 0.577291, 0])  # center of mass, body0,0.004000,-0.257068
Inertia1        = np.array([[16.328381,-1.276728, 0.000016],[-1.276728, 0.612003, -5.9e-5],[0.000016,  -5.9e-5  , 16.503728]])
Mark3           = [0,0,0]


# Second Body: LiftBoom
Mark4           = [-90*1e-3, 1426.1*1e-3, 0]

L2              = 3.01055           # Length in x-direction
H2              = 0.45574           # Height in y-direction
W2              = 0.263342          # Width in z-direction
pMid2           = np.array([1.229248, 0.055596, 0])
m2              = 143.66
Inertia2        = np.array([[1.055433, 1.442440,  -0.000003],[ 1.442440,  66.577004, 0],[ -0.000003,              0  ,  67.053707]])
LiftP           = np.array(Mark4)
Mark8           = [304.19e-3, -100.01e-3, 0]
Mark9           = [1258e-3,194.59e-3, 0]
Mark10          = [2685e-3,0.15e-03,0]
Mark11          = [2885e-3,15.15e-3,0]                          
L3              = 2.580         # Length in x-direction
H3              = 0.419         # Height in y-direction
W3              = 0.220         # Width in z-direction
m3              = 141.942729+ 15.928340
pMid3           = np.array([ 0.659935,  0.251085, 0])  # center of mass
Inertia3        = np.array([[1.055433, 1.442440,  -0.000003],[1.442440,  66.577004,    0],[ -0.000003, 0,        67.053707]])
Mark13          = [0, 0, 0]
Mark14          = [-0.095,0.2432,0]
MarkEx          =[-415e-3,287e-3, 0]

L4              = 0.557227    # Length in x-direction
H4              = 0.1425      # Height in y-direction
W4              = 0.15        # Width in z-direction
pMid4           = np.array([0.257068, 0.004000 , 0])
m4              = 11.524039
Inertia4        = np.array([[0.333066, 0.017355, 0],[0.017355, 0.081849, 0],[0,              0, 0.268644]])
Mark15          =[0,0, 0]
Mark16          =[0.456, -0.0405 , 0]
l4              = np.sqrt((Mark16[0]-Mark15[0])**2+(Mark16[1]-Mark15[1])**2)
l4_cmB2         = np.sqrt((Mark16[0]-pMid4[0])**2+(Mark16[1]-pMid4[1])**2)
MarkcmB2        = [l4_cmB2, 0, 0]
graphicsBody4   = GraphicsDataFromSTLfile(fileName4, color4blue,verbose=False, invertNormals=True,invertTriangles=True)
graphicsBody4   = AddEdgesAndSmoothenNormals(graphicsBody4, edgeAngle=0.25*pi,addEdges=True, smoothNormals=True)

L5              = 0.569009       # Length in x-direction
H5              = 0.078827       # Height in y-direction
W5              = 0.15           # Width in z-direction
pMid5           = np.array([0.267208, 0, 0])
m5              = 7.900191
Inertia5        = np.array([[0.052095, 0, 0],[0,  0.260808, 0],[0,              0,  0.216772]])
Mark18          =[0,0, 0]
Mark19          =[0.48, 0, 0]
l5              = np.sqrt((Mark19[0]-Mark18[0])**2+(Mark19[1]-Mark18[1])**2)
l5_cmT          = np.sqrt((Mark19[0]-pMid5[0])**2+(Mark19[1]-pMid5[1])**2)
Mark5cmT        = [l5_cmT, 0, 0]
graphicsBody5   = GraphicsDataFromSTLfile(fileName5, color4blue,verbose=False, invertNormals=True,invertTriangles=True)
graphicsBody5   = AddEdgesAndSmoothenNormals(graphicsBody5, edgeAngle=0.25*pi,addEdges=True, smoothNormals=True)

pMid6           = np.array([1.15, 0.06, 0])
m6              = 58.63
Inertia6        = np.array([[0.13, 0, 0],[0,  28.66, 0],[0,              0,  28.70]])
Mark20          =[0,0, 0]
TipLoadMark     =[2.45, 0.05, 0]
graphicsBody6   = GraphicsDataFromSTLfile(fileName6, color4blue,verbose=False, invertNormals=True,invertTriangles=True)
graphicsBody6   = AddEdgesAndSmoothenNormals(graphicsBody6, edgeAngle=0.25*pi,addEdges=True, smoothNormals=True)

Mark5           = [170*1e-3, 386.113249*1e-3, 0]
pS              = 160e5
pT              = 1e5  
A_d             = (1/100)



#fileNameT       = 'TiltBoomANSYS/TiltBoom' #for load/save of FEM data

import math as mt


import scipy.io 
from scipy.optimize import fsolve, newton

#EXUDYN Libraries

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from matplotlib.patches import FancyArrowPatch





import random
import time


from scipy.stats import norm
import sys, os, time, math, scipy.io
#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                            #LIBRARIES
#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#Files and folders
# from SLIDE.fnnModels import NNtestModel

#from SLIDE.fnnLib import * 
#import SLIDE.fnnLib

from timeit import default_timer as timer
from enum import Enum #for data types


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


