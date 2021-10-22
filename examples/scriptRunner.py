# 2019/04/10~
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu

# Test a movie recommendation problem. The nodes are either items or users
# and the edges are rating similarities estimated by a Pearson correlation
# coefficient (either rating similarities between items or rating similarities
# between users). The graph signal defined on top of this graph are the
# ratings given to items by a specific user (if the nodes are items) or the
# ratings given by the users to a specific item (if the nodes are users).
# The objective is to estimate the rating on a target node(s), that is, an
# interpolation problem of the graph signal at a target node(s).

# Outputs:
# - Text file with all the hyperparameters selected for the run and the
#   corresponding results (hyperparameters.txt)
# - Pickle file with the random seeds of both torch and numpy for accurate
#   reproduction of results (randomSeedUsed.pkl)
# - The parameters of the trained models, for both the Best and the Last
#   instance of each model (savedModels/)
# - The figures of loss and evaluation through the training iterations for
#   each model (figs/ and trainVars/)
# - If selected, logs in tensorboardX certain useful training variables

#%%##################################################################
#                                                                   #
#                    IMPORTING                                      #
#                                                                   #
#####################################################################

#\\\ Standard libraries:

import sys,os

os.chdir('..')
cwd = os.getcwd()
print(cwd)
sys.path.append(cwd)

import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']
import matplotlib.pyplot as plt
import pickle
import datetime
from copy import deepcopy

import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim

#\\\ Own libraries:
import alegnn.utils.graphTools as graphTools
import alegnn.utils.dataTools
import alegnn.utils.graphML as gml
import alegnn.modules.architectures as archit
import alegnn.modules.model as model
import alegnn.modules.training as training
import alegnn.modules.evaluation as evaluation
import alegnn.modules.loss as loss

#\\\ Separate functions:
from alegnn.utils.miscTools import writeVarValues
from alegnn.utils.miscTools import saveSeed

import movieGNN as mv
import movieGNNWorkingNoDuality as mvnd



epsilons,numberPerturbations,trainStabilityEpsilon,nEpochs,nDataSplits = [0.0001,0.001,0.01,0.1,0.2,0.5],10,0.1,20,10
mvnd.movieFunction(epsilons,numberPerturbations,trainStabilityEpsilon,nEpochs,nDataSplits)




dualNumberOfBatchesPerDual= 10


dualDelta=0.1
dualEta=0.1
trainStabilityEpsilon=0.2
nEpochs=20
nDataSplits=10

mv.moviePerturbationFunction(dualDelta,dualNumberOfBatchesPerDual,dualEta,trainStabilityEpsilon,nEpochs,nDataSplits)

dualDelta=0.1
dualEta=0.05
trainStabilityEpsilon=0.2
nEpochs=20
nDataSplits=10

mv.moviePerturbationFunction(dualDelta,dualNumberOfBatchesPerDual,dualEta,trainStabilityEpsilon,nEpochs,nDataSplits)

dualDelta=0.1
dualEta=0.02
trainStabilityEpsilon=0.2
nEpochs=20
nDataSplits=10

mv.moviePerturbationFunction(dualDelta,dualNumberOfBatchesPerDual,dualEta,trainStabilityEpsilon,nEpochs,nDataSplits)

dualDelta=0.1
dualEta=0.01
trainStabilityEpsilon=0.2
nEpochs=20
nDataSplits=10

mv.moviePerturbationFunction(dualDelta,dualNumberOfBatchesPerDual,dualEta,trainStabilityEpsilon,nEpochs,nDataSplits)

##

dualDelta=0.05
dualEta=0.1
trainStabilityEpsilon=0.2
nEpochs=20
nDataSplits=10

mv.moviePerturbationFunction(dualDelta,dualNumberOfBatchesPerDual,dualEta,trainStabilityEpsilon,nEpochs,nDataSplits)

dualDelta=0.05
dualEta=0.05
trainStabilityEpsilon=0.2
nEpochs=20
nDataSplits=10

mv.moviePerturbationFunction(dualDelta,dualNumberOfBatchesPerDual,dualEta,trainStabilityEpsilon,nEpochs,nDataSplits)

dualDelta=0.05
dualEta=0.02
trainStabilityEpsilon=0.2
nEpochs=20
nDataSplits=10

mv.moviePerturbationFunction(dualDelta,dualNumberOfBatchesPerDual,dualEta,trainStabilityEpsilon,nEpochs,nDataSplits)

dualDelta=0.05
dualEta=0.01
trainStabilityEpsilon=0.2
nEpochs=20
nDataSplits=10

mv.moviePerturbationFunction(dualDelta,dualNumberOfBatchesPerDual,dualEta,trainStabilityEpsilon,nEpochs,nDataSplits)


##

dualDelta=0.08
dualEta=0.1
trainStabilityEpsilon=0.2
nEpochs=20
nDataSplits=10

mv.moviePerturbationFunction(dualDelta,dualNumberOfBatchesPerDual,dualEta,trainStabilityEpsilon,nEpochs,nDataSplits)

dualDelta=0.08
dualEta=0.05
trainStabilityEpsilon=0.2
nEpochs=20
nDataSplits=10

mv.moviePerturbationFunction(dualDelta,dualNumberOfBatchesPerDual,dualEta,trainStabilityEpsilon,nEpochs,nDataSplits)

dualDelta=0.08
dualEta=0.02
trainStabilityEpsilon=0.2
nEpochs=20
nDataSplits=10

mv.moviePerturbationFunction(dualDelta,dualNumberOfBatchesPerDual,dualEta,trainStabilityEpsilon,nEpochs,nDataSplits)

dualDelta=0.08
dualEta=0.01
trainStabilityEpsilon=0.2
nEpochs=20
nDataSplits=10

mv.moviePerturbationFunction(dualDelta,dualNumberOfBatchesPerDual,dualEta,trainStabilityEpsilon,nEpochs,nDataSplits)