#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Details:  Sequential networks
#
# Author:   Peter Manzl
# Date:     2025-06-23
#
# Copyright:This file is part of Exudyn. Exudyn is free software.
# You can redistribute it and/or modify it under the terms of the Exudyn license. 
# See 'LICENSE.txt' for more details.
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# driver file for RNN_Lib min

from RNN_Lib import * 
from exudyn.processing import ProcessParameterList
from itertools import combinations
import pickle
import timeit
import matplotlib.colors as mcolors


import torch
torch.set_num_threads(1)


def myTrainingWrapper(parameterDict): 
    modelType= parameterDict['modelType']
    num_epochs = parameterDict['num_epochs']
    mask = parameterDict['mask']
    hidden_size = parameterDict['hidden_size']
    if 'cutOutputSteps' in parameterDict.keys(): 
        cutOutputSteps = parameterDict['cutOutputSteps']
    else: 
        cutOutputSteps = 0
    return myTraining(modelType, num_epochs, mask, hidden_size, cutOutputSteps)
 

def myTraining(modelType, num_epochs, mask, hidden_size, cutOutputSteps=0): 
    # hidden_size = 64           # Number of hidden units
    # num_layers = 2             # Number of LSTM layers
    output_size = 1            # Output size (1-dimensional regression)
    learning_rate = 0.001
    input_size = len(mask)
    num_recurrent_layers = 2

    # Create the model
    if modelType == 'RNN': 
        model = Seq2SeqRNN(len(mask), hidden_size, num_recurrent_layers , output_size)
    elif modelType == 'FFN' or modelType == 'FNN': 
        model = Seq2SeqFNNNew(len(mask), nTotal-cutOutputSteps, nTotal-nDamped-cutOutputSteps, hidden_size, output_size)
    

    elif modelType.upper == 'LSTM': 
        model = Seq2SeqLSTM(len(mask), hidden_size, num_recurrent_layers , output_size)
    # model_FNN_new = Seq2SeqFNNNew(len(mask), nTotal, nTotal-nDamped, hidden_size, output_size)
    
    loaderTraining, loaderVal = prepareData(data, mask, cutOutputSteps=cutOutputSteps)
    criterion = nn.MSELoss() # Loss for training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    losses, losses_val, epoch_val, tTraining = training(model, num_epochs, optimizer, criterion, 
                                                            loaderTraining, loaderVal, flagVerbose=False)
    input_ = loaderVal.dataset[0][0].reshape(1,-1,input_size)
    NTestInference = 100
    tInference = timeit.timeit(lambda: model(input_), number=NTestInference)/NTestInference
    
    return losses, losses_val, epoch_val, tTraining, tInference, model
     
if __name__ == '__main__': 
    
    paramList = []
    mask_lengths = [5,4,3,2]
    
    # best_masks = [[1,2,3,4] , [1,2], [0,1,2,3], [0,1,3,4]]
    for modelType in ['FFN', 'RNN']:
        for hidden_size in [200]: # 200
            all_masks = []
            mask_original = [0,1,2,3,4]
            if not ('best_masks' in locals()): 
                if 5 in mask_lengths: 
                    all_masks += [mask_original]
                for k in range(4,0,-1): 
                    if k in mask_lengths: 
                        all_masks += list(combinations(mask_original, k))
            else: 
                all_masks = [mask_original] + best_masks
            for mask in all_masks: 
                for num_epochs in [12]:
                    for cutOutputSteps in [0,99]: 
                        if modelType == 'FFN' or modelType == 'FNN': 
                            num_epochs *= 10
                        dict_ = {'modelType': modelType, 
                                 'hidden_size': hidden_size, 
                                 'mask': mask, 
                                 'num_epochs': num_epochs, 
                                 'cutOutputSteps': cutOutputSteps}
                        paramList += [dict_]
    dirName = 'model_{}_hsize_{}_masks_{}_numepochs_{}'.format(modelType, hidden_size, str(mask_lengths), num_epochs)
    if 'cutOutputSteps' in dict_ and dict_['cutOutputSteps'] == 99: 
        dirName += '_singleStep'
    flagRun = True
    if dirName in os.listdir(): 
        inp_ = input('result directory exists. Overwrite? (y/n)')
        print('input: ', inp_)
        flagRun  = inp_.lower() in ['y', 'yes']
    if flagRun: 
        print('start processing parameter list with {} threads. '.format(os.cpu_count()-1))
        resultList  = ProcessParameterList(myTrainingWrapper, parameterList=paramList, useMultiProcessing=True, showProgress=True, nThreads=os.cpu_count()-1)
        os.makedirs(dirName, exist_ok=True)
        with open(dirName + "/results.pkl", "wb") as file:
            pickle.dump(resultList, file)
        for i in range(len(resultList)): 
            torch.save(resultList[i][-1], dirName + '/Model_' + str(i) + '.pt')
    else: 
        with open(dirName + "/results.pkl", "rb") as file:
            resultList = pickle.load(file)
            
    # losses, losses_val, epoch_val, tTraining, tInference, model = out
    #%% 
    colList = list(mcolors.TABLEAU_COLORS) * 3
    nResults = len(resultList)
    nBest = np.min([10, nResults])
    best_losses = []
    for i in range(nResults): 
        best_losses += [np.min(resultList[i][1])]
    j = 0
    for i in range(nResults): 
        
        losses, losses_val, epoch_val, tTraining, tInference, model = resultList[i]
        if np.min(losses_val) > np.sort(best_losses)[nBest -1]: # get the 5 lowest
            continue
        plt.semilogy(epoch_val, losses_val, label= model.type + ', mask: ' + str(paramList[i]['mask']), color=colList[j])
        # plt.semilogy(losses, '--', color=colList[j], alpha=0.4)
        j+= 1
    plt.grid()
    plt.legend()
    plt.ylim([0.8e-4, 0.1])

    #%% 