#!/usr/bin/env python
# coding=utf-8
#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import numpy as np
import time
import os
import copy
import argparse
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from utils_pytorch import *

def compute_exemplars(tg_model, evalloader, index1, index2, num_samples, alpha_dr_herding_2, nb_protos_cl, num_outputs=50, device=None):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tg_model.eval()


    #features = np.zeros([num_samples, num_features])
    
    output_prob = np.zeros([num_samples, num_outputs])
    
    predictions = np.ones(num_samples)*-1
    target_s = np.zeros(num_samples)

    start_idx = 0

    with torch.no_grad():
        for inputs, targets in evalloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = tg_model(inputs)
            outputs = F.softmax(outputs, dim=1)

            output_prob[start_idx:start_idx+inputs.shape[0], :] = outputs


            
            #print(f"Np Squeeze Ouputs = {np.squeeze(outputs)[0:2,0:10]}")
            #print(f" Ouputs = {outputs[0:2,0:10]}")

            _, predicted = outputs.max(1)
            predictions[start_idx:start_idx+inputs.shape[0]] = predicted
            #print(f" predicted shape = {predicted.shape} {predicted[0:10]}")
            target_s[start_idx:start_idx+inputs.shape[0]] = targets

            start_idx = start_idx+inputs.shape[0]

        #predictions = torch.from_numpy(predictions).double()
        #predictions = predictions.to(device)
        #print(f"Shape of output prob = {output_prob.shape}. Prediction Shape = {predictions.shape}")
        #print(f"Predictions type = {predictions.type()} Targets Type = {targets.type()}")
        print(f"shape = {np.equal(predictions,target_s)[0:100]}") 
        print(f"predicted = {predictions[0:10]}")
        print(f"labesl = {targets[0:10]}")
        #fake_output_prob = np.ones(output_prob.shape)*-1
        #chosen_out_prob = np.where(np.equal(output_prob.max(1)[1], target_s), output_prob, fake_output_prob)
        
        tf_index_list = 

        ratios = np.zeros(predictions.shape)

        for row in range(output_prob.shape[0]):
            if(predictions[row]==target_s[row]):
                output_prob[row,:] = output_prob[row,:]
                temp = np.sort(output_prob[row,:])
                ratios[row] = float(temp[-1])/temp[-2] 


            else:
                output_prob[row,:] = np.ones(output_prob[row,:].shape)*-1
                ratios[row] = 100

        top_exemplar_indices = np.argsort(ratios)[0:2*nb_protos_cl]  

        rank = 1 
        for index in range(top_exemplar_indices.shape[0]):
            alpha_dr_herding_2[index1, index, index2] = rank 



    assert(start_idx==num_samples)
    return alpha_dr_herding_2