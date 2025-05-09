#-*- coding: utf-8 -*-
import argparse
import os
import datetime
import random
import torch
import numpy as np
from torch import sparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils
import model
from sklearn.metrics import roc_auc_score
import evaluation
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')

def screen(net, tsoft_memory_loader,visc_memory_loader, test_loader,epoch,args):
    net.eval()
    total_top1,  total_num, tsoft_feature_bank, visc_feature_bank = 0.0,  0, [],[]
    with torch.no_grad():
        # generate feature bank
        for data, target in tsoft_memory_loader:
            feature,_ = net(data.to(device, non_blocking=True))
            tsoft_feature_bank.append(feature)
        tsoft_feature_bank = torch.cat(tsoft_feature_bank, dim=0).contiguous()
        posidx = tsoft_memory_loader.dataset.target_id
        tsoft_pos_features = tsoft_feature_bank[posidx]
        tsoft_pos_center = tsoft_pos_features.mean(dim=0,keepdim=True)
        negidx = tsoft_memory_loader.dataset.nontarget_id
        tsoft_neg_features = tsoft_feature_bank[negidx]
        tsoft_neg_center = tsoft_neg_features.mean(dim=0,keepdim=True)

        for data, target in visc_memory_loader:
            _,feature = net(data.to(device, non_blocking=True))
            visc_feature_bank.append(feature)
        visc_feature_bank = torch.cat(visc_feature_bank, dim=0).contiguous()
        posidx = visc_memory_loader.dataset.target_id
        visc_pos_features = visc_feature_bank[posidx]
        visc_pos_center = visc_pos_features.mean(dim=0,keepdim=True)
        negidx = visc_memory_loader.dataset.nontarget_id
        visc_neg_features = visc_feature_bank[negidx]
        visc_neg_center = visc_neg_features.mean(dim=0,keepdim=True)
        
        tsoft_test_bank,visc_test_bank = [],[]
        test_bar = tqdm(test_loader)
        for data in test_bar:
            test_bar.set_description('Screen Epoch: [{}/{}] '.format(epoch, args.epochs))
            data = data.to(device, non_blocking=True)
            tsoft_feature, visc_feature = net(data)
            tsoft_test_bank.append(tsoft_feature)
            visc_test_bank.append(visc_feature)
        tsoft_test_bank = torch.cat(tsoft_test_bank, dim=0).contiguous()
        visc_test_bank = torch.cat(visc_test_bank, dim=0).contiguous()

        tsoft_pos_score = torch.mm(tsoft_pos_center, tsoft_test_bank.T).squeeze()  # [num_test]
        tsoft_neg_score = torch.mm(tsoft_neg_center, tsoft_test_bank.T).squeeze()  # [num_test]
        tsoft_score = (tsoft_pos_score - tsoft_neg_score)/ (torch.abs(tsoft_pos_score) + torch.abs(tsoft_neg_score) )

        visc_pos_score = torch.mm(visc_pos_center, visc_test_bank.T).squeeze()  # [num_test]
        visc_neg_score = torch.mm(visc_neg_center, visc_test_bank.T).squeeze()  # [num_test]
        visc_score = (visc_pos_score - visc_neg_score) / (torch.abs(visc_pos_score) + torch.abs(visc_neg_score))
        score = tsoft_score * visc_score
        sim_weight, sim_indices = score.topk(k=10, dim=-1)  # [bs,top-k]
    return np.array(sim_indices.cpu())

