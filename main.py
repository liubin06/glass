# -*- coding: utf-8 -*-
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
import screening
import pandas as pd

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')
print('Torch version: {}, Gpu is available: {}'.format(torch.__version__, USE_CUDA))
torch.autograd.set_detect_anomaly(True)


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def criterion(anchor, positive, negative):
    '''
    :param batch:
    :return: loss
    '''
    pos_sim = torch.sum(anchor * positive, dim=-1)  # [bs,]
    neg_sim = torch.sum(anchor * negative, dim=-1)  # [bs,]
    loss = -torch.log(torch.sigmoid((pos_sim - neg_sim) / args.tau)).mean()
    return loss


def train(net, tsoft_data_loader, visc_data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(tsoft_data_loader)
    for tsoft_anchor, tsoft_pos, tsoft_neg in train_bar:
        tsoft_anchor, tsoft_pos, tsoft_neg = tsoft_anchor.to(device, non_blocking=True), tsoft_pos.to(device,
                                                                                                      non_blocking=True), tsoft_neg.to(
            device, non_blocking=True)
        tsoft_anchor_emb, _ = net(tsoft_anchor)
        tsoft_pos_emb, _ = net(tsoft_pos)
        tsoft_neg_emb, _ = net(tsoft_neg)

        visc_anchor, visc_pos, visc_neg = next(iter(visc_data_loader))
        visc_anchor, visc_pos, visc_neg = visc_anchor.to(device, non_blocking=True), visc_pos.to(device,
                                                                                                 non_blocking=True), visc_neg.to(
            device, non_blocking=True)
        _, visc_anchor_emb = net(visc_anchor)
        _, visc_pos_emb = net(visc_pos)
        _, visc_neg_emb = net(visc_neg)
        # calculate loss value
        loss = criterion(tsoft_anchor_emb, tsoft_pos_emb, tsoft_neg_emb) + criterion(visc_anchor_emb, visc_pos_emb,
                                                                                     visc_neg_emb)

        # optimize
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        total_num += args.batch_size
        total_loss += loss.item() * args.batch_size

        train_bar.set_description(
            'Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, args.epochs, total_loss / total_num))

    return total_loss / total_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--root', type=str, default='data', help='Path to data directory')
    parser.add_argument('--batch_size', default=512, type=int, help='Batch size in each mini-batch')
    parser.add_argument('--num_workers', default=0, type=int, help='Batch size in each mini-batch')
    parser.add_argument('--epochs', default=100, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--learning_rate', default=1e-6, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-7, type=float, help='Weight_decay')
    parser.add_argument('--num_components', default=18, type=int, help='Number of components')
    parser.add_argument('--embedding_dim', default=128, type=int,
                        help='The dimension of the embeddings associated with each component')
    parser.add_argument('--out_dim', default=1024, type=int, help='The dimension of the out put feature')
    parser.add_argument('--fm_dim', default=6, type=int, help='The dimension for factorization of the adjacency matrix')
    parser.add_argument('--num_heads', default=1, type=int, help='Number of attention heads')
    parser.add_argument('--tau', default=0.1, type=float, help='Temperature scalling for loss function')
    parser.add_argument('--noise_std', default=0.1, type=float,
                        help='Stanrd deviriation of noise perbulation for data augmentation')

    init_seed(2024)
    args = parser.parse_args()
    print(args)

    ####################### Step1: Data Preparation #######################
    tsoft_interval, visc_interval = [600, 700], [3, 5]
    print('The interval of tsoft to be SCREENED:', tsoft_interval, '\n', 'The interval of visc to be SCREENED:',
          visc_interval)
    tsoftrain_path = args.root + '/train_tsoft.csv'
    visctrain_path = args.root + '/train_visc.csv'

    tsofvalid_path = args.root + '/validation_tsoft.csv'
    viscvalid_path = args.root + '/validation_visc.csv'

    test_path = args.root + '/test.csv'
    traintsoft = utils.load_train(tsoftrain_path)  # load data as ndarray
    trainvisc = utils.load_train(visctrain_path)  # load data as ndarray

    validtsoft = utils.load_validate(tsofvalid_path)
    validvisc = utils.load_validate(viscvalid_path)

    testdata = utils.load_test(test_path)

    mean_tsoft, std_tsoft = traintsoft.mean(axis=0)[:args.num_components], traintsoft.std(axis=0)[:args.num_components]
    mean_visc, std_visc = trainvisc.mean(axis=0)[:args.num_components], trainvisc.std(axis=0)[:args.num_components]

    train_data_tsoft = utils.MyData(traintsoft, mean_tsoft, std_tsoft, args.num_components, tsoft_interval,
                                    args.noise_std, phase='Training')
    memor_data_tsoft = utils.MyData(traintsoft, mean_tsoft, std_tsoft, args.num_components, tsoft_interval,
                                    args.noise_std, phase='Evaluation')

    train_data_visc = utils.MyData(trainvisc, mean_visc, std_visc, args.num_components, visc_interval, args.noise_std,
                                   phase='Training')
    memor_data_visc = utils.MyData(trainvisc, mean_visc, std_visc, args.num_components, visc_interval, args.noise_std,
                                   phase='Evaluation')

    valid_data_tsoft = utils.MyData(validtsoft, mean_tsoft, std_tsoft, args.num_components, tsoft_interval,
                                    args.noise_std, phase='Evaluation')
    valid_data_visc = utils.MyData(validvisc, mean_visc, std_visc, args.num_components, visc_interval, args.noise_std,
                                   phase='Evaluation')

    test_data = utils.MyData(testdata, mean_tsoft, std_tsoft, args.num_components, None, None, phase='Screening')
    print(
        "Number of training samples within desired Tsoft :{}; Number of training samples out of desired Tsoft interval:{}".format(
            sum(train_data_tsoft.binary_label), len(train_data_tsoft) - sum(train_data_tsoft.binary_label)))

    print('Number of testing samples to be SCREENED :{}'.format(len(test_data)))
    tsoft_train_loader = DataLoader(train_data_tsoft,  # load data as minibatch for GPU computation.
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    drop_last=False,
                                    num_workers=args.num_workers)
    visc_train_loader = DataLoader(train_data_visc,  # load data as minibatch for GPU computation.
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   drop_last=False,
                                   num_workers=args.num_workers)

    tsoft_memor_loader = DataLoader(memor_data_tsoft,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=args.num_workers)
    visc_memor_loader = DataLoader(memor_data_visc,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   drop_last=False,
                                   num_workers=args.num_workers)

    tsoft_valid_loader = DataLoader(valid_data_tsoft,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=args.num_workers
                                    )
    visc_valid_loader = DataLoader(valid_data_visc,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   drop_last=False,
                                   num_workers=args.num_workers
                                   )

    test_loader = DataLoader(test_data,
                             batch_size=args.batch_size,
                             shuffle=False,
                             drop_last=False,
                             num_workers=args.num_workers
                             )

    ######################## Step2: Model Setup #######################
    model = model.DeepGlassNet(args.num_components, args.embedding_dim, args.fm_dim, args.num_heads, args.out_dim).to(
        device)

    ######################## Step3: Optimizer Config #######################
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    ######################## Step4: Model Training #######################
    if not os.path.exists('./results'):
        os.makedirs('./results')
    result = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, tsoft_train_loader, visc_train_loader, optimizer)
        if epoch % 1 == 0:
            tsoft_pre, tsoft_auc = evaluation.eval(model, tsoft_memor_loader, tsoft_valid_loader, 'Tsoft')
            print('Validation Epoch: [{}/{}]: tosft Precision:{:.1f}%, tsoft AUC:{:.4f}'.format(epoch, args.epochs,
                                                                                                tsoft_pre * 100,
                                                                                                tsoft_auc))
            visc_pre, visc_auc = evaluation.eval(model, visc_memor_loader, visc_valid_loader, 'visc')
            print('Validation Epoch: [{}/{}]: visc Precision:{:.1f}%, visc AUC:{:.4f}'.format(epoch, args.epochs,
                                                                                              visc_pre * 100, visc_auc))
            result.append([tsoft_pre, tsoft_auc, visc_pre, visc_auc, train_loss])
        if epoch % 1 == 0:
            screened_id = screening.screen(model, tsoft_memor_loader, visc_memor_loader, test_loader, epoch, args)
            print('Top-10 Screened Samples at Epoch: [{}/{}]'.format(epoch, args.epochs))
            predict = testdata[screened_id]
            print(predict)
    best_rest = np.array(result).max(axis=0)
    best_idx = np.array(result).argmax(axis=0)
    np.savetxt('result.csv', np.array(result), fmt='%.4f', delimiter=',')
    print('\t')
