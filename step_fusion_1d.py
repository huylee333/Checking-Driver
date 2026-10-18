"""
Stage 2: Train 1D Swin Transformer (giữ nguyên 100% logic từ step_fusion_1d.py gốc)

File này KHÔNG có thay đổi logic so với ../step_fusion_1d.py
Chỉ điều chỉnh: import path, đường dẫn data mặc định (../data/), tự tạo outpath/logpath.
"""

import sys
import os

# Ưu tiên models/ của chính repo này (đã có SwinTransformer_cls.py chép nguyên từ DuST gốc),
# sau đó mới tới thư mục cha (bố cục cũ: dust_dinov3/ nằm trong repo DuST gốc).
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(1, PARENT_DIR)

import torch
import torch.nn as nn
from models.SwinTransformer_cls import SwinTransformer_1D
from torch.utils.data import Dataset, DataLoader, TensorDataset

import cv2
import numpy as np
import pandas as pd
import random

import torch.optim as optim
from torch.nn import CrossEntropyLoss

import logging

from torch.optim.lr_scheduler import _LRScheduler
import math
import argparse


def evaluate_test_loss(model, test_loader, criterion, epoch, outpath, save_ckpt=True):
    model.eval()
    running_test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs_1d, labels in test_loader:
            inputs_1d, labels = inputs_1d.to(device), labels.to(device)

            outputs = model(inputs_1d)
            test_loss = criterion(outputs, labels)
            running_test_loss += test_loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    average_test_loss = running_test_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    if save_ckpt:  # hành vi gốc: lưu checkpoint mỗi lần đánh giá (chỉ khi bật --save_all_epochs)
        torch.save(model.state_dict(), outpath+'e'+str(epoch)+'_'+str(np.round(test_accuracy,4))+'.pth')
    return average_test_loss, test_accuracy
    
if __name__ == "__main__":
    
    # Training settings (giữ nguyên 100% từ gốc)
    parser = argparse.ArgumentParser(description='Train 1d Swin Transformer')
    parser.add_argument('--batch_size', type=int, default=2000,
                        help='input batch size for training (default: 2000)')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--seed', type=int, default=93728645,
                        help='random seed (default: 93728645)')
    parser.add_argument('--print_every', type=int, default=1,
                        help='epoch interval for validation and model saving (e.g., 1 for every 1 epoch).')
    parser.add_argument('--save_all_epochs', action='store_true',
                        help='lưu checkpoint mỗi lần đánh giá như bản gốc (~760 MB mỗi file). '
                             'Mặc định: chỉ giữ best_1d.pth (val loss thấp nhất) + final_1d.pth')

    # Model
    parser.add_argument('--num_frames', type=int, default=51,
                        help='number of frames (time points)')
    parser.add_argument('--num_channels', type=int, default=3,
                        help='number of channels (default: 3)')

    # Inputs
    parser.add_argument('--path_kine_train', type=str, default='../data/kine_train.pkl',
                        help='Path to the kine training data')
    parser.add_argument('--path_kine_val', type=str, default='../data/kine_val.pkl',
                        help='Path to the kine validation data')
    
    parser.add_argument('--path_label_train', type=str, default='../data/label_train.pkl',
                        help='Path to training label')
    parser.add_argument('--path_label_val', type=str, default='../data/label_val.pkl',
                        help='Path to validation label')
    
    # Outputs
    parser.add_argument('--outpath', type=str, default='./saved_models/step_1d/',
                        help='where to save models')
    parser.add_argument('--logpath', type=str, default='./logs/',
                        help='where to save logs')

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    outpath = args.outpath
    os.makedirs(args.outpath, exist_ok=True)
    os.makedirs(args.logpath, exist_ok=True)

    # Configure logging
    logging.basicConfig(filename=args.logpath+'step_1d.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    logging.info('----- Define Data Loader -----')
    print('----- Define Data Loader -----')
    bsize=args.batch_size
    seq_len = args.num_frames
    num_chl = args.num_channels
    logging.info(f'batch size {bsize}')
    print(f'batch size {bsize}')

    X_train = torch.from_numpy(pd.read_pickle(args.path_kine_train).astype('float32'))
    X_val = torch.from_numpy(pd.read_pickle(args.path_kine_val).astype('float32'))
    label_train = torch.from_numpy(pd.read_pickle(args.path_label_train))
    label_val = torch.from_numpy(pd.read_pickle(args.path_label_val))
    
    train_dataset = TensorDataset(X_train, label_train)
    train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True)

    val_dataset = TensorDataset(X_val, label_val)
    val_loader = DataLoader(val_dataset, batch_size=bsize, shuffle=False)
    
    logging.info('----- Define 1D swin -----')
    print('----- Define 1D swin -----')

    # Giữ nguyên 100% config 1D Swin từ gốc
    model = SwinTransformer_1D(seq_len=seq_len, 
                                 in_chans=num_chl, 
                                 num_classes=4,
                                 window_size=8, 
                                 drop_path_rate=0.1,
                                 patch_size=1,
                                 num_heads=[16, 16, 16, 16], 
                                 depths=[2, 2, 6, 2],
                                 embed_dim=256)
    
    device_id = 0
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    num_epochs = args.num_epochs
    print_every = args.print_every
    lr = args.lr
    logging.info(f'learning rate = {lr}')
    print(f'learning rate = {lr}')

    # Set loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.999), weight_decay=0.02)    
    
    logging.info('start training')
    print('start training')

    test_loss_list = []
    test_acc_list = []
    epoch_list = []
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0

        for i, (inputs_1d, labels) in enumerate(train_loader, 0):

            inputs_1d, labels = inputs_1d.to(device), labels.to(device)
            
            optimizer.zero_grad()

            outputs = model(inputs_1d)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        if (epoch + 1) == 100:
            optimizer = optim.Adam(model.parameters(), lr=3e-5, betas=(0.9,0.999), weight_decay=0.02)
            print("Changed learning rate to 3e-5")
        
        if (epoch + 1) % print_every == 0:
            epoch_list.append(epoch+1)
            test_loss_item, test_acc_item = evaluate_test_loss(model, val_loader, criterion, epoch+1, outpath,
                                                               save_ckpt=args.save_all_epochs)
            logging.info(f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader)}, Val Loss: {test_loss_item}, Val Acc: {test_acc_item}")
            print(f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader)}, Val Loss: {test_loss_item}, Val Acc: {test_acc_item}")
            test_loss_list.append(test_loss_item)
            test_acc_list.append(test_acc_item)
            if not args.save_all_epochs and test_loss_item < best_val_loss:
                best_val_loss = test_loss_item
                torch.save(model.state_dict(), outpath+'best_1d.pth')  # ghi đè, chỉ giữ 1 file
                logging.info(f'best_1d.pth <- epoch {epoch + 1} (val loss {test_loss_item:.4f}, val acc {test_acc_item:.2f})')


    logging.info("Training completed!")
    print("Training completed!")
    
    torch.save(model.state_dict(), outpath+'final_1d.pth')
    np.save(outpath+'val_loss.npy', np.array(test_loss_list))
    np.save(outpath+'val_acc.npy', np.array(test_acc_list))
    logging.info(f'epoch {epoch_list[np.argmin(test_loss_list)]} minimize val loss (index start from 1)')
    print(f'epoch {epoch_list[np.argmin(test_loss_list)]} minimize val loss (index start from 1)')
