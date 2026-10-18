"""
Stage 4: Multimodal Fusion Training (Giữ nguyên logic từ step_fusion.py gốc)

Mục đích:
  Kết hợp đặc trưng vận động học Kinematic (1D Swin) với đặc trưng 
  video (DINOv3 Video Encoder) đã trích xuất từ Stage 3.

File này KHÔNG thay đổi logic mô hình hay thuật toán so với step_fusion.py gốc.
Chỉ cấu hình:
  - sys.path để import models từ project gốc
  - Đường dẫn mặc định tương đối với thư mục dust_dinov3/
  - Video features lấy từ ./video_features/ (được trích xuất từ DINOv3)
  - embed_dim=256 cho khớp với Stage 2 (bản gốc để 128 nên không load được model Stage 2)
"""

import sys
import os

# Thêm parent directory vào path để import models từ project gốc
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(1, PARENT_DIR)

import torch
import torch.nn as nn
from models.SwinTransformer_cls import SwinTransformer_1D
from models.SwinTransformer_fusion import SwinTransformer_1D as SwinTransformer_1D_fusion
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


def evaluate_test_loss(model, test_loader, criterion, epoch, outpath, device, save_ckpt=True):
    """Đánh giá loss và accuracy trên tập validation/test."""
    model.eval()
    running_test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs_1d, labels, vid_feat in test_loader:
            inputs_1d = inputs_1d.to(device)
            labels = labels.to(device)
            vid_feat = vid_feat.to(device)

            outputs = model(inputs_1d, vid_feat)
            test_loss = criterion(outputs, labels)
            running_test_loss += test_loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    average_test_loss = running_test_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    if save_ckpt:  # hành vi gốc: lưu checkpoint mỗi lần đánh giá (chỉ khi bật --save_all_epochs)
        save_path = os.path.join(outpath, f'e{epoch}_{np.round(test_accuracy, 4)}.pth')
        torch.save(model.state_dict(), save_path)
    return average_test_loss, test_accuracy
    

if __name__ == "__main__":
    
    # Training settings
    parser = argparse.ArgumentParser(description='Train 1d Swin Transformer + DINOv3 Video Multimodal Fusion')
    parser.add_argument('--batch_size', type=int, default=2000,
                        help='input batch size for training (default: 2000)')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--seed', type=int, default=93728645,
                        help='random seed (default: 93728645)')
    parser.add_argument('--print_every', type=int, default=1,
                        help='epoch interval for validation and model saving (default: 1).')
    parser.add_argument('--save_all_epochs', action='store_true',
                        help='lưu checkpoint mỗi lần đánh giá như bản gốc (~760 MB mỗi file). '
                             'Mặc định: chỉ giữ best_fusion.pth (val loss thấp nhất) + final_fusion.pth')

    # Model
    parser.add_argument('--path_1d_model', type=str, default='./saved_models/step_1d/final_1d.pth',
                        help='Path to pre-trained swin 1d model (default: ./saved_models/step_1d/final_1d.pth)')
    parser.add_argument('--num_frames', type=int, default=51,
                        help='number of frames for kinematic sequence (time points)')
    parser.add_argument('--num_channels', type=int, default=3,
                        help='number of channels (default: 3)')

    # Inputs (tương thích đường dẫn khi chạy từ dust_dinov3)
    parser.add_argument('--path_kine_train', type=str, default='../data/kine_train.pkl',
                        help='Path to the kine training data')
    parser.add_argument('--path_kine_val', type=str, default='../data/kine_val.pkl',
                        help='Path to the kine validation data')

    parser.add_argument('--path_vidfeat_train', type=str, default='./video_features/train_3d_features.pkl',
                        help='Path to the training video feature (extracted from DINOv3)')
    parser.add_argument('--path_vidfeat_val', type=str, default='./video_features/val_3d_features.pkl',
                        help='Path to the validation video feature (extracted from DINOv3)')
    
    parser.add_argument('--path_label_train', type=str, default='../data/label_train.pkl',
                        help='Path to training label')
    parser.add_argument('--path_label_val', type=str, default='../data/label_val.pkl',
                        help='Path to validation label')
    
    # Outputs
    parser.add_argument('--outpath', type=str, default='./saved_models/step_fusion/',
                        help='where to save models')
    parser.add_argument('--logpath', type=str, default='./logs/',
                        help='where to save logs')

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    outpath = args.outpath
    
    os.makedirs(args.outpath, exist_ok=True)
    os.makedirs(args.logpath, exist_ok=True)
    
    # Configure logging
    log_file = os.path.join(args.logpath, 'step_fusion.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    logging.info('----- Define Data Loader -----')
    print('----- Define Data Loader -----')

    bsize = args.batch_size
    seq_len = args.num_frames
    num_chl = args.num_channels
    logging.info(f'batch size {bsize}')
    print(f'batch size {bsize}')
      
    X_train = torch.from_numpy(pd.read_pickle(args.path_kine_train).astype('float32'))
    X_val = torch.from_numpy(pd.read_pickle(args.path_kine_val).astype('float32'))
    
    X_add_train = torch.from_numpy(pd.read_pickle(args.path_vidfeat_train).astype('float32'))
    X_add_val = torch.from_numpy(pd.read_pickle(args.path_vidfeat_val).astype('float32'))

    label_train = torch.from_numpy(pd.read_pickle(args.path_label_train))
    label_val = torch.from_numpy(pd.read_pickle(args.path_label_val))
    
    train_dataset = TensorDataset(X_train, label_train, X_add_train)
    train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True)

    val_dataset = TensorDataset(X_val, label_val, X_add_val)
    val_loader = DataLoader(val_dataset, batch_size=bsize, shuffle=False)
    
    logging.info('----- Define 1D Swin Fusion Model -----')
    print('----- Define 1D Swin Fusion Model -----')

    model = SwinTransformer_1D_fusion(seq_len=seq_len, 
                                      in_chans=num_chl, 
                                      num_classes=4,
                                      window_size=8, 
                                      patch_size=1,
                                      num_heads=[16, 16, 16, 16], 
                                      depths=[2, 2, 6, 2],
                                      # Sửa lỗi kế thừa từ repo gốc: Stage 2 (step_fusion_1d.py) dùng embed_dim=256
                                      # (commit eec315b "fix embed dim errors") nhưng step_fusion.py gốc vẫn để 128
                                      # -> load_state_dict báo size mismatch. Đặt 256 để load được trọng số Stage 2.
                                      embed_dim=256)

    # Giống bản gốc: bắt buộc load 1D Swin đã train ở Stage 2 (bỏ head), không train từ đầu
    logging.info(f'Loading pre-trained 1D Swin weights from {args.path_1d_model}')
    print(f'Loading pre-trained 1D Swin weights from {args.path_1d_model}')
    loaded_state_dict = torch.load(args.path_1d_model, map_location=device)
    partial_state_dict = {key: value for key, value in loaded_state_dict.items() if not key.startswith('head')}
    model.load_state_dict(partial_state_dict, strict=False)

    model = model.to(device)

    num_epochs = args.num_epochs
    print_every = args.print_every
    lr = args.lr
    logging.info(f'learning rate = {lr}')
    print(f'learning rate = {lr}')
    
    # Set loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.02)    
    
    logging.info('start training')
    print('start training')

    test_loss_list = []
    test_acc_list = []
    epoch_list = []
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs_1d, labels, vid_feat) in enumerate(train_loader):
            inputs_1d = inputs_1d.to(device)
            labels = labels.to(device)
            vid_feat = vid_feat.to(device)

            optimizer.zero_grad()
            outputs = model(inputs_1d, vid_feat)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch + 1) % print_every == 0:
            epoch_list.append(epoch + 1)
            test_loss_item, test_acc_item = evaluate_test_loss(model, val_loader, criterion, epoch + 1, outpath, device,
                                                               save_ckpt=args.save_all_epochs)
            logging.info(f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader)}, Val Loss: {test_loss_item}, Val Acc: {test_acc_item}")
            print(f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader)}, Val Loss: {test_loss_item}, Val Acc: {test_acc_item}")
            test_loss_list.append(test_loss_item)
            test_acc_list.append(test_acc_item)
            if not args.save_all_epochs and test_loss_item < best_val_loss:
                best_val_loss = test_loss_item
                torch.save(model.state_dict(), os.path.join(outpath, 'best_fusion.pth'))  # ghi đè, chỉ giữ 1 file
                logging.info(f'best_fusion.pth <- epoch {epoch + 1} (val loss {test_loss_item:.4f}, val acc {test_acc_item:.2f})')

    logging.info("Training completed!")
    print("Training completed!")

    # Giống bản gốc: lưu model cuối cùng + lịch sử val loss/acc
    torch.save(model.state_dict(), os.path.join(outpath, 'final_fusion.pth'))
    np.save(os.path.join(outpath, 'val_loss.npy'), np.array(test_loss_list))
    np.save(os.path.join(outpath, 'val_acc.npy'), np.array(test_acc_list))
    logging.info(f'epoch {epoch_list[np.argmin(test_loss_list)]} minimize val loss (index start from 1)')
    print(f'epoch {epoch_list[np.argmin(test_loss_list)]} minimize val loss (index start from 1)')
