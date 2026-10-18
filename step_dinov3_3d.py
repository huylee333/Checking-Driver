"""
Stage 1: Train DINOv3 Video Encoder (thay thế step_fusion_3d.py)

Thay đổi so với step_fusion_3d.py gốc:
  - SwinTransformer3D → DINOv3VideoEncoder (mặc định ViT-S/16, bản DINOv3 nhỏ nhất)
  - DINOv3 backbone frozen, chỉ train: temporal encoder + projection + head
  - Optimizer AdamW giữ nguyên, chỉ optimize trainable params
  - Tự tạo thư mục outpath/logpath (thư mục dust_dinov3/ ban đầu chưa có)
  - Mọi thứ khác (data loading, scheduler, eval, logging, lưu model) giữ nguyên
"""

import os
import sys

# Đảm bảo 'models' trỏ tới dust_dinov3/models dù chạy script từ thư mục nào
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Console Windows (cp1252) không in được đường dẫn có dấu tiếng Việt (vd. "Chuyên đề") -> tránh crash khi print
try:
    sys.stdout.reconfigure(errors='backslashreplace')
except AttributeError:
    pass

import torch
import torch.nn as nn
from models.dinov3_video_encoder import DINOv3VideoEncoder
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


# Giữ nguyên CombinedLRScheduler từ step_fusion_3d.py gốc
class CombinedLRScheduler(_LRScheduler):
    def __init__(self, optimizer, num_epochs, end_of_linear=2.5, eta_min=0, last_epoch=-1):
        self.num_epochs = num_epochs
        self.end_of_linear = end_of_linear
        self.eta_min = eta_min
        super(CombinedLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.end_of_linear:
            # Linear decay
            factor = 1 - self.last_epoch / self.end_of_linear
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            T_max = self.num_epochs - self.end_of_linear
            return [self.eta_min + (base_lr - self.eta_min) * 
                    (1 + math.cos(math.pi * (self.last_epoch - self.end_of_linear) / T_max)) / 2
                    for base_lr in self.base_lrs]


# Giữ nguyên evaluate_test_loss từ step_fusion_3d.py gốc
def evaluate_test_loss(model, test_loader, criterion, epoch, outpath, save_ckpt=True):
    model.eval()
    running_test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs_3d, labels in test_loader:
            inputs_3d, labels = inputs_3d.float().to(device), labels.to(device)

            outputs = model(inputs_3d)
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
    
    # Training settings (giữ nguyên cấu trúc args từ step_fusion_3d.py)
    parser = argparse.ArgumentParser(description='Train DINOv3 Video Encoder (replaces 3D Video Swin)')
    parser.add_argument('--batch_size', type=int, default=9,
                        help='input batch size for training (default: 9)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=93728645,
                        help='random seed (default: 93728645)')
    parser.add_argument('--print_every', type=int, default=3,
                        help='epoch interval for validation and model saving (e.g., 3 for every 3 epochs).')
    parser.add_argument('--save_all_epochs', action='store_true',
                        help='lưu checkpoint mỗi lần đánh giá như bản gốc (tốn dung lượng). '
                             'Mặc định: chỉ giữ best_3d.pth (val loss thấp nhất) + final_3d.pth')

    # Model
    parser.add_argument('--num_frames', type=int, default=77,
                        help='number of frames (time points)')
    parser.add_argument('--height', type=int, default=224,
                        help='frame height')
    parser.add_argument('--width', type=int, default=224,
                        help='frame width')
    
    # DINOv3-specific args
    parser.add_argument('--dinov3_model', type=str, 
                        default='vit_small_patch16_dinov3.lvd1689m',
                        help='timm / HuggingFace model ID for DINOv3 (default: vit_small_patch16_dinov3.lvd1689m, smallest DINOv3)')
    parser.add_argument('--dinov3_weights', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.safetensors'),
                        help='local DINOv3 weights file (default: dust_dinov3/model.safetensors). '
                             'Pass "" to download from Hugging Face Hub instead.')
    parser.add_argument('--temporal_layers', type=int, default=2,
                        help='number of temporal transformer encoder layers (default: 2)')
    parser.add_argument('--temporal_heads', type=int, default=None,
                        help='number of attention heads for temporal encoder (default: auto)')
    parser.add_argument('--chunk_size', type=int, default=16,
                        help='number of frames processed simultaneously through DINOv3 (default: 16)')

    # Inputs (giữ nguyên paths từ gốc)
    parser.add_argument('--path_video_train', type=str, default='../data/video_train.pkl',
                        help='Path to the video training data')
    parser.add_argument('--path_video_val', type=str, default='../data/video_val.pkl',
                        help='Path to the video validation data')
    
    parser.add_argument('--path_label_train', type=str, default='../data/label_train.pkl',
                        help='Path to training label')
    parser.add_argument('--path_label_val', type=str, default='../data/label_val.pkl',
                        help='Path to validation label')
    
    # Outputs
    parser.add_argument('--outpath', type=str, default='./saved_models/step_3d/',
                        help='where to save models')
    parser.add_argument('--logpath', type=str, default='./logs/',
                        help='where to save logs')

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    outpath = args.outpath
    os.makedirs(args.outpath, exist_ok=True)
    os.makedirs(args.logpath, exist_ok=True)

    # Configure logging
    logging.basicConfig(filename=args.logpath+'step_dinov3_3d.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    
    logging.info('----- Define Data Loader -----')
    print('----- Define Data Loader -----')

    bsize=args.batch_size
    logging.info(f'batch size {bsize}')
    print(f'batch size {bsize}')
    
    logging.info(f'----- data path {args.path_video_train}')
    print(f'----- data path {args.path_video_train}')
  
    video_train = torch.from_numpy(pd.read_pickle(args.path_video_train))
    video_val = torch.from_numpy(pd.read_pickle(args.path_video_val))
    label_train = torch.from_numpy(pd.read_pickle(args.path_label_train))
    label_val = torch.from_numpy(pd.read_pickle(args.path_label_val))
    
    train_dataset = TensorDataset(video_train, label_train)
    train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True)

    val_dataset = TensorDataset(video_val, label_val)
    val_loader = DataLoader(val_dataset, batch_size=bsize, shuffle=False)
    
    # =====================================================================
    # THAY ĐỔI: DINOv3VideoEncoder thay cho SwinTransformer3D
    # =====================================================================
    logging.info('----- Define DINOv3 Video Encoder -----')
    print('----- Define DINOv3 Video Encoder -----')

    model = DINOv3VideoEncoder(
        num_classes=4,              # Giữ nguyên 4 classes như gốc
        num_frames=args.num_frames,
        model_name=args.dinov3_model,
        output_dim=1024,            # Khớp với fusion head gốc: Linear(num_features + 1024, ...)
        temporal_layers=args.temporal_layers,
        temporal_heads=args.temporal_heads,
        chunk_size=args.chunk_size,
        freeze_backbone=True,       # DINOv3 frozen, chỉ train temporal + head
        weights_path=args.dinov3_weights or None   # '' -> tải từ Hugging Face Hub
    )

    # Log thông tin model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    weights_src = args.dinov3_weights or 'Hugging Face Hub'
    logging.info(f'DINOv3 weights: {weights_src}')
    print(f'DINOv3 weights: {weights_src}')
    logging.info(f'DINOv3 model: {args.dinov3_model}')
    logging.info(f'Total params: {total_params:,} | Trainable params: {trainable_params:,}')
    print(f'DINOv3 model: {args.dinov3_model}')
    print(f'Total params: {total_params:,} | Trainable params: {trainable_params:,}')
    
    device_id = 0
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    num_epochs = args.num_epochs
    print_every = args.print_every
    lr = args.lr
    logging.info(f'learning rate = {lr}')
    print(f'learning rate = {lr}')
    
    # Set loss function and optimizer
    # Chỉ optimize trainable parameters (temporal encoder + proj + head)
    criterion = nn.CrossEntropyLoss()
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params_list, lr=lr, betas=(0.9,0.999), weight_decay=0.02)
    scheduler = CombinedLRScheduler(optimizer, num_epochs=num_epochs, end_of_linear=2.5)
    
    
    logging.info('start training')
    test_loss_list = []
    test_acc_list = []
    epoch_list = []
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0

        for i, (inputs_3d, labels) in enumerate(train_loader, 0):
            if i % 50 ==0:
                logging.info('---- '+str(i)+' th batch -----')
            inputs_3d, labels = inputs_3d.float().to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs_3d)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

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
                torch.save(model.state_dict(), outpath+'best_3d.pth')  # ghi đè, chỉ giữ 1 file
                logging.info(f'best_3d.pth <- epoch {epoch + 1} (val loss {test_loss_item:.4f}, val acc {test_acc_item:.2f})')


        scheduler.step()
    logging.info("Training completed!")
    print("Training completed!")
    
    torch.save(model.state_dict(), outpath+'final_3d.pth')
    np.save(outpath+'val_loss.npy', np.array(test_loss_list))
    np.save(outpath+'val_acc.npy', np.array(test_acc_list))
    logging.info(f'epoch {epoch_list[np.argmin(test_loss_list)]} minimize val loss (index start from 1)')
    print(f'epoch {epoch_list[np.argmin(test_loss_list)]} minimize val loss (index start from 1)')
