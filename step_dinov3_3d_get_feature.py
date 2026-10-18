"""
Stage 3: Extract Video Features using trained DINOv3 Video Encoder
(Thay thế step_fusion_3d_get_feature.py)

Mục đích:
  Dùng model DINOv3VideoEncoder đã train ở Stage 1 để trích xuất 
  vector đặc trưng (B, 1024) từ video_train.pkl và video_val.pkl.
  Các đặc trưng này sẽ được lưu thành train_3d_features.pkl và 
  val_3d_features.pkl để đưa vào Stage 4 (Multimodal Fusion).

Thay đổi so với step_fusion_3d_get_feature.py gốc:
  - SwinTransformer3D → DINOv3VideoEncoder (mặc định ViT-S/16, bản DINOv3 nhỏ nhất)
  - Model weights mặc định: ./saved_models/step_3d/final_3d.pth (output của Stage 1)
  - Các tham số kiến trúc (--dinov3_model_name, --temporal_layers, --temporal_heads)
    PHẢI giống Stage 1, nếu không load_state_dict sẽ báo lỗi
  - train_loader dùng shuffle=False để thứ tự feature khớp với kine/label
    (bản gốc dùng shuffle=True, làm lệch thứ tự feature so với label ở Stage 4)
  - Output features giữ nguyên shape (N, 1024) khớp với Stage 4 gốc
"""

import sys
import os

# Ensure current dir and parent dir are in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(1, PARENT_DIR)

import torch
import torch.nn as nn
from models.dinov3_video_encoder import DINOv3VideoEncoder
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import logging
import pickle
import argparse


def get_features(model, loader, device):
    """Trích xuất feature qua model.get_vid_feature(video_tensor).
    
    Giữ nguyên signature và logic với get_features() gốc trong
    step_fusion_3d_get_feature.py
    """
    model.eval()
    features = []
    with torch.no_grad():
        for i, (video_tensor, _) in enumerate(loader):
            video_tensor = video_tensor.float().to(device)
            output = model.get_vid_feature(video_tensor)
            features.append(output.cpu().numpy())
            
            if (i + 1) % 5 == 0 or (i + 1) == len(loader):
                logging.info(f'Processed {i + 1}/{len(loader)} batches')
                print(f'Processed {i + 1}/{len(loader)} batches')
    return np.concatenate(features, axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract Video Features with DINOv3 Video Encoder')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='input batch size for feature extraction (default: 8 để tránh OOM)')
    parser.add_argument('--seed', type=int, default=93728645,
                        help='random seed (default: 93728645)')
    
    # Model
    parser.add_argument('--path_3d_model', type=str, 
                        default='./saved_models/step_3d/final_3d.pth',
                        help='Path to trained DINOv3 video model')
    parser.add_argument('--dinov3_model_name', type=str, 
                        default='vit_small_patch16_dinov3.lvd1689m',
                        help='timm / HuggingFace DINOv3 model name, must match Stage 1 (default: vit_small_patch16_dinov3.lvd1689m)')
    parser.add_argument('--temporal_layers', type=int, default=2,
                        help='number of temporal transformer layers, must match Stage 1 (default: 2)')
    parser.add_argument('--temporal_heads', type=int, default=None,
                        help='number of temporal attention heads, must match Stage 1 (default: auto)')
    parser.add_argument('--chunk_size', type=int, default=16,
                        help='number of frames processed simultaneously through DINOv3 (default: 16)')
    parser.add_argument('--num_frames', type=int, default=77,
                        help='number of frames (time points)')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='number of classes')
    parser.add_argument('--output_dim', type=int, default=1024,
                        help='Output feature dimension (default: 1024, khớp fusion head)')

    # Inputs (mặc định lấy từ thư mục data/ của project gốc)
    parser.add_argument('--path_video_train', type=str, default='../data/video_train.pkl',
                        help='Path to the video training data')
    parser.add_argument('--path_video_val', type=str, default='../data/video_val.pkl',
                        help='Path to the video validation data')
    parser.add_argument('--path_label_train', type=str, default='../data/label_train.pkl',
                        help='Path to training label')
    parser.add_argument('--path_label_val', type=str, default='../data/label_val.pkl',
                        help='Path to validation label')

    # Outputs
    parser.add_argument('--outpath', type=str, default='./video_features/',
                        help='where to save extracted features')
    parser.add_argument('--logpath', type=str, default='./logs/',
                        help='where to save logs')

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.outpath, exist_ok=True)
    os.makedirs(args.logpath, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        filename=os.path.join(args.logpath, 'get_dinov3_features.log'), 
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    logging.info('----- Define Data Loader -----')
    print('----- Define Data Loader -----')

    bsize = args.batch_size
    logging.info(f'batch size {bsize}')
    print(f'batch size {bsize}')
    
    logging.info(f'Loading data from {args.path_video_train} and {args.path_video_val}')
    print(f'Loading data from {args.path_video_train} and {args.path_video_val}')
  
    # Giữ nguyên dtype khi nạp (float16 từ prepare_uah_data.py chiếm nửa RAM),
    # get_features() đổi sang float32 theo từng batch bằng .float() -> kết quả như nhau.
    video_train = torch.from_numpy(pd.read_pickle(args.path_video_train))
    video_val = torch.from_numpy(pd.read_pickle(args.path_video_val))
    
    label_train = torch.from_numpy(pd.read_pickle(args.path_label_train))
    label_val = torch.from_numpy(pd.read_pickle(args.path_label_val))
    
    train_dataset = TensorDataset(video_train, label_train)
    train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=False)

    val_dataset = TensorDataset(video_val, label_val)
    val_loader = DataLoader(val_dataset, batch_size=bsize, shuffle=False)

    logging.info('---- Initialize DINOv3 Video Encoder ----')
    print('---- Initialize DINOv3 Video Encoder ----')

    model = DINOv3VideoEncoder(
        num_classes=args.num_classes,
        num_frames=args.num_frames,
        model_name=args.dinov3_model_name,
        output_dim=args.output_dim,
        temporal_layers=args.temporal_layers,
        temporal_heads=args.temporal_heads,
        chunk_size=args.chunk_size,
        freeze_backbone=True,
        pretrained=False   # checkpoint Stage 1 đã chứa toàn bộ trọng số DINOv3, không cần tải lại
    )

    # Giống bản gốc: bắt buộc phải có model đã train ở Stage 1
    if not os.path.exists(args.path_3d_model):
        raise FileNotFoundError(f'Không tìm thấy checkpoint Stage 1: {args.path_3d_model}. Hãy chạy step_dinov3_3d.py trước.')
    logging.info(f'Loading weights from {args.path_3d_model}')
    print(f'Loading weights from {args.path_3d_model}')
    checkpoint = torch.load(args.path_3d_model, map_location=device)
    model.load_state_dict(checkpoint)

    model = model.to(device)
    
    logging.info('Extracting features for training set...')
    print('Extracting features for training set...')
    train_features = get_features(model, train_loader, device)
    train_out_file = os.path.join(args.outpath, 'train_3d_features.pkl')
    with open(train_out_file, 'wb') as f:
        pickle.dump(train_features, f)
    logging.info(f'Saved train features shape: {train_features.shape} to {train_out_file}')
    print(f'Saved train features shape: {train_features.shape} to {train_out_file}')

    logging.info('Extracting features for validation set...')
    print('Extracting features for validation set...')
    val_features = get_features(model, val_loader, device)
    val_out_file = os.path.join(args.outpath, 'val_3d_features.pkl')
    with open(val_out_file, 'wb') as f:
        pickle.dump(val_features, f)
    logging.info(f'Saved val features shape: {val_features.shape} to {val_out_file}')
    print(f'Saved val features shape: {val_features.shape} to {val_out_file}')
    
    print('Stage 3 feature extraction complete!')
