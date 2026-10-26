"""
Stage 1: Train DINOv3 Video Encoder (thay thế step_fusion_3d.py)

Thay đổi so với step_fusion_3d.py gốc:
  - SwinTransformer3D → DINOv3VideoEncoder (mặc định ViT-S/16, bản DINOv3 nhỏ nhất)
  - DINOv3 được FINE-TUNE toàn bộ (như Swin-B được train toàn bộ trong bài gốc).
    Dùng --freeze_backbone để quay về chế độ đóng băng.
  - Backbone DINOv3 dùng learning rate = lr * --backbone_lr_mult (mặc định 0.1),
    để không phá trọng số pretrain. Phần temporal/proj/head dùng lr như gốc.
  - Hàm loss mới:
        loss = CE_label_smoothing(logits, nhãn) + supcon_weight * SupCon(z, nhãn)
    z là vector 1024 đầu ra của encoder (đi vào fusion ở Stage 4), nên loss tác động
    trực tiếp lên đặc trưng video và, qua đó, lên DINOv3.
  - Mixed precision bf16 (tắt bằng --no_amp) để chạy batch lớn trên A100.
  - Ghi log độ lớn gradient của DINOv3 và mức thay đổi trọng số DINOv3 để chứng minh
    backbone thực sự được cập nhật (dino_drift.json).
  - Giữ nguyên: CombinedLRScheduler, AdamW (betas, weight_decay), số epoch, data loading,
    cách đánh giá trên val, lưu best/final, tên file output.
"""

import os
import sys
import json
import contextlib

# Đảm bảo 'models' trỏ tới dust_dinov3/models dù chạy script từ thư mục nào
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Console Windows (cp1252) không in được đường dẫn có dấu tiếng Việt -> tránh crash khi print
try:
    sys.stdout.reconfigure(errors='backslashreplace')
except AttributeError:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dinov3_video_encoder import DINOv3VideoEncoder
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd

import torch.optim as optim

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


def supcon_loss(features, labels, temperature=0.1):
    """Supervised contrastive loss (Khosla et al., 2020).

    Với mỗi mẫu i có ít nhất một mẫu cùng nhãn trong batch, kéo i lại gần các mẫu cùng nhãn
    và đẩy ra xa các mẫu khác nhãn. Mẫu không có cặp cùng nhãn trong batch bị bỏ qua.
    features: (B, D), labels: (B,). Tính trong float32 để ổn định với mixed precision.
    """
    z = F.normalize(features.float(), dim=1)
    sim = z @ z.t() / temperature
    B = z.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)                     # loại chính mẫu đó khỏi mẫu so sánh
    pos = (labels[:, None] == labels[None, :]) & ~eye     # cặp dương: cùng nhãn, khác mẫu
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_count = pos.sum(dim=1)
    valid = pos_count > 0
    if not valid.any():
        return features.sum() * 0.0                       # batch không có cặp nào: loss = 0
    mean_log_prob_pos = (log_prob * pos).sum(dim=1)[valid] / pos_count[valid]
    return -mean_log_prob_pos.mean()


def backbone_grad_norm(model):
    """Norm L2 gradient của toàn bộ tham số backbone DINOv3 (0 nếu backbone đóng băng)."""
    sq = 0.0
    for p in model.dino.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().float().pow(2).sum())
    return math.sqrt(sq)


def backbone_drift(model, init_state):
    """Tỉ lệ thay đổi trọng số backbone so với lúc khởi tạo: ||W - W0|| / ||W0||."""
    diff_sq, base_sq = 0.0, 0.0
    cur = model.dino.state_dict()
    for k, w0 in init_state.items():
        w = cur[k].detach().float().cpu()
        diff_sq += float((w - w0).pow(2).sum())
        base_sq += float(w0.pow(2).sum())
    return math.sqrt(diff_sq) / max(math.sqrt(base_sq), 1e-12)


# Giữ nguyên evaluate_test_loss từ step_fusion_3d.py gốc
def evaluate_test_loss(model, test_loader, criterion, epoch, outpath, device, save_ckpt=True):
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
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 16). Cần >= 8 để SupCon có cặp cùng nhãn')
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

    # Fine-tune / loss
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='đóng băng DINOv3 (chỉ train temporal + proj + head). Mặc định: fine-tune toàn bộ')
    parser.add_argument('--backbone_lr_mult', type=float, default=0.1,
                        help='learning rate của backbone = lr * giá trị này (default: 0.1)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='label smoothing cho CrossEntropy khi train (default: 0.1)')
    parser.add_argument('--supcon_weight', type=float, default=0.5,
                        help='trọng số λ của SupCon trên vector 1024 (default: 0.5). 0 = tắt')
    parser.add_argument('--supcon_temp', type=float, default=0.1,
                        help='nhiệt độ của SupCon (default: 0.1)')
    parser.add_argument('--no_amp', action='store_true',
                        help='tắt mixed precision bf16 (mặc định bật khi có GPU)')

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
    parser.add_argument('--path_video_train', type=str, default='./data/video_train.pkl',
                        help='Path to the video training data')
    parser.add_argument('--path_video_val', type=str, default='./data/video_val.pkl',
                        help='Path to the video validation data')

    parser.add_argument('--path_label_train', type=str, default='./data/label_train.pkl',
                        help='Path to training label')
    parser.add_argument('--path_label_val', type=str, default='./data/label_val.pkl',
                        help='Path to validation label')

    # Outputs
    parser.add_argument('--outpath', type=str, default='./saved_models/step_3d/',
                        help='where to save models')
    parser.add_argument('--logpath', type=str, default='./logs/',
                        help='where to save logs')

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == 'cuda') and (not args.no_amp)
    outpath = args.outpath
    os.makedirs(args.outpath, exist_ok=True)
    os.makedirs(args.logpath, exist_ok=True)

    # Configure logging
    logging.basicConfig(filename=args.logpath+'step_dinov3_3d.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print(f'device={device} | amp={use_amp} | freeze_backbone={args.freeze_backbone}')
    logging.info(f'device={device} | amp={use_amp} | args={vars(args)}')

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
        freeze_backbone=args.freeze_backbone,
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

    # Loss
    #  - train: CE có label smoothing + SupCon trên vector 1024
    #  - đánh giá: CE thuần, để val loss vẫn đo đúng bài toán phân loại
    criterion_train = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    criterion_eval = nn.CrossEntropyLoss()

    # Optimizer: 2 nhóm tham số, backbone dùng lr nhỏ hơn
    backbone_params = [p for n, p in model.named_parameters() if n.startswith('dino.') and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith('dino.') and p.requires_grad]
    backbone_lr = lr * args.backbone_lr_mult
    param_groups = [{'params': head_params, 'lr': lr}]
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': backbone_lr})
    optimizer = optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999), weight_decay=0.02)
    scheduler = CombinedLRScheduler(optimizer, num_epochs=num_epochs, end_of_linear=2.5)
    logging.info(f'head/temporal lr={lr} | backbone lr={backbone_lr if backbone_params else "n/a"} '
                 f'| backbone params={sum(p.numel() for p in backbone_params):,}')
    print(f'head lr={lr} | backbone lr={backbone_lr if backbone_params else "n/a"}')

    # Snapshot trọng số DINOv3 lúc khởi tạo để đo mức thay đổi
    dino_init = None
    if backbone_params:
        dino_init = {k: v.detach().float().cpu().clone() for k, v in model.dino.state_dict().items()
                     if v.is_floating_point()}

    amp_ctx = (lambda: torch.autocast(device_type='cuda', dtype=torch.bfloat16)) if use_amp \
        else (lambda: contextlib.nullcontext())

    logging.info('start training')
    test_loss_list = []
    test_acc_list = []
    epoch_list = []
    drift_list = []
    grad_norm_epoch = []
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        running_ce = 0.0
        running_sc = 0.0
        epoch_gn = []

        for i, (inputs_3d, labels) in enumerate(train_loader, 0):
            inputs_3d, labels = inputs_3d.float().to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with amp_ctx():
                feat = model.get_vid_feature(inputs_3d)          # (B, 1024): đầu ra encoder
                logits = model.head(feat)                        # (B, 4)
                ce = criterion_train(logits.float(), labels)
                sc = supcon_loss(feat, labels, args.supcon_temp) if args.supcon_weight > 0 \
                    else feat.new_zeros(())
                loss = ce + args.supcon_weight * sc
            loss.backward()

            if backbone_params:
                gn = backbone_grad_norm(model)
                epoch_gn.append(gn)
            else:
                gn = 0.0
            optimizer.step()

            running_loss += loss.item()
            running_ce += ce.item()
            running_sc += float(sc)

            if i % 50 == 0:
                msg = (f'---- {i} th batch ----- ce={ce.item():.4f} supcon={float(sc):.4f} '
                       f'backbone_grad_norm={gn:.4e}')
                logging.info(msg)

        n_batches = len(train_loader)
        if epoch_gn:
            grad_norm_epoch.append(float(np.mean(epoch_gn)))

        if (epoch + 1) % print_every == 0:
            epoch_list.append(epoch+1)
            test_loss_item, test_acc_item = evaluate_test_loss(model, val_loader, criterion_eval, epoch+1, outpath, device,
                                                               save_ckpt=args.save_all_epochs)
            train_line = (f"Train Loss: {running_loss / n_batches:.4f} "
                          f"(CE {running_ce / n_batches:.4f}, SupCon {running_sc / n_batches:.4f})")
            logging.info(f"Epoch: {epoch + 1}/{num_epochs}, {train_line}, Val Loss: {test_loss_item}, Val Acc: {test_acc_item}")
            print(f"Epoch: {epoch + 1}/{num_epochs}, {train_line}, Val Loss: {test_loss_item}, Val Acc: {test_acc_item}")
            test_loss_list.append(test_loss_item)
            test_acc_list.append(test_acc_item)
            if dino_init is not None:
                drift = backbone_drift(model, dino_init)
                drift_list.append({'epoch': epoch + 1, 'drift': drift})
                logging.info(f'DINOv3 weight drift ||W-W0||/||W0|| = {drift:.6f}')
                print(f'DINOv3 weight drift = {drift:.6f}')
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
    if dino_init is not None:
        final_drift = backbone_drift(model, dino_init)
        with open(outpath+'dino_drift.json', 'w', encoding='utf-8') as f:
            json.dump({'final_drift': final_drift, 'per_eval': drift_list,
                       'backbone_grad_norm_per_epoch': grad_norm_epoch}, f, indent=2)
        logging.info(f'final DINOv3 weight drift = {final_drift:.6f}')
        print(f'final DINOv3 weight drift = {final_drift:.6f}')
    logging.info(f'epoch {epoch_list[np.argmin(test_loss_list)]} minimize val loss (index start from 1)')
    print(f'epoch {epoch_list[np.argmin(test_loss_list)]} minimize val loss (index start from 1)')
