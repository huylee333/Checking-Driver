# DuST-DINOv3 (Experimental Branch)

> **Lưu ý bố cục:** tài liệu này viết cho bố cục phát triển gốc, khi các file nằm trong thư mục `dust_dinov3/` bên trong repo DuST (`DUST/dust_dinov3/`, data ở `DUST/data/`). Trong repo Checking-Driver, các script nằm ở thư mục gốc repo, notebook ở `notebooks/`, tài liệu ở `docs/`. Danh sách đầy đủ các thay đổi so với DuST gốc: xem [README.md](../README.md).

Thử nghiệm thay **Video Swin Transformer 3D** (video encoder của DuST) bằng **DINOv3 bản nhỏ nhất: ViT-S/16** (`vit_small_patch16_dinov3.lvd1689m`, ~21.6M params, 384 chiều) trong kiến trúc **DuST: Dual Swin Transformer for Multi-modal Video and Time-Series Modeling**.

Chỉ phần video encoder thay đổi. Pipeline 4 stage, nhánh 1D Swin cho chuỗi thời gian, fusion head và cách huấn luyện giữ nguyên như project gốc.

> Mô tả bộ dữ liệu UAH-DriveSet và cách chuyển sang định dạng `.pkl` của pipeline: xem [README_DATA.md](README_DATA.md).

---

## 1. Nguyên tắc thiết kế

- **Giữ nguyên ý tưởng & logic gốc**: video encoder → trích video feature 1024 chiều → ghép với 1D Swin trên kinematic → phân loại. 4 stage chạy tuần tự như bản gốc.
- **Không đụng code gốc**: mọi thay đổi nằm trong `dust_dinov3/`. Các model 1D Swin được import trực tiếp từ `../models/`.
- **DINOv3 ViT-S/16 (nhỏ nhất)**: 12 block, patch 16, 1 CLS + 4 register token, hidden 384. Trọng số pretrained tải tự động qua `timm` (Hugging Face Hub, repo `timm/vit_small_patch16_dinov3.lvd1689m`).
- **Factorized spatio-temporal**:
  - DINOv3 (frozen) chạy trên từng frame, lấy **CLS token** (`global_pool='token'`), xử lý theo chunk để tránh tràn VRAM với 77 frame.
  - Temporal Transformer Encoder (2 layer, 6 head, temporal positional embedding + temporal CLS token) học quan hệ thời gian giữa các frame.
  - `Linear(384 → 1024)` để khớp chiều video feature mà fusion head gốc cần: `Linear(num_features + 1024, num_classes)`.
- **Chỉ train phần mới** ở Stage 1 (temporal encoder + projection + head, ~3.95M params). Backbone DINOv3 đóng băng. Đây là khác biệt so với bản gốc (Swin 3D được fine-tune toàn bộ), chọn để vừa GPU 6GB.

| | Bản gốc | Bản DINOv3 |
|---|---|---|
| Video encoder | Swin3D-B (embed 128, depths 2-2-18-2) | DINOv3 ViT-S/16 (frozen) + Temporal Transformer |
| Pretrain | Swin-B ImageNet (inflate 2D→3D) | DINOv3 LVD-1689M |
| Video feature | 1024 (AvgPool3d) | 1024 (temporal CLS → Linear 384→1024) |
| Input | `(B, 3, 77, 224, 224)` | `(B, 3, 77, 224, 224)` (không đổi) |

---

## 2. Cấu trúc thư mục & ánh xạ với project gốc

| File trong `dust_dinov3/` | File gốc tương ứng | Vai trò |
|---|---|---|
| `models/dinov3_video_encoder.py` | `models/video_swin_transformer.py` | Video encoder mới dựa trên DINOv3 ViT-S/16 |
| `models/__init__.py` | (không có) | Thêm `../models/` vào search path để dùng lại 1D Swin gốc |
| `step_dinov3_3d.py` | `step_fusion_3d.py` | **Stage 1**: train video encoder (temporal + head) |
| `step_fusion_1d.py` | `step_fusion_1d.py` | **Stage 2**: train 1D Swin trên kinematic (logic giữ nguyên) |
| `step_dinov3_3d_get_feature.py` | `step_fusion_3d_get_feature.py` | **Stage 3**: trích video feature `(N, 1024)` |
| `step_fusion.py` | `step_fusion.py` | **Stage 4**: train fusion video + kinematic |
| `README_DATA.md` | (không có) | Tài liệu bộ dữ liệu UAH-DriveSet |
| `model.safetensors` | (không có) | Trọng số pretrained DINOv3 ViT-S/16, Stage 1 đọc mặc định |
| `prepare_uah_data.py` | `gen_data.py` (toy data) | Làm sạch UAH-DriveSet, kiểm chứng đồng bộ video–cảm biến, tạo 6 file pickle |
| `run_colab.ipynb` | (không có) | Chạy dữ liệu + 4 stage trên Google Colab GPU, lưu checkpoint lên Drive |

### Những điểm khác bản gốc (ngoài việc đổi encoder)

| Điểm | Lý do |
|---|---|
| Đường dẫn data mặc định là `../data/` | Chạy từ trong `dust_dinov3/`, dùng chung thư mục `data/` của project gốc |
| Tự tạo `saved_models/...`, `logs/`, `video_features/` | Thư mục `dust_dinov3/` ban đầu chưa có các thư mục này, `logging` sẽ lỗi |
| Stage 3: `shuffle=False` cho train loader | Bản gốc dùng `shuffle=True`, làm thứ tự feature lệch với kinematic/label ở Stage 4 |
| Stage 3: thêm `--temporal_layers`, `--temporal_heads`, `--chunk_size` | Phải khớp kiến trúc Stage 1 để `load_state_dict` thành công |
| Stage 3: không ép cả mảng video sang float32 khi nạp | Video float16 từ `prepare_uah_data.py` chỉ chiếm nửa RAM. Mỗi batch vẫn được đổi sang float32 bằng `.float()`, nên kết quả không đổi |
| Stage 1, 2, 4: mặc định chỉ lưu `best_*.pth` (val loss thấp nhất) + `final_*.pth` | Bản gốc lưu checkpoint mỗi lần đánh giá. Model 1D/fusion ~760 MB mỗi file, 200 epoch là ~150 GB mỗi stage. Lịch sử `val_loss.npy`, `val_acc.npy` vẫn lưu đủ, cách train không đổi. Dùng `--save_all_epochs` để lưu như bản gốc |
| Stage 4: `embed_dim=256` cho 1D Swin fusion | Lỗi kế thừa từ repo gốc: Stage 2 đã đổi sang 256 (commit `eec315b`) nhưng `step_fusion.py` gốc vẫn 128, nên không load được model Stage 2 (size mismatch) |

---

## 3. Cài đặt

Môi trường giống README gốc, cộng thêm `timm` đủ mới để có DINOv3:

```bash
pip install "timm>=1.0.20"      # đã kiểm thử với timm 1.0.27
pip install pandas opencv-python
# (tuỳ chọn) fallback Hugging Face: pip install transformers
#   repo facebook/dinov3-* trên HF là gated, cần `huggingface-cli login` + chấp nhận license
```

**Trọng số DINOv3**: Stage 1 mặc định đọc file local `dust_dinov3/model.safetensors` (tải thủ công từ repo Hugging Face `timm/vit_small_patch16_dinov3.lvd1689m`, ~86MB, SHA-256 `2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040`), nên chạy được khi không có mạng. Thiếu file này thì Stage 1 báo lỗi. Muốn tải tự động qua mạng thì truyền `--dinov3_weights ""`. Stage 3 không cần file này vì checkpoint Stage 1 đã chứa trọng số DINOv3.

---

## 4. Chạy thử nghiệm tuần tự

Luôn `cd` vào `dust_dinov3/` trước khi chạy. Model, log và feature sẽ được lưu bên trong `dust_dinov3/`.

```bash
cd dust_dinov3
```

Dữ liệu đầu vào (`../data/*.pkl`) có thể là toy data (`python gen_data.py` chạy từ thư mục gốc) hoặc UAH-DriveSet tạo bằng `prepare_uah_data.py` (xem mục 7 của [README_DATA.md](README_DATA.md)):

```bash
python prepare_uah_data.py --uah_root ../UAH-DRIVESET-v1 --out_dir ../data --check_only   # xem report trước
python prepare_uah_data.py --uah_root ../UAH-DRIVESET-v1 --out_dir ../data
```

Chạy trên Google Colab GPU: mở `run_colab.ipynb` bằng extension Google Colab trong VS Code, chọn kernel Colab có GPU, chạy lần lượt các cell.

### Stage 1: Train DINOv3 video encoder
```bash
python step_dinov3_3d.py --batch_size 9 --num_epochs 100
```
Tham số riêng của DINOv3:

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--dinov3_model` | `vit_small_patch16_dinov3.lvd1689m` | Tên model DINOv3 trong timm |
| `--dinov3_weights` | `dust_dinov3/model.safetensors` | File trọng số local. `""` để tải từ Hugging Face Hub |
| `--temporal_layers` | `2` | Số layer temporal transformer |
| `--temporal_heads` | auto (384/64 = 6) | Số head temporal |
| `--chunk_size` | `16` | Số frame qua DINOv3 mỗi lần, giảm nếu OOM |

Output: `./saved_models/step_3d/final_3d.pth`, `eX_acc.pth`, `val_loss.npy`, `val_acc.npy`.

### Stage 2: Train 1D Swin trên kinematic
```bash
python step_fusion_1d.py --batch_size 2000 --num_epochs 200
```
Output: `./saved_models/step_1d/final_1d.pth`.

### Stage 3: Trích video feature
```bash
python step_dinov3_3d_get_feature.py --batch_size 8
```
Nếu Stage 1 dùng `--dinov3_model`, `--temporal_layers`, `--temporal_heads` hoặc `--num_frames` khác mặc định, phải truyền đúng giá trị đó ở đây (`--dinov3_model_name` cho tên model). Output: `./video_features/train_3d_features.pkl`, `val_3d_features.pkl`, shape `(N, 1024)`.

### Stage 4: Train fusion
```bash
python step_fusion.py --batch_size 2000 --num_epochs 200
```
Output: `./saved_models/step_fusion/final_fusion.pth`, `val_loss.npy`, `val_acc.npy`.

---

## 5. Lưu ý

- **Chuẩn hoá ảnh**: DINOv3 được pretrain với ảnh RGB scale `[0, 1]` rồi chuẩn hoá ImageNet (`mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`). Code không tự chuẩn hoá (giống bản gốc), nên video trong `.pkl` phải được chuẩn hoá sẵn khi tạo dữ liệu. Xem [README_DATA.md](README_DATA.md).
- **RAM**: giống bản gốc, toàn bộ `video_*.pkl` được nạp vào RAM. Một mẫu `(3, 77, 224, 224)` chiếm ~46MB ở float32 (~93MB ở float64).
- **Tốc độ**: backbone frozen nhưng vẫn chạy lại DINOv3 cho mọi frame ở mỗi epoch Stage 1.
- **Kiểm thử**: cả 4 stage đã chạy thông trên CPU với toy data nhỏ (8 frame). Stage 1 và 3 đã chạy offline với `model.safetensors`, trọng số DINOv3 trong `final_3d.pth` trùng khớp file gốc. Feature Stage 3 có shape `(N, 1024)`, output timm đúng là CLS token. Chưa chạy với dữ liệu UAH thật.
