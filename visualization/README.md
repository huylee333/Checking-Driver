# Trực quan hoá dữ liệu huấn luyện

Thư mục này vẽ 6 hình mô tả dữ liệu UAH-DriveSet **đúng như mô hình DuST-DINOv3 dùng để train**: pickle video/gia tốc/nhãn, `windows_*.csv` và `uah_trip_report.csv` trong `runs/dinov3_vits16_run1/data`. Chú thích trong hình dùng tiếng Anh.

```bash
cd dust_dinov3
python visualization/make_figures.py                 # vẽ cả 6 hình vào visualization/figures/
python visualization/make_figures.py --only 2 4      # chỉ vẽ lại vài hình
```

Hình 4a cần thư mục dữ liệu gốc (`--uah`, mặc định `../UAH-DRIVESET-v1/DRIVERSET_H`) để chạy lại bước đồng bộ trên một chuyến. Hình 6 đọc log train (`--logs`, mặc định `runs/dinov3_vits16_run1/logs`). Các hình còn lại chỉ cần thư mục `data` của lần train.

## Chọn hình theo cách các bài báo trình bày dữ liệu

| Hình | Nội dung | Cơ sở |
|---|---|---|
| 1 | Thành phần dữ liệu | Bài giới thiệu UAH-DriveSet trình bày số chuyến theo tài xế, hành vi, loại đường; các bài ML luôn báo phân bố lớp và cách chia train/val |
| 2 | Mẫu đa phương thức | Bài DuST và UAH-DriveSet đều minh hoạ khung hình video kèm tín hiệu cảm biến cùng thời điểm |
| 3 | Phân bố đặc trưng theo lớp | Cách chuẩn để cho thấy các lớp tách biệt đến đâu trên dữ liệu thô |
| 4 | Kiểm chứng đồng bộ | Bước tiền xử lý quan trọng nhất của project, cần chứng minh bằng dữ liệu |
| 5 | Lọc dữ liệu | Minh bạch quy trình làm sạch: dữ liệu chỉ bị loại, không bị sinh hay sửa |

## Các hình và điều rút ra

**Hình 1: `fig1_dataset_composition.png`.** 39/40 chuyến dùng được; chuyến D1 drowsy motorway bị loại vì không kiểm chứng được đồng bộ video. Train là tài xế D1–D5, val là D6. Tập train lệch lớp nhẹ (Normal chiếm nhiều nhất), tập val ít mẫu Aggressive vì D6 chỉ có 1 chuyến hung hăng.

| | Normal | Aggressive | Drowsy | Tổng |
|---|---|---|---|---|
| Train (đoạn 5 s) | 120 | 80 | 72 | 272 |
| Val (đoạn 5 s) | 16 | 8 | 16 | 40 |

**Hình 2: `fig2_multimodal_samples.png`.** Mỗi lớp một mẫu của tập val, hiển thị đúng dữ liệu đưa vào mô hình: 5 trong 77 khung hình (đã cắt vuông 224×224) và 51 mẫu gia tốc 3 trục của cùng 5 giây. Mẫu được chọn khách quan: đoạn có độ biến thiên gia tốc gần trung vị của lớp nhất, không chọn đoạn "đẹp".

**Hình 3: `fig3_acceleration_variability.png`.** Độ lệch chuẩn gia tốc trong mỗi đoạn, chung một trục log để so sánh giữa các trục.

| Trung vị độ lệch chuẩn (g) | X | Y | Z |
|---|---|---|---|
| Normal | 0.021 | 0.015 | 0.013 |
| Aggressive | 0.023 | 0.020 | 0.025 |
| Drowsy | 0.020 | 0.022 | 0.015 |

Aggressive biến thiên mạnh nhất ở trục Z, trục tương ứng phanh và tăng tốc với cách gắn điện thoại của UAH. Drowsy cao ở trục Y. Các hộp chồng lấn nhau nhiều, nên một đoạn 5 giây đơn lẻ khó phân loại chỉ bằng gia tốc. Điều này khớp với độ chính xác thấp của nhánh 1D.

**Hình 4: `fig4_sync_verification.png`.** (a) Tương quan giữa chuyển động ngang của video và yaw rate theo từng độ lệch thời gian, trên chuyến D6 normal secondary: đỉnh rõ tại −8.8 s, lệch 1.2 s so với offset theo tên file (−10 s). (b) Độ lệch cần sửa so với tên file trên 39 chuyến: trung vị +1.8 s, khoảng −0.2 đến +2.6 s. 35 chuyến kiểm chứng bằng yaw rate, 4 chuyến bằng hướng GPS.

**Hình 5: `fig5_window_filtering.png`.** 6.051 đoạn ứng viên, còn 5.866 đoạn hợp lệ sau làm sạch, 312 đoạn được dùng để train (8 đoạn mỗi chuyến, giới hạn bởi RAM của nhánh video). 185 đoạn bị loại: cảm biến chưa kích hoạt 158, ngoài phạm vi video 22, khoảng trống mẫu > 0.5 s là 3, đoạn bị ghi trùng 2.

**Hình 6: `fig6_training_curves.png`.** Đường cong train đọc từ log. Stage 2 (1D Swin) overfit: train loss giảm 1.49 → 0.49, val loss tăng sau epoch ~110. Stage 1 bản fine-tune + SupCon (dừng ở epoch 45): SupCon nằm ở ln(15) = 2.71, đúng mức khi đặc trưng không mang thông tin lớp, và val acc đứng ở 40% (bằng đoán lớp đa số).

Ngoài ra thư mục `figures/` có 4 sơ đồ không do script vẽ: `model_pic.png` (kiến trúc DuST gốc), `Pipline_data.png` (pipeline dữ liệu), `dust_dinov3_architecture.svg` và `stage1_training_step.svg` (vẽ tay bằng SVG).

## Kiểm tra bố cục

`make_figures.py` tự kiểm tra mỗi hình trước khi lưu: chữ chồng chữ, chữ tràn khỏi khung và các vùng vẽ chồng nhau. Kết quả lần chạy cuối: 0 lỗi ở cả 6 hình. Các hình cũng đã được xem bằng mắt để bắt chỗ chữ đè lên dữ liệu, loại lỗi mà kiểm tra tự động không phát hiện.

Màu 3 lớp (Normal xanh dương, Aggressive cam, Drowsy xanh lá) giữ cố định ở mọi hình, đã qua kiểm tra mù màu cho mọi cặp. Màu xanh lá có độ tương phản với nền dưới 3:1, nên lớp luôn đi kèm chữ (nhãn trục, tiêu đề) và có bảng số liệu ở trên.
