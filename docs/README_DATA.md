# UAH-DriveSet v1: ghi chú dữ liệu cho DuST-DINOv3

Tài liệu tóm tắt bộ dữ liệu `UAH-DRIVESET-v1/` để không phải mở lại thư mục data. Các con số bên dưới được đo trực tiếp trên bản dữ liệu trong project (2026-10-18). Ý nghĩa cột của các file `RAW_*`, `PROC_*`, `EVENTS_*` lấy theo paper gốc vì README đi kèm dataset chỉ mô tả 2 file `SEMANTIC_*`. Những chỗ chưa chắc được ghi rõ là "chưa xác minh".

- Paper: E. Romera, L.M. Bergasa, R. Arroyo, "Need Data for Driving Behavior Analysis? Presenting the Public UAH-DriveSet", IEEE ITSC 2016.
- Công cụ đọc chính thức: https://github.com/Eromera/uah_driveset_reader
- Tác giả cho phép dùng cho nghiên cứu, cần trích dẫn paper trên. Xem `UAH-DRIVESET-v1/UAH-DRIVESET-v1/LICENSE`.

---

## 1. Tổng quan

- Thu bằng app DriveSafe trên iPhone gắn kính lái: camera trước + IMU + GPS.
- 6 tài xế (D1–D6), mỗi người một xe, 3 hành vi: **NORMAL**, **AGGRESSIVE**, **DROWSY**.
- 2 loại đường: **MOTORWAY** (~25 km) và **SECONDARY** (~16 km).
- Tổng **40 chuyến**, ~500 phút, dung lượng ~3.4 GB (chủ yếu là video).

| Hành vi \ Đường | MOTORWAY | SECONDARY | Tổng |
|---|---|---|---|
| NORMAL | 6 | 11 | 17 |
| AGGRESSIVE | 6 | 5 | 11 |
| DROWSY | 6 | 6 | 12 |

- NORMAL SECONDARY thường có 2 chuyến/tài xế (`NORMAL1`, `NORMAL2`).
- D6 chỉ có 5 chuyến (thiếu AGGRESSIVE-SECONDARY và chỉ 1 NORMAL-SECONDARY). D1–D5 mỗi người 7 chuyến.

---

## 2. Cấu trúc thư mục

```
UAH-DRIVESET-v1/UAH-DRIVESET-v1/
├── README, LICENSE
├── D1/ ... D6/
│   └── <YYYYMMDDhhmmss>-<km>km-<Driver>-<BEHAVIOR>[1|2]-<ROAD>/
│       ├── <YYYYMMDDhhmmss>-....mp4      # video camera trước
│       ├── RAW_GPS.txt                   # ~1 Hz
│       ├── RAW_ACCELEROMETERS.txt        # ~10 Hz
│       ├── PROC_LANE_DETECTION.txt       # ~22-26 Hz
│       ├── PROC_VEHICLE_DETECTION.txt    # ~3-8 Hz
│       ├── PROC_OPENSTREETMAP_DATA.txt   # thưa, theo sự kiện
│       ├── EVENTS_INERTIAL.txt           # danh sách sự kiện
│       ├── EVENTS_LIST_LANE_CHANGES.txt  # danh sách sự kiện
│       ├── SEMANTIC_ONLINE.txt           # ~1 Hz
│       ├── SEMANTIC_FINAL.txt            # 54 giá trị tổng kết
│       └── *.txt~                        # file backup của editor, BỎ QUA
```

- Ví dụ tên chuyến: `20151110175712-16km-D1-NORMAL1-SECONDARY`.
- Nhãn hành vi và loại đường lấy trực tiếp từ tên thư mục.
- Mọi file `.txt` phân tách bằng khoảng trắng, không có header. Cột 1 luôn là timestamp tính bằng giây kể từ lúc bắt đầu chuyến (trừ `SEMANTIC_FINAL`).

---

## 3. Video

| Thuộc tính | Giá trị |
|---|---|
| Độ phân giải | 960×540 |
| FPS | 29.97 (ngoại lệ: D1-NORMAL2-SECONDARY 28.55, D4-DROWSY-SECONDARY 25.99) |
| Thời lượng | 532–1198 s mỗi chuyến |
| Định dạng | `.mp4`, đọc được bằng OpenCV |

Vì FPS không đồng nhất, hãy lấy frame theo **thời gian** (`CAP_PROP_POS_MSEC` hoặc `frame_idx = round(t * fps)` với fps của từng video), đừng giả định 30 fps.

---

## 4. Ý nghĩa cột từng file

### RAW_ACCELEROMETERS.txt (~10 Hz, 11 cột)
| # | Ý nghĩa |
|---|---|
| 1 | Timestamp (s) |
| 2 | Hệ thống kích hoạt (1 = đang hoạt động, 0 = chưa) |
| 3–5 | Gia tốc X, Y, Z thô (đơn vị G) |
| 6–8 | Gia tốc X, Y, Z đã lọc Kalman (G) |
| 9–11 | Roll, Pitch, Yaw. Paper ghi độ, nhưng dữ liệu thực tế là **radian** (yaw nằm trong ±π, pitch ≈ −1.5 do điện thoại dựng đứng) |

Theo paper, trục X là ngang xe, Y là dọc xe, Z là thẳng đứng. **Dữ liệu không khớp với mô tả đó.** Đối chiếu với gia tốc dọc tính từ tốc độ GPS trên 14 chuyến, trục **Z** tương quan rõ nhất (0.1–0.5), trong khi X và Y gần như bằng 0. Điều này hợp lý vì điện thoại dựng đứng trên kính lái (pitch ≈ −1.5 rad), nên trục Z của máy hướng về phía trước xe. Không trục nào khớp ổn định với gia tốc ngang. Vì vậy không nên gắn nhãn kiểu "phanh" hay "đánh lái" cho từng trục.

Lỗi đã phát hiện trong file này (đo trên toàn bộ 40 chuyến):
- **Trùng timestamp**: hầu hết chuyến có vài dòng cùng timestamp nhưng giá trị khác nhau. Đây là các mẫu bị dồn buffer (thường ở đầu chuyến hoặc ngay sau một khoảng mất dữ liệu), không biết thời điểm thật, nên bị bỏ cả nhóm.
- **Timestamp quay ngược**: D6-DROWSY-MOTORWAY nhảy từ 1077.63 s về 1067.90 s ở cuối file, phần sau là một đoạn ghi lại với giá trị khác (RAW_GPS cũng vậy). Script bỏ phần sau lần quay ngược. Khoảng 1067.90–1077.63 s có hai bản ghi mâu thuẫn, không biết bản nào đúng, nên mọi cửa sổ chạm vào khoảng này đều bị loại.
- **Mất dữ liệu**: D3-NORMAL-MOTORWAY có khoảng trống 8 s (80.29 s → 88.27 s), D6-NORMAL-MOTORWAY có khoảng trống ~1 s. Cửa sổ chứa khoảng trống > 0.5 s bị loại.
- **Cột 2 = 0** (hệ thống chưa kích hoạt): ~2% số dòng, chủ yếu đầu/cuối chuyến. Cửa sổ chứa dòng như vậy bị loại.
- **Yaw hỏng ở D1-DROWSY-MOTORWAY**: yaw rate gần như không tương quan với tốc độ đổi hướng GPS (0.01, các chuyến khác khoảng ±0.8).

### RAW_GPS.txt (~1 Hz, 12 cột)
| # | Ý nghĩa |
|---|---|
| 1 | Timestamp (s) |
| 2 | Tốc độ (km/h) |
| 3, 4 | Vĩ độ, kinh độ |
| 5 | Độ cao (m) |
| 6, 7 | Độ chính xác dọc, ngang (m) |
| 8 | Hướng đi (độ) |
| 9 | Biến thiên hướng đi (độ) |
| 10–12 | Không được mô tả (giá trị chủ yếu là số nguyên nhỏ). Bỏ qua |

### PROC_LANE_DETECTION.txt (~22–26 Hz, 5 cột)
| # | Ý nghĩa |
|---|---|
| 1 | Timestamp (s) |
| 2 | Vị trí xe so với tâm làn X (m) |
| 3 | Góc xe so với làn Phi (độ) |
| 4 | Bề rộng làn W (m) |
| 5 | Trạng thái detector. `-9` = chưa có dữ liệu (đầu chuyến). Theo paper: `-1` calibrating, `0` initializing, `1` không thấy làn, `2` bám làn tốt (chiếm ~87% số dòng) |

Khi cột 5 khác `2`, các cột 2–4 không đáng tin (thường là `-9` hoặc `0`).

### PROC_VEHICLE_DETECTION.txt (~3–8 Hz, 5 cột)
| # | Ý nghĩa |
|---|---|
| 1 | Timestamp (s) |
| 2 | Khoảng cách tới xe phía trước (m), `-1` nếu không có |
| 3 | Thời gian tới va chạm (s), `-1` nếu không có |
| 4 | Số xe phát hiện được (0–5) |
| 5 | Tốc độ GPS (km/h) |

### PROC_OPENSTREETMAP_DATA.txt (thưa, 10 cột)
| # | Ý nghĩa |
|---|---|
| 1 | Timestamp (s) |
| 2 | Giới hạn tốc độ (km/h) |
| 3 | Độ tin cậy giới hạn tốc độ (giá trị −1…3, chưa xác minh ý nghĩa từng mức) |
| 4 | Loại đường OSM (`motorway`, `primary`, `motorway_link`, ..., `-` = không rõ) |
| 5 | Số làn theo chiều đi |
| 6 | Làn hiện tại |
| 7, 8 | Vĩ độ, kinh độ |
| 9 | Độ trễ truy vấn OSM (s) |
| 10 | Tốc độ (km/h) |

### EVENTS_INERTIAL.txt (6 cột, mỗi dòng 1 sự kiện)
| # | Ý nghĩa |
|---|---|
| 1 | Timestamp (s) |
| 2 | Loại: `1` phanh, `2` rẽ, `3` tăng tốc |
| 3 | Mức độ: `1` thấp, `2` trung bình, `3` cao |
| 4, 5 | Vĩ độ, kinh độ |
| 6 | Thời điểm tuyệt đối `YYYYMMDDhhmmss` |

Có 1 dòng rỗng trong toàn bộ dataset. Parser cần bỏ qua dòng trống.

### EVENTS_LIST_LANE_CHANGES.txt (6 cột)
| # | Ý nghĩa |
|---|---|
| 1 | Timestamp (s) |
| 2 | Loại chuyển làn: dấu là hướng (trái/phải), `±2` là chuyển làn nhanh/bất thường (theo paper, chưa xác minh hướng của dấu) |
| 3, 4 | Vĩ độ, kinh độ |
| 5 | Thời gian chuyển làn (s) |
| 6 | Ngưỡng thời gian để coi là bất thường (s) |

### SEMANTIC_ONLINE.txt (~1 Hz, 27 cột) và SEMANTIC_FINAL.txt (54 giá trị)
Đã mô tả đầy đủ trong `UAH-DRIVESET-v1/UAH-DRIVESET-v1/README`. Tóm tắt:
- `SEMANTIC_ONLINE`: timestamp, GPS, các điểm số 0–100 (tổng, tăng tốc, phanh, rẽ, weaving, drifting, quá tốc độ, bám xe) và tỉ lệ normal/drowsy/aggressive do DriveSafe ước lượng. Cột 4–15 tính trên cửa sổ 60 s gần nhất, cột 16–27 tính luỹ kế từ đầu chuyến.
- `SEMANTIC_FINAL`: mỗi dòng 1 giá trị tổng kết chuyến. Dòng 11 có dạng `16 km` (có chữ), nên đọc từng dòng dạng chuỗi rồi mới ép kiểu, đừng dùng `np.loadtxt` cho cả file.
- Đây là **output của thuật toán DriveSafe**, không phải ground truth. Không đưa vào input của model nếu muốn đánh giá công bằng.

---

## 5. Đồng bộ video và dữ liệu cảm biến

Timestamp trong các file `.txt` tính từ thời điểm bắt đầu chuyến (tên thư mục). Video bắt đầu ở thời điểm khác (tên file `.mp4`).

```
offset     = route_start - video_start          (giây, từ 14 ký tự đầu của tên)
video_time = data_time + offset
```

- Phần lớn chuyến có offset trong khoảng −22 s đến +37 s.
- **Offset theo tên file chỉ là gần đúng.** Kiểm chứng bằng dữ liệu (mục 7) cho thấy offset thật thường **muộn hơn 1–2.6 s** so với offset theo tên file. Tên file chỉ có độ phân giải 1 s, và camera có thể cần thời gian khởi động. Với cửa sổ 5 s, lệch 2 s là đáng kể, nên phải dùng offset đã kiểm chứng.
- **D5-AGGRESSIVE-SECONDARY**: offset +137 s (video bắt đầu sớm hơn 2 phút). Dữ liệu cảm biến chỉ có từ 13.9 s đến 441.8 s.
- **D2-DROWSY-SECONDARY** (`20151120164606`): tên video `20151120165604` cho offset −598 s, khiến chỉ 9% dữ liệu khớp với video. Đây là lỗi gõ phút trong tên video, **đã kiểm chứng**: quét toàn bộ độ trễ tìm ra đỉnh tương quan tại +3.6 s. Khi thử giả thuyết offset +2 s, yaw rate cho đỉnh tại +4.4 s với tương quan 0.82, quét rộng cũng ra +4.4 s. Notebook truyền sẵn `--offset_override 20151120164606-16km-D2-DROWSY-SECONDARY=2`.
- **D1-DROWSY-MOTORWAY**: yaw của điện thoại hỏng, và hướng GPS cũng không cho đỉnh tương quan rõ ràng. Chuyến này **bị loại** vì không kiểm chứng được đồng bộ.

Kết quả kiểm chứng trên 40 chuyến: 38 chuyến đạt ngay (34 bằng yaw rate, 4 bằng hướng GPS), D2-DROWSY-SECONDARY đạt khi dùng override, D1-DROWSY-MOTORWAY bị loại. Như vậy dùng được **39/40 chuyến**.
- **D3-DROWSY-SECONDARY**: video chỉ dài 566 s, trong khi dữ liệu kéo dài tới 668 s. ~100 s cuối không có video, cần bỏ khi cắt cửa sổ.
- Khi cắt cửa sổ, chỉ giữ cửa sổ nằm trọn trong cả video lẫn dữ liệu cảm biến.

---

## 6. Chuyển UAH-DriveSet sang định dạng của pipeline DuST

Pipeline (gốc và DINOv3) đọc 6 file pickle trong `data/`:

| File | Shape | Kiểu | Ghi chú |
|---|---|---|---|
| `video_train.pkl`, `video_val.pkl` | `(N, 3, 77, 224, 224)` | float32 | Kênh RGB, `T=77` frame |
| `kine_train.pkl`, `kine_val.pkl` | `(N, 3, 51)` | float | 3 kênh, 51 mẫu |
| `label_train.pkl`, `label_val.pkl` | `(N,)` | int64 | Giá trị trong `0..3` (model có 4 lớp) |

Mẫu thứ `i` của video, kine và label phải cùng một cửa sổ thời gian. Script `prepare_uah_data.py` (mục 7) làm việc này như sau:

1. **Cửa sổ**: 51 mẫu gia tốc liên tiếp, tức khoảng 5 s ở 10 Hz. Cửa sổ ứng viên cách nhau 5 s.
2. **Kinematic**: 3 kênh gia tốc đã lọc Kalman X, Y, Z (cột 6–8), giữ nguyên giá trị thật, **không nội suy**. Chuẩn hoá z-score theo thống kê tập train.
3. **Video**: 77 frame thật trải đều trên đúng khoảng thời gian của 51 mẫu đó (~15 fps), quy đổi bằng offset đã kiểm chứng ở mục 5. Cắt vuông giữa, resize 224×224, BGR→RGB, chia 255, chuẩn hoá ImageNet `mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)` như lúc DINOv3 được pretrain. Đổi trục về `(3, 77, 224, 224)`.
4. **Nhãn** theo tên thư mục: `NORMAL → 0`, `AGGRESSIVE → 1`, `DROWSY → 2`. Model giữ 4 output như bản gốc, lớp 3 không dùng, không ảnh hưởng tới training. Nhãn là hành vi được yêu cầu cho **cả chuyến**, nên một cửa sổ trong chuyến DROWSY có thể trông bình thường. Đây là nhiễu nhãn vốn có của dataset.
5. **Chia tập theo tài xế** (mặc định D6 là val). Không chia ngẫu nhiên theo cửa sổ vì các cửa sổ trong cùng chuyến rất giống nhau và sẽ rò rỉ sang tập val.
6. **Lưu** video dạng float16 để tiết kiệm một nửa RAM. Stage 1 và 3 tự đổi sang float32 theo từng batch.

### Lưu ý dung lượng
- Một mẫu video `(3, 77, 224, 224)` chiếm ~23 MB ở float16 (~46 MB ở float32). Code nạp toàn bộ pickle vào RAM.
- Với ~500 phút dữ liệu, cửa sổ 5 s không chồng lấn cho ~5800 mẫu, tức ~135 GB. Vì vậy script mặc định chỉ lấy tối đa 8 cửa sổ mỗi chuyến, vừa RAM của Colab miễn phí (~12 GB). Có nhiều RAM hơn thì tăng `--max_windows_per_trip`.
- Nếu đổi số frame hay số mẫu kinematic, truyền `--num_frames` tương ứng cho các script (Stage 1 và 3 dùng chung một giá trị, Stage 2 và 4 dùng chung một giá trị).

---

## 7. Script làm sạch: `prepare_uah_data.py`

Script hiện thực mục 5 và 6 với nguyên tắc **không sinh dữ liệu**: mọi giá trị đầu ra là mẫu thật, cửa sổ nào không đạt thì bị loại chứ không sửa hay lấp.

```bash
cd dust_dinov3
python prepare_uah_data.py --uah_root ../UAH-DRIVESET-v1 --out_dir ../data --check_only   # chỉ kiểm tra, ghi report
python prepare_uah_data.py --uah_root ../UAH-DRIVESET-v1 --out_dir ../data                # tạo pickle
```

### Các kiểm tra

| Cấp | Kiểm tra | Xử lý khi sai |
|---|---|---|
| Chuyến | Tên thư mục đúng định dạng, đúng 1 file `.mp4`, có `RAW_ACCELEROMETERS` và `RAW_GPS`, video giải mã được | Loại chuyến |
| Chuyến | Timestamp quay ngược | Chỉ giữ phần trước lần quay ngược. Khoảng thời gian bị ghi trùng là mơ hồ, cửa sổ chạm vào bị loại |
| Chuyến | Dòng trùng timestamp | Bỏ cả nhóm |
| Chuyến | **Đồng bộ video–cảm biến** (xem dưới) | Loại chuyến nếu không kiểm chứng được |
| Cửa sổ | Khoảng cách giữa 2 mẫu gia tốc > 0.5 s, hoặc 51 mẫu trải dài ngoài 4.6–5.6 s | Loại cửa sổ |
| Cửa sổ | Có dòng cờ kích hoạt = 0 | Loại cửa sổ |
| Cửa sổ | Thời điểm quy đổi nằm ngoài video | Loại cửa sổ |
| Cửa sổ | Frame gần nhất lệch quá nửa chu kỳ frame, hoặc video mất frame (> 3 chu kỳ) trong cửa sổ | Loại cửa sổ |
| Cửa sổ | Frame tối (độ sáng trung bình < 15) | Loại cửa sổ |
| Cửa sổ | Video đứng hình (> 50% cặp frame liền nhau gần như giống hệt) trong khi GPS báo xe chạy > 5 km/h | Loại cửa sổ |

**Kiểm chứng đồng bộ.** Tên file chỉ có độ phân giải 1 giây và có thể gõ nhầm, nên script kiểm chứng bằng chính dữ liệu:
- Tín hiệu video: median optical flow theo phương ngang ở dải chân trời (bỏ trời, capo và cột A bên phải). Khi xe quay, cảnh ở xa trôi ngang.
- Tín hiệu cảm biến: yaw rate từ cột yaw của `RAW_ACCELEROMETERS`. Nếu không đạt thì thử tốc độ đổi hướng GPS.
- Hai tín hiệu được lọc thông dải (làm mượt 1 s, trừ trung bình trượt 30 s), rồi tính tương quan theo từng độ trễ, bước 0.2 s.
- Chuyến **đạt** khi trong khoảng ±10 s quanh offset gốc có đỉnh tương quan dương ≥ 0.15, cao gấp ≥ 1.5 lần đỉnh thứ hai (cách > 10 s), **và** quét toàn bộ độ trễ khả dĩ cũng ra đỉnh ở cùng chỗ (±2 s).
- Offset dùng để cắt là offset đỉnh tương quan (độ phân giải 0.2 s), không phải offset theo tên file.
- `--offset_override TRIP=GIÂY` cho phép thử một giả thuyết offset khác (vd. lỗi gõ tên file). Giả thuyết đó vẫn phải qua đúng kiểm chứng trên mới được dùng.

### Lấy mẫu trong một cửa sổ
- **Kinematic**: 51 dòng `RAW_ACCELEROMETERS` liên tiếp đã qua làm sạch, cột 6–8, giữ nguyên giá trị. Không nội suy.
- **Video**: 77 thời điểm trải đều trên đúng khoảng `[t_mẫu_đầu, t_mẫu_cuối]` của 51 dòng đó, cộng offset đã kiểm chứng. Mỗi thời điểm lấy frame thật gần nhất theo timestamp giải mã. Cắt vuông ở giữa, resize 224×224, RGB, chuẩn hoá ImageNet.
- Cửa sổ ứng viên cách nhau 5 s (không chồng lấn). Mỗi chuyến chọn tối đa `--max_windows_per_trip` cửa sổ hợp lệ, trải đều theo thời gian.

### File đầu ra (trong `--out_dir`)

| File | Nội dung |
|---|---|
| `video_{train,val}.pkl` | `(N, 3, 77, 224, 224)`, float16 mặc định |
| `kine_{train,val}.pkl` | `(N, 3, 51)` float32, z-score theo thống kê tập train |
| `label_{train,val}.pkl` | `(N,)` int64: 0 NORMAL, 1 AGGRESSIVE, 2 DROWSY |
| `windows_{train,val}.csv` | Mỗi dòng mô tả 1 mẫu, **cùng thứ tự với pickle**: chuyến, tài xế, nhãn, thời điểm dữ liệu, offset, thời điểm video, tốc độ GPS |
| `kine_norm_stats.json` | Mean/std từng kênh dùng để chuẩn hoá |
| `uah_trip_report.csv` | Mỗi chuyến: trạng thái, lý do loại, số dòng bị bỏ khi làm sạch, kết quả kiểm chứng đồng bộ, số cửa sổ hợp lệ/được chọn/được ghi và lý do loại cửa sổ |

Mặc định val là tài xế D6, các tài xế còn lại là train.
