---
title: Kiem Tra Hanh Vi Lai Xe
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.25.0
python_version: "3.10"
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Kiểm tra hành vi lái xe từ cảm biến và video (thử nghiệm)
---

# Web kiểm tra hành vi lái xe (bản thử nghiệm)

Web giúp nhân viên an toàn của công ty xe dịch vụ xem lại một chuyến đi của tài xế. AI chia chuyến thành các đoạn 5 giây và cho biết mỗi đoạn giống kiểu lái nào nhất: **Bình thường**, **Lái xe hung hăng** hay **Dấu hiệu buồn ngủ**. Web tổng hợp thành kết luận, dòng thời gian, danh sách đoạn cần xem lại, khung hình video và biểu đồ chi tiết.

Web chạy cả 3 model của DuST-DINOv3 và cho xem kết quả theo từng cách:

| Cách phân tích | Model | File trong `weights/` |
|---|---|---|
| Chỉ cảm biến | 1D Swin (Stage 2) | `final_1d.pth` + `kine_norm_stats.json` |
| Chỉ video | DINOv3 ViT-S/16 + Temporal Transformer (Stage 1) | `video_3d.pth` |
| Kết hợp | 1D Swin fusion (Stage 4) | `fusion.pth` |

## Cấu trúc

| File | Vai trò |
|---|---|
| `app.py` | FastAPI: trang web, kiểm tra file khi upload, hàng đợi xử lý, hàm GPU cho ZeroGPU |
| `inference.py` | Đọc và kiểm tra file cảm biến, làm sạch và cắt đoạn giống lúc train, chạy 3 model, tổng hợp kết quả |
| `video_pipeline.py` | Đồng bộ video với cảm biến, lấy 77 frame mỗi đoạn, chạy DINOv3, tạo ảnh thu nhỏ |
| `swin1d.py`, `swin_fusion.py`, `dinov3_encoder.py` | Bản sao nguyên vẹn mã model của project, để thư mục này tự chạy được |
| `static/index.html` | Giao diện, một file HTML + JavaScript thuần |
| `tools/precompute_demo.py` | Tạo dữ liệu chuyến mẫu (chạy 1 lần trên máy) |
| `weights/` | 4 file model ở bảng trên |

## Dữ liệu đầu vào

| File | Bắt buộc | Giới hạn |
|---|---|---|
| `RAW_ACCELEROMETERS.txt` (UAH-DriveSet, 11 cột) | Có | 20 MB |
| `RAW_GPS.txt` (12 cột) | Không. Dùng để hiện quãng đường, tốc độ và hỗ trợ đồng bộ video | 20 MB |
| Video hành trình `.mp4` | Không. Có video thì chạy thêm nhánh video và fusion | 300 MB |

- Có thể chọn cả thư mục chuyến đi, web tự tìm 3 file trên.
- Chuyến đi phải dài từ 1 phút đến 2 giờ.
- **Nhánh video** cần đồng bộ được video với cảm biến (cùng cách kiểm chứng như lúc tạo dữ liệu train). Không đồng bộ được thì web chỉ hiện kết quả cảm biến và báo rõ lý do.
- **DINOv3** chạy trên GPU qua ZeroGPU. Nếu hết hạn mức GPU, web chuyển sang CPU và chỉ phân tích video 24 đoạn trải đều trong chuyến (đổi bằng biến môi trường `VIDEO_CPU_MAX_WINDOWS`).

## Các trường hợp lỗi được xử lý

| Trường hợp | Kết quả |
|---|---|
| File quá lớn | Từ chối ngay trên trình duyệt; máy chủ đọc theo từng phần và dừng khi vượt giới hạn (HTTP 413) |
| Sai loại file (video ở ô văn bản, `.exe`, file nhị phân, sai mã hoá) | Báo lỗi rõ ô nào sai (HTTP 400) |
| Sai số cột, dùng dấu phẩy, giá trị không phải số, NaN | Báo dòng bị lỗi |
| Chọn nhầm file GPS và cảm biến | Nhận ra và gợi ý file đúng |
| Chuyến dưới 1 phút hoặc trên 2 giờ, không còn đoạn hợp lệ | Báo lý do |
| Video hỏng, không mở được, quá ngắn hoặc quá dài | Báo lỗi trước khi phân tích |
| Video không đồng bộ được hoặc không trùng thời gian với cảm biến | Chỉ hiện kết quả cảm biến, kèm cảnh báo |
| Nhiều người cùng phân tích | Xử lý lần lượt, hiện thứ tự chờ; quá 3 yêu cầu thì báo bận (HTTP 429) |
| Mất mạng, phân tích quá 20 phút | Thông báo và cho phép thử lại |

## Chuyến mẫu

Nút "Phân tích chuyến mẫu" phân tích sẵn một chuyến của tài xế D6 (tài xế không có trong dữ liệu train). Dữ liệu chuyến mẫu nằm trong **dataset private** `huy-le/checking-driver-demo`, Space tải về lúc khởi động bằng secret `HF_TOKEN`. Web hiển thị khung hình video nhưng không có đường nào tải file gốc. Đặc trưng DINOv3 của chuyến mẫu được tính sẵn để demo luôn chạy được; Temporal Transformer, nhánh video và fusion vẫn chạy trực tiếp mỗi lần bấm.

Tạo lại chuyến mẫu:

```bash
cd web
python tools/precompute_demo.py --trip "<thư mục chuyến UAH>" --out demo
hf upload huy-le/checking-driver-demo demo . --repo-type=dataset --private
```

## Chạy trên máy

```bash
cd web
pip install -r requirements.txt fastapi uvicorn python-multipart huggingface_hub
python app.py
```

Mở http://localhost:7860. Nếu có thư mục `web/demo/` thì nút chuyến mẫu hoạt động.

## Deploy lên Hugging Face Spaces

1. Space dùng **SDK Gradio**, phần cứng **ZeroGPU** (mặc định của tài khoản miễn phí). ZeroGPU luôn dùng Python 3.10, cần ít nhất 1 hàm `@spaces.GPU` và chỉ nhận một số bản torch; `app.py` và `requirements.txt` đã xử lý.
2. Upload thư mục `web/` (trừ `demo/`, vì dữ liệu chuyến mẫu chỉ nằm trong dataset private):

   ```bash
   hf upload huy-le/Checking-Driver web . --repo-type=space --exclude "demo/*" --exclude "__pycache__/*"
   ```

3. Trong Settings của Space, thêm secret `HF_TOKEN` (token có quyền đọc dataset private).

Space miễn phí sẽ ngủ sau 48 giờ không có người dùng. Lần mở tiếp theo mất khoảng 1–2 phút để khởi động lại.

## Giới hạn cần biết

- **Model đang dùng là bản thử nghiệm**: video lấy checkpoint epoch 18 (val 57.5%), fusion epoch 17 (val 42.5%), cảm biến val 32.5%. Kết quả chưa đủ tin cậy để đánh giá tài xế.
- **Bản quyền dữ liệu**: UAH-DriveSet chỉ dùng phi thương mại và cấm phát tán lại dữ liệu. Để dùng thật cho công ty xe dịch vụ cần train lại trên dữ liệu của chính công ty.
- **Nhãn**: UAH gán một nhãn cho cả chuyến, nên kết quả từng đoạn 5 giây chỉ là mức "giống kiểu lái nào".
- **Ngưỡng hiển thị** (đánh dấu đoạn khi AI chắc chắn ≥ 60%, mức "Cao" khi ≥ 80%, kết luận theo tỉ lệ 15% / 40%) là ngưỡng đặt tay, chưa được hiệu chỉnh.

Trích dẫn dữ liệu: E. Romera, L.M. Bergasa, R. Arroyo, "Need Data for Driving Behavior Analysis? Presenting the Public UAH-DriveSet", IEEE ITSC 2016.
