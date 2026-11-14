"""A03. Làm ngập hàng đợi, từ chối dịch vụ (DoS).

Hai mức tấn công:
  1. Gửi hơn MAX_PENDING yêu cầu NHỎ, hợp lệ cùng lúc -> các yêu cầu thừa phải nhận 429 (test_small_requests...).
  2. Gửi nhiều yêu cầu CÓ FILE GẦN SÁT GIỚI HẠN (19.5 MB) cùng lúc, trong container giới hạn RAM.
     `MAX_PENDING` chỉ đếm job đã được NHẬN vào hàng đợi; việc đọc hết file vào bộ nhớ (`_read_text`,
     `load_sensors`) xảy ra TRƯỚC bước đếm đó, nên về lý thuyết số lượng yêu cầu đang đọc file cùng lúc
     không bị chặn bởi MAX_PENDING -> bộ nhớ dùng có thể tăng theo số kết nối đồng thời, không theo
     MAX_PENDING. Test đo trực tiếp: container có bị kernel giết (OOM, exit 137) không, và có hồi phục
     để trả lời bình thường sau đó không.
"""
import concurrent.futures as cf
import time

import attack_kit as kit

SMALL_TRIP = kit.sensor_text(kit.trip_with(lambda t, rng: rng.normal(0, 0.05, (len(t), 3))))


def test_more_than_max_pending_gets_429(ready_container):
    """MAX_PENDING = 3 (xem web/app.py). Gửi 8 yêu cầu NHỎ, hợp lệ cùng lúc tới /api/analyze: không quá 3
    được nhận (status khác 429), các yêu cầu còn lại phải nhận 429 kèm lý do, không phải lỗi 500."""
    c = ready_container

    def analyze(i):
        return kit.post_multipart(c.url + '/api/analyze', {'acc_file': (f'RAW_ACCELEROMETERS_{i}.txt', SMALL_TRIP)}, timeout=30)

    with cf.ThreadPoolExecutor(8) as ex:
        results = list(ex.map(analyze, range(8)))
    codes = [s for s, _ in results]
    accepted = sum(1 for s in codes if s == 200)
    rejected = sum(1 for s in codes if s == 429)
    others = [s for s in codes if s not in (200, 429)]
    assert not others, f'mã trạng thái không mong đợi: {codes}'
    assert accepted <= 3, f'nhận nhiều hơn MAX_PENDING=3 job cùng lúc: {accepted} (toàn bộ: {codes})'
    assert rejected >= 5, f'mong ít nhất 5 bị 429, được: {codes}'
    # dọn hàng đợi: chờ các job được nhận chạy xong trước khi test khác dùng container này
    deadline = time.time() + 120
    while time.time() < deadline:
        s, body = kit.get_json(c.url + '/api/status')
        if s == 200:
            break
        time.sleep(1)


def test_concurrent_near_max_size_uploads_do_not_crash_the_container(start_limited_container):
    """10 file ~19.3 MB (gần giới hạn 20 MB) gửi CÙNG LÚC tới một container chỉ có 2 GiB RAM (vừa đủ để
    nạp model khi rảnh -- đo thực tế hết khoảng 1.67 GiB/2 GiB ngay khi chưa có yêu cầu nào -- nhưng không
    còn đủ nếu 10 bản sao của file được giữ trong bộ nhớ cùng lúc).
    Trước khi vá (xem nhật ký): việc đọc file vào bộ nhớ không bị giới hạn bởi MAX_PENDING, dẫn tới
    kernel giết container (OOM) khi đủ nhiều kết nối đồng thời. Sau khi vá (giữ chỗ bằng semaphore
    trước khi đọc file): các yêu cầu vượt quá MAX_PENDING bị 429 ngay, không đọc file, nên dùng RAM có
    giới hạn và container phải sống sót."""
    c = start_limited_container(mem='2g', cpus=2)
    payload = kit.big_text_trip(19.3)

    def upload(i):
        t0 = time.time()
        try:
            status, body = kit.post_multipart(c.url + '/api/analyze',
                                               {'acc_file': (f'RAW_ACCELEROMETERS_{i}.txt', payload)}, timeout=120)
            return i, status, round(time.time() - t0, 1)
        except Exception as e:
            return i, f'EXC:{type(e).__name__}', round(time.time() - t0, 1)

    with cf.ThreadPoolExecutor(10) as ex:
        results = list(ex.map(upload, range(10)))

    state = c.inspect('{{.State.Status}} oom={{.State.OOMKilled}}')
    assert 'oom=true' not in state, f'container bị kernel giết vì hết RAM (DoS thành công): {state}; kết quả từng yêu cầu: {results}'
    assert 'running' in state, f'container không còn sống sau đợt tấn công: {state}; log: {c.logs()[-1500:]}'

    status, body = kit.get_json(c.url + '/api/status')
    assert status == 200, f'container sống nhưng không trả lời /api/status: {status} {body}'

    # mọi yêu cầu phải có một kết thúc rõ ràng (400 nội dung sai cột, hoặc 429 bị từ chối do bận) --
    # không có yêu cầu nào được "chấp nhận xử lý" (200) với file nhiều giờ như thế này.
    codes = [r[1] for r in results]
    assert all(s in (400, 429) for s in codes), f'có mã trạng thái không mong đợi: {results}'
