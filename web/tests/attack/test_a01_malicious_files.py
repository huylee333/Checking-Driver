"""A01. Tệp độc hại giả dạng hợp lệ (type confusion, polyglot).

Attacker đổi đuôi file, tiêu đề Content-Type, hoặc chèn vài byte "mẫu" của một loại file khác, với hi vọng
máy chủ tin vào cái vỏ bên ngoài mà không kiểm tra nội dung thật. Mọi trường hợp dưới đây dùng container thật
(model thật đã nạp, `analyzer.ready == True`) vì endpoint chỉ kiểm tra file SAU khi qua cổng `ready`.

Kỳ vọng: máy chủ luôn trả 400/413 (không bao giờ 200 với dữ liệu vô nghĩa, không bao giờ 500), và vẫn sống,
vẫn "ready" sau mỗi lần thử.
"""
import attack_kit as kit

VALID_ROW = '0.000000 1 0.01 0.02 0.03 0.01 0.02 0.03 0.0 0.0 0.0\n'
VALID_TEXT = (VALID_ROW * 60).encode()  # 60 dòng hợp lệ, không đủ dài để thành 1 chuyến (không quan trọng ở đây)


def assert_alive_and_ready(c):
    status, body = kit.get_json(c.url + '/api/status')
    assert status == 200, f'container không trả lời sau tấn công: {status}'
    assert body.get('ready') is True, f'container mất trạng thái "ready" sau tấn công: {body}'


CASES = [
    # (mô tả, tên file, nội dung, lý do)
    ('đuôi kép .txt.exe', 'RAW_ACCELEROMETERS.txt.exe', VALID_TEXT),
    ('đuôi kép .exe.txt (ẩn trong vỏ .txt)', 'malware.exe.txt', VALID_TEXT),
    ('nội dung nhị phân, đuôi .txt', 'RAW_ACCELEROMETERS.txt', b'\x00\x01\x02\xff\xfe' * 200),
    ('tiền tố magic-byte PDF, đuôi .txt', 'RAW_ACCELEROMETERS.txt', b'%PDF-1.4\n' + VALID_TEXT),
    ('tiền tố magic-byte GIF, đuôi .txt', 'RAW_ACCELEROMETERS.txt', b'GIF89a' + VALID_TEXT),
    ('tiền tố magic-byte MZ (PE), đuôi .txt', 'RAW_ACCELEROMETERS.txt', b'MZ\x90\x00\x03\x00' + VALID_TEXT),
    ('byte NUL sau vị trí quét 8192 (lách kiểm tra đầu file)', 'RAW_ACCELEROMETERS.txt',
     VALID_TEXT * 200 + b'\x00' + VALID_TEXT),
    ('không có đuôi file', 'RAW_ACCELEROMETERS', VALID_TEXT),
    ('tên file rất dài (2000 ký tự)', 'a' * 2000 + '.txt', VALID_TEXT),
]


def test_malicious_text_files_rejected_safely(ready_container):
    c = ready_container
    for desc, name, content in CASES:
        status, body = kit.post_multipart(c.url + '/api/analyze', {'acc_file': (name, content, 'text/plain')})
        assert status in (400, 413), f'[{desc}] mong 400/413, được {status}: {body[:200]}'
        assert_alive_and_ready(c)


def test_text_claiming_to_be_video_is_rejected_before_reading_content(ready_container):
    """Đổi đuôi .txt thành đuôi video: phải bị chặn ngay vì "đây là file video", không parse nội dung."""
    c = ready_container
    status, body = kit.post_multipart(c.url + '/api/analyze', {
        'acc_file': ('RAW_ACCELEROMETERS.mp4', VALID_TEXT, 'text/plain')})
    assert status == 400
    assert b'video' in body.lower() or 'video'.encode() in body
    assert_alive_and_ready(c)


def test_content_type_header_is_not_trusted(ready_container):
    """Content-Type nói "application/octet-stream" nhưng đuôi file .txt hợp lệ: máy chủ vẫn xử lý theo ĐUÔI
    FILE (không theo header, header do client tự ý đặt, không đáng tin), nên vẫn phải báo lỗi nội dung (vì nội
    dung ở đây là text hợp lệ về CỘT nhưng không đủ dài để thành một chuyến), KHÔNG được 500."""
    c = ready_container
    status, body = kit.post_multipart(
        c.url + '/api/analyze',
        {'acc_file': ('RAW_ACCELEROMETERS.txt', VALID_TEXT, 'application/octet-stream')})
    assert status == 400, f'mong 400 (chuyến quá ngắn), được {status}: {body[:200]}'
    assert_alive_and_ready(c)


def test_video_field_with_text_content_fails_to_open_cleanly(ready_container):
    """File có đuôi .mp4 (qua được bước kiểm đuôi) nhưng nội dung thực chỉ là văn bản: OpenCV phải báo lỗi mở
    video, máy chủ trả 400 kèm lý do, không treo, không sập."""
    c = ready_container
    status, body = kit.post_multipart(c.url + '/api/analyze', {
        'acc_file': ('RAW_ACCELEROMETERS.txt', kit.sensor_text(kit.trip_with(lambda t, r: r.normal(0, 0.05, (len(t), 3))))),
        'video_file': ('trip.mp4', b'day khong phai video, chi la text gia lam .mp4' * 50, 'video/mp4'),
    })
    assert status == 400, f'mong 400, được {status}: {body[:200]}'
    assert 'video' in body.decode('utf-8', 'ignore').lower() or 'Video' in body.decode('utf-8', 'ignore')
    assert_alive_and_ready(c)
