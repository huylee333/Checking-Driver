"""A04. Path traversal và tiêm tên file (Web/API).

Đọc `web/app.py` trước khi viết test này: tên file người dùng gửi lên CHỈ được dùng để lấy ĐUÔI file
(`os.path.splitext(f.filename)[1]`) -- không có chỗ nào ghép tên file gốc vào đường dẫn trên đĩa (file
video luôn được lưu với tên cố định `'video' + ext` trong một thư mục tạm do máy chủ tự sinh bằng
`tempfile.mkdtemp`). Máy chủ cũng không có `app.mount(StaticFiles...)` nào, nên không có "thư mục tĩnh"
để traversal tới. `GET /api/job/{jid}` tra `jid` trong một dict Python, không chạm tới hệ thống file.

Test dưới đây xác nhận đúng những gì đọc code cho thấy: không có đường vào, chứ không chỉ "không bắt lỗi
được request". Nếu một bản sửa sau này vô tình dùng `f.filename` để ghép đường dẫn, các test đọc nội
dung trả về (so khớp ngược với nội dung bí mật mong có) sẽ bắt được.
"""
import os

import attack_kit as kit

VALID_TEXT = kit.sensor_text(kit.trip_with(lambda t, rng: rng.normal(0, 0.05, (len(t), 3))))

TRAVERSAL_NAMES = [
    '../../../../etc/passwd.txt',
    r'..\..\..\Windows\win.ini.txt',
    'a\x00b.txt',
    '....//....//....//etc/passwd.txt',
    '/etc/passwd.txt',
    'C:\\Windows\\System32\\drivers\\etc\\hosts.txt',
    '%2e%2e%2fetc%2fpasswd.txt',          # percent-encoded, không được giải mã thêm lần nào
]


def test_traversal_filenames_never_reach_the_filesystem(ready_container):
    """Mọi tên file traversal vẫn phải được xử lý bằng ĐÚNG luật của đuôi `.txt` (nội dung không hợp lệ
    về cột/độ dài -> 400), không có dấu hiệu nào cho thấy máy chủ đọc/viết nhầm một đường dẫn khác."""
    c = ready_container
    for name in TRAVERSAL_NAMES:
        status, body = kit.post_multipart(c.url + '/api/analyze', {'acc_file': (name, b'khong phai so' * 50)})
        assert status == 400, f'tên "{name}": mong 400 (nội dung không phải bảng số), được {status}: {body[:150]}'
        text = body.decode('utf-8', 'ignore')
        # Không bao giờ được thấy nội dung một file hệ thống thật rò ra trong phản hồi.
        assert 'root:' not in text and '[fonts]' not in text, f'tên "{name}": phản hồi có vẻ chứa nội dung file hệ thống: {text[:200]}'


def test_traversal_filename_with_otherwise_valid_content_is_accepted_like_any_other_name(ready_container):
    """Nếu tên file traversal nhưng NỘI DUNG hợp lệ: máy chủ xử lý bình thường như mọi tên khác (vì tên
    file không ảnh hưởng gì tới việc đọc/ghi), job vẫn chạy và trả kết quả -- đúng là an toàn, vì tên file
    chưa bao giờ được dùng làm đường dẫn, không phải vì hệ thống "chặn" được tên đó."""
    c = ready_container
    status, body = kit.post_multipart(c.url + '/api/analyze', {'acc_file': ('../../../../etc/passwd.txt', VALID_TEXT)})
    assert status == 200, f'mong 200 (nội dung hợp lệ, tên file không liên quan): {status} {body[:200]}'


def test_job_id_is_looked_up_not_used_as_a_path(ready_container):
    """`jid` traversal (`../`, byte rỗng, rất dài) qua `/api/job/{jid}` phải cho 404 "không tìm thấy",
    không phải 500, không phải đọc được file nào khác."""
    c = ready_container
    import urllib.parse
    for jid in ('../../etc/passwd', '..%2f..%2fapp.py', 'a' * 5000, '%00', 'конец'):
        status, body = kit.get_json(c.url + '/api/job/' + urllib.parse.quote(jid, safe=''))
        assert status in (404, 400), f'jid={jid!r}: mong 404 (không tìm thấy), được {status}: {body}'


def test_no_static_files_route_exists(ready_container):
    """Xác nhận lại bằng HTTP thật (không chỉ đọc code): không có route phục vụ file tĩnh theo đường dẫn
    tùy ý (ví dụ /static/..), vì `app.py` không có `app.mount(StaticFiles...)` nào."""
    c = ready_container
    for path in ('static/../app.py', 'static/app.py', '../requirements-docker.txt', 'weights/final_1d.pth'):
        status, body = kit.get_json(c.url + '/' + path)
        assert status in (404, 405), f'{path}: mong 404/405, được {status} (có thể đã lộ file tĩnh): {str(body)[:200]}'
