"""A06. Lộ thông tin (lỗi chi tiết, token, cấu hình).

Bốn hướng: (1) lỗi có kèm stack trace/đường dẫn máy chủ không; (2) /docs, /openapi.json có công khai
ngoài ý muốn không; (3) HF_TOKEN giả có lộ ra phản hồi/log không; (4) header HTTP có lộ gì không.
"""
import re

import attack_kit as kit

VALID_TEXT = kit.sensor_text(kit.trip_with(lambda t, rng: rng.normal(0, 0.05, (len(t), 3))))
LEAK_PATTERNS = [
    r'Traceback \(most recent call last\)',
    r'File "/home/user/app/\w+\.py", line \d+',
    r'/home/user/app',           # đường dẫn tuyệt đối trên máy chủ
    r'site-packages',
]


def _check_no_leak(text, where):
    for pat in LEAK_PATTERNS:
        assert not re.search(pat, text), f'{where}: phản hồi có dấu hiệu lộ thông tin nội bộ ({pat}): {text[:300]}'


def test_weird_inputs_never_return_a_raw_traceback(ready_container):
    """Một loạt đầu vào kỳ quặc (không hợp lệ nhưng không nhắm vào một lỗi cụ thể): mọi phản hồi phải là
    lỗi gọn (400/413/422/404), không bao giờ là 500 kèm traceback."""
    c = ready_container
    error_cases = [
        ({'acc_file': ('a.txt', b'')}, {}),
        ({'acc_file': ('a.txt', b' \n\n\n ')}, {}),
        ({'acc_file': ('a.txt', ('1 2 3\n' * 60).encode())}, {}),       # sai số cột
        ({'acc_file': ('a.txt', VALID_TEXT), 'gps_file': ('g.txt', b'not a gps file' * 20)}, {}),
    ]
    for files, data in error_cases:
        status, body = kit.post_multipart(c.url + '/api/analyze', files, data=data)
        assert status in (400, 413, 422), f'{files.keys()}: mong lỗi gọn, được {status}: {body[:200]}'
        _check_no_leak(body.decode('utf-8', 'ignore'), f'/api/analyze {list(files)}')

    # trip_name rất dài không phải lỗi (không có giới hạn độ dài) -- chỉ cần không sập, không lộ gì.
    status, body = kit.post_multipart(c.url + '/api/analyze', {'acc_file': ('a.txt', VALID_TEXT)}, data={'trip_name': 'x' * 100000})
    assert status in (200, 400, 413, 422), f'trip_name dài: {status}: {body[:200]}'
    _check_no_leak(body.decode('utf-8', 'ignore'), '/api/analyze trip_name dài')
    if status == 200:
        import json
        import time
        jid = json.loads(body)['job_id']
        deadline = time.time() + 30
        while time.time() < deadline:
            s, b = kit.get_json(c.url + '/api/job/' + jid)
            if s == 200 and b.get('status') in ('done', 'error'):
                break
            time.sleep(0.5)

    for jid in ('not-a-real-id', '', '0' * 64):
        status, body = kit.get_json(c.url + '/api/job/' + jid)
        assert status == 404
        _check_no_leak(str(body), f'/api/job/{jid}')


def test_docs_endpoints_are_a_known_accepted_exposure(ready_container):
    """FastAPI mặc định công khai /docs, /redoc, /openapi.json. Đây là hành vi CHỦ ĐỊNH của framework,
    không phải lỗi -- test này chỉ xác nhận lại hiện trạng để không ai "phát hiện lại" nó ngoài ý muốn,
    và để checkpoint có bằng chứng thay vì chỉ suy đoán."""
    c = ready_container
    for path in ('docs', 'openapi.json', 'redoc'):
        status, _ = kit.get_raw(c.url + '/' + path)
        assert status == 200, f'/{path}: {status} (nếu đổi thành 404, nhớ sửa lại ghi chú trong checkpoint)'


def test_response_headers_do_not_leak_secrets_or_internal_paths(ready_container):
    c = ready_container
    status, body = kit.post_multipart(c.url + '/api/analyze', {'acc_file': ('a.txt', b'sai cot' * 20)})
    import urllib.request
    req = urllib.request.Request(c.url + '/api/status')
    with urllib.request.urlopen(req, timeout=10) as r:
        headers = dict(r.getheaders())
    header_text = ' '.join(f'{k}: {v}' for k, v in headers.items())
    _check_no_leak(header_text, 'response headers')
    assert 'HF_TOKEN' not in header_text and 'hf_' not in header_text.lower()


def test_fake_hf_token_never_appears_in_any_response_or_log():
    """Chạy một container RIÊNG với HF_TOKEN GIẢ (không phải token thật), rồi dò mọi phản hồi và log
    container xem có lộ chuỗi token ra ngoài không."""
    import os
    import subprocess
    import uuid

    import pytest
    from docker_helpers import IMAGE
    if not IMAGE:
        pytest.skip('set CD_IMAGE to the image name to run the attack tests')

    os.environ['PATH'] += r';C:\Program Files\Docker\Docker\resources\bin'
    fake_token = 'hf_FAKE' + uuid.uuid4().hex
    name = 'cdatk-tok-' + uuid.uuid4().hex[:8]
    try:
        subprocess.run(['docker', 'run', '-d', '--name', name, '-e', f'HF_TOKEN={fake_token}',
                        '-p', '127.0.0.1::7860', IMAGE], check=True, capture_output=True, text=True)
        from docker_helpers import Container
        c = Container(name)
        c.wait_web(120)

        status, status_body = kit.get_json(c.url + '/api/status')
        assert fake_token not in str(status_body)

        err_status, err_body = kit.post_multipart(c.url + '/api/analyze', {'acc_file': ('a.txt', b'x' * 10)})
        assert fake_token not in err_body.decode('utf-8', 'ignore')

        logs = c.logs()
        assert fake_token not in logs, f'token giả xuất hiện trong log container: ...{logs[max(0, logs.find(fake_token)-80):][:200]}'
    finally:
        subprocess.run(['docker', 'rm', '-f', name], capture_output=True, text=True)
