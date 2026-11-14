"""Tools for the attack tests. Everything here builds HARMLESS-LOOKING input files on the local machine
(synthetic data only) so that the attack tests can throw them at our own app.

Never point these at a server you do not own.
"""
import json
import mimetypes
import urllib.error
import urllib.request
import uuid

import numpy as np


# ----------------------------------------------------------------------------------------------------------------
# HTTP: multipart POST built with the standard library only (no `requests` -- it is not in requirements-dev.txt,
# and these tests are collected by `pytest web/tests -q` even when skipped, so an import here must never fail
# on a machine that only has the dev requirements. Same lesson as T06, see CHECKPOINT_TEST_VA_TAN_CONG.md).
# ----------------------------------------------------------------------------------------------------------------
def post_multipart(url, files, data=None, timeout=60, headers=None):
    """files: {field: (filename, bytes, content_type)}. data: {field: str}. Returns (status, body_bytes).
    Never raises on an HTTP error status (returns that status and the error body instead)."""
    boundary = uuid.uuid4().hex
    body = bytearray()

    def add(header, payload):
        body.extend(f'--{boundary}\r\n'.encode())
        body.extend(header.encode())
        body.extend(payload if isinstance(payload, bytes) else payload.encode())
        body.extend(b'\r\n')

    for name, value in (data or {}).items():
        add(f'Content-Disposition: form-data; name="{name}"\r\n\r\n', value)
    for field, spec in files.items():
        filename, content, ctype = (*spec, None)[:3] if len(spec) == 2 else spec
        ctype = ctype or mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        header = (f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
                 f'Content-Type: {ctype}\r\n\r\n')
        add(header, content)
    body.extend(f'--{boundary}--\r\n'.encode())
    req = urllib.request.Request(url, data=bytes(body), method='POST', headers={
        'Content-Type': f'multipart/form-data; boundary={boundary}', **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def post_json_path(url, path='api/analyze', timeout=10):
    """POST with no body at all (just checks the server answers something, never a stack trace)."""
    req = urllib.request.Request(f'{url}/{path}', data=b'', method='POST')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def get_raw(url, timeout=10):
    """(status, body bytes), không ép JSON -- dùng cho route trả HTML (/docs, /redoc)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def get_json(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}


# ----------------------------------------------------------------------------------------------------------------
# Video cắt cụt: dùng cv2.VideoWriter (thư viện chuẩn) tạo video thật rồi cắt bớt phần cuối file.
# ----------------------------------------------------------------------------------------------------------------
def make_real_video(path, seconds=8.0, fps=10, size=(64, 64)):
    """Ghi một video .mp4 thật (khung hình xám đơn giản, không có nội dung gì đặc biệt)."""
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, size)
    n = int(seconds * fps)
    for i in range(n):
        frame = np.full((size[1], size[0], 3), (i * 3) % 256, dtype=np.uint8)
        out.write(frame)
    out.release()


def truncate_file(path, keep_fraction):
    """Cắt file tại `path` còn lại `keep_fraction` phần đầu (0 < keep_fraction <= 1), ghi đè tại chỗ."""
    data = open(path, 'rb').read()
    cut = max(1, int(len(data) * keep_fraction))
    open(path, 'wb').write(data[:cut])


# ----------------------------------------------------------------------------------------------------------------
# Sensor files
# ----------------------------------------------------------------------------------------------------------------
def sensor_text(rows):
    """Text of an accelerometer file from a (n, 11) array."""
    return ('\n'.join(' '.join(f'{x:.6f}' for x in r) for r in rows) + '\n').encode()


def big_text_trip(megabytes=19.5):
    """A syntactically valid accelerometer file of about `megabytes` MB. It is many hours long, so the app rejects
    it, but only AFTER reading and parsing all of it."""
    row = '{t:.6f} 1 0.01 0.02 0.03 0.01 0.02 0.03 0.0 0.0 0.0\n'
    out, size, i = [], 0, 0
    while size < megabytes * 1024 * 1024:
        s = row.format(t=i / 10)
        out.append(s)
        size += len(s)
        i += 1
    return ''.join(out).encode()


def trip_with(fn, seconds=70.0, hz=10.0, seed=0):
    """A 70-second trip whose three filtered acceleration columns (5-7) come from fn(t, rng) -> (n, 3)."""
    n = int(seconds * hz) + 1
    rng = np.random.default_rng(seed)
    a = np.zeros((n, 11))
    a[:, 0] = np.arange(n) / hz
    a[:, 1] = 1
    a[:, 2:5] = rng.normal(0, 0.05, (n, 3))
    a[:, 5:8] = fn(a[:, 0], rng)
    return a
