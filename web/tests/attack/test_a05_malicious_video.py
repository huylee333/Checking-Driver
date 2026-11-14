"""A05. Video độc hại: file cắt cụt (truncated).

(Đã bỏ phần "video bomb" tự ghép byte định dạng nhị phân tay khỏi bộ test này -- xem nhật ký làm việc:
dùng `cv2.VideoWriter` tạo video THẬT rồi cắt bớt phần cuối, không tự ghép định dạng container/codec.)

Kỳ vọng: video cắt cụt ở mọi mức độ phải bị `video_pipeline.probe_video` phát hiện và từ chối bằng 400,
không treo, không sập, không để lại thư mục tạm trên đĩa máy chủ.
"""
import os
import tempfile

import attack_kit as kit

VALID_TEXT = kit.sensor_text(kit.trip_with(lambda t, rng: rng.normal(0, 0.05, (len(t), 3))))


def _video_bytes(seconds=8.0, fps=10, size=(64, 64)):
    fd, path = tempfile.mkstemp(suffix='.mp4')
    os.close(fd)
    try:
        kit.make_real_video(path, seconds=seconds, fps=fps, size=size)
        return open(path, 'rb').read()
    finally:
        os.remove(path)


FULL_VIDEO = _video_bytes()


def _truncate(data, keep_fraction):
    return data[:max(1, int(len(data) * keep_fraction))]


def test_truncated_video_rejected_cleanly(ready_container):
    c = ready_container
    for keep in (0.9, 0.5, 0.2, 0.05, 0.01):
        truncated = _truncate(FULL_VIDEO, keep)
        status, body = kit.post_multipart(c.url + '/api/analyze', {
            'acc_file': ('RAW_ACCELEROMETERS.txt', VALID_TEXT),
            'video_file': ('trip.mp4', truncated),
        }, timeout=60)
        assert status == 400, f'keep={keep}: mong 400 (video hỏng), được {status}: {body[:200]}'
        text = body.decode('utf-8', 'ignore')
        assert 'Video' in text or 'video' in text, f'keep={keep}: lỗi không nói về video: {text}'
        status2, status_body = kit.get_json(c.url + '/api/status')
        assert status2 == 200 and status_body.get('ready'), f'keep={keep}: máy chủ không còn "ready" sau khi thử: {status_body}'


def test_empty_video_file_rejected(ready_container):
    c = ready_container
    status, body = kit.post_multipart(c.url + '/api/analyze', {
        'acc_file': ('RAW_ACCELEROMETERS.txt', VALID_TEXT),
        'video_file': ('trip.mp4', b''),
    })
    assert status == 400
    assert 'trống' in body.decode('utf-8', 'ignore')


def test_video_shorter_than_minimum_trip_is_rejected(ready_container):
    """Video thật, mở được, nhưng chỉ dài 3 giây (< MIN_TRIP_SEC = 60s)."""
    c = ready_container
    short = _video_bytes(seconds=3.0)
    status, body = kit.post_multipart(c.url + '/api/analyze', {
        'acc_file': ('RAW_ACCELEROMETERS.txt', VALID_TEXT),
        'video_file': ('trip.mp4', short),
    })
    assert status == 400
    assert 'ngắn' in body.decode('utf-8', 'ignore')


def test_no_leftover_temp_directory_after_rejected_video(ready_container):
    """Mỗi video bị từ chối phải xoá luôn thư mục tạm (`tempfile.mkdtemp(prefix='trip_')`), không để
    tích tụ trên đĩa máy chủ qua nhiều lần tấn công."""
    c = ready_container
    before = c.exec('sh', '-c', 'ls /tmp 2>/dev/null | grep -c "^trip_" || true').strip()
    for _ in range(5):
        kit.post_multipart(c.url + '/api/analyze', {
            'acc_file': ('RAW_ACCELEROMETERS.txt', VALID_TEXT),
            'video_file': ('trip.mp4', _truncate(FULL_VIDEO, 0.05)),
        })
    after = c.exec('sh', '-c', 'ls /tmp 2>/dev/null | grep -c "^trip_" || true').strip()
    assert after == before, f'còn {after} thư mục trip_* trên đĩa (trước: {before}) -- nghi rò thư mục tạm'
