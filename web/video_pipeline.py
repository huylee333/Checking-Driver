"""
Nhánh video cho web: đồng bộ video <-> cảm biến, lấy 77 frame cho mỗi đoạn 5 giây,
chạy DINOv3 (backbone) để lấy CLS token từng frame, tạo ảnh thu nhỏ để hiển thị.

Các bước giống hệt lúc tạo dữ liệu train (dust_dinov3/prepare_uah_data.py):
  - đồng bộ: tương quan optical flow ngang của video với yaw rate (dự phòng: hướng GPS),
    chỉ chấp nhận offset có đỉnh tương quan rõ và khớp giữa quét gần và quét rộng;
  - frame: 77 thời điểm trải đều trên khoảng thời gian thật của 51 mẫu gia tốc, lấy frame
    gần nhất (lệch <= nửa chu kỳ frame), cắt vuông giữa, 224x224, RGB, chuẩn hoá ImageNet.
"""

import base64
import datetime as dt
import os
import re

import cv2
import numpy as np
import torch

SYNC_FS = 5.0
NUM_FRAMES = 77
SIZE = 224
THUMB_W, THUMB_H = 288, 162
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TS_RE = re.compile(r'(\d{14})')


# ----------------------------------------------------------------------------
# Thông tin video
# ----------------------------------------------------------------------------
def probe_video(path):
    """Mở thử video. Trả về (fps, số frame, thời lượng giây) hoặc ném ValueError."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError('Không mở được video. File có thể hỏng hoặc không phải video.')
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    ok, _ = cap.read()
    cap.release()
    if not ok or fps <= 0 or n <= 0:
        raise ValueError('Video không đọc được frame nào.')
    return float(fps), int(n), float(n / fps)


def name_offset(trip_name, video_name):
    """Offset theo tên file kiểu UAH (route_start - video_start, giây). None nếu không suy ra được."""
    a = TS_RE.search(trip_name or '')
    b = TS_RE.search(video_name or '')
    if not (a and b):
        return None
    try:
        ta = dt.datetime.strptime(a.group(1), '%Y%m%d%H%M%S')
        tb = dt.datetime.strptime(b.group(1), '%Y%m%d%H%M%S')
    except ValueError:
        return None
    return (ta - tb).total_seconds()


# ----------------------------------------------------------------------------
# Đồng bộ
# ----------------------------------------------------------------------------
def scan_video(path, progress=None):
    """1 lượt giải mã: timestamp mọi frame + tín hiệu optical flow ngang ở SYNC_FS Hz."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    frame_times, flow_t, flow_x = [], [], []
    prev, next_t = None, 0.0
    while True:
        if not cap.grab():
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        frame_times.append(t)
        if progress and len(frame_times) % 3000 == 0:
            progress(f'Đang đọc video để đồng bộ: {100 * len(frame_times) // total}%')
        if t + 1e-6 < next_t:
            continue
        ok, f = cap.retrieve()
        if not ok:
            break
        next_t = t + 1.0 / SYNC_FS
        g = cv2.cvtColor(cv2.resize(f, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        g = g[54:100, 16:256]  # dải chân trời, bỏ A-pillar bên phải
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_t.append(t)
            flow_x.append(float(np.median(flow[..., 0])))
        prev = g
    cap.release()
    return fps, np.asarray(frame_times), np.asarray(flow_t), np.asarray(flow_x)


def _band(x, short=1.0, long=30.0):
    def ma(y, sec):
        n = max(1, int(round(sec * SYNC_FS)))
        return np.convolve(y, np.ones(n) / n, mode='same')
    y = ma(x, short)
    return y - ma(y, long)


def _reference_signals(acc, gps):
    refs = [('yaw_rate', acc[:, 0], np.gradient(np.unwrap(acc[:, 10]), acc[:, 0]))]
    if gps is not None and len(gps) > 30:
        tg = gps[:, 0]
        rate = -np.gradient(np.unwrap(np.deg2rad(gps[:, 7])), tg)
        rate[gps[:, 1] < 10] = 0.0
        refs.append(('gps_course_rate', tg, rate))
    return refs


def _corr_scan(flow_t, flow_x, ref_t, ref_x, lags, min_overlap):
    g0 = np.ceil(ref_t[0])
    grid = np.arange(g0, ref_t[-1], 1 / SYNC_FS)
    s = _band(np.interp(grid, ref_t, ref_x))
    vgrid = np.arange(0, flow_t[-1], 1 / SYNC_FS)
    v = _band(np.interp(vgrid, flow_t, np.clip(flow_x, -10, 10)))
    need = max(min_overlap * len(grid), 60 * SYNC_FS)
    out = np.full(len(lags), np.nan)
    for i, L in enumerate(lags):
        k = int(round((g0 + L) * SYNC_FS))
        i0, i1 = max(0, -k), min(len(s), len(v) - k)
        if i1 - i0 < need:
            continue
        a = v[i0 + k:i1 + k] - v[i0 + k:i1 + k].mean()
        b = s[i0:i1] - s[i0:i1].mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        if den > 0:
            out[i] = (a * b).sum() / den
    return out


def _peak(lags, corr, exclude_sec=10.0):
    c = np.nan_to_num(corr, nan=-np.inf)
    i = int(np.argmax(c))
    far = np.abs(lags - lags[i]) > exclude_sec
    second = float(np.nanmax(np.where(far, np.nan_to_num(np.abs(corr), nan=0), np.nan))) if far.any() else 0.0
    return float(lags[i]), float(corr[i]), second


def verify_offset(base_offset, flow_t, flow_x, acc, gps, search=10.0, min_corr=0.15, min_ratio=1.5):
    """Kiểm chứng offset. base_offset=None (không có tên file kiểu UAH) -> lấy đỉnh của quét rộng
    làm offset gốc, sau đó vẫn phải qua kiểm tra quét gần như bình thường."""
    if len(flow_t) < 60 * SYNC_FS:
        return dict(verified=False, reason='video quá ngắn để đồng bộ')
    step = 0.2
    wide = np.arange(-flow_t[-1], flow_t[-1] + 1e-9, step)
    res = dict(verified=False, name_offset=base_offset)
    for ref, rt, rx in _reference_signals(acc, gps):
        lag_w, corr_w, second_w = _peak(wide, _corr_scan(flow_t, flow_x, rt, rx, wide, 0.5))
        base = lag_w if base_offset is None else base_offset
        near = np.arange(base - search, base + search + 1e-9, step)
        lag_n, corr_n, second_n = _peak(near, _corr_scan(flow_t, flow_x, rt, rx, near, 0.0))
        ratio = corr_n / max(second_n, 1e-6)
        ok = corr_n >= min_corr and ratio >= min_ratio and abs(lag_w - lag_n) <= 2.0
        res.update(ref=ref, offset=lag_n, corr=corr_n, ratio=ratio, wide_lag=lag_w, verified=bool(ok))
        if ok:
            break
    if not res['verified']:
        res['reason'] = 'không tìm được độ lệch thời gian đáng tin giữa video và cảm biến'
    return res


# ----------------------------------------------------------------------------
# Lấy frame
# ----------------------------------------------------------------------------
def plan_frames(windows, acc, offset, frame_times, fps, kine_len=51):
    """Với mỗi cửa sổ hợp lệ, chọn chỉ số 77 frame thật gần nhất. Trả về list (window_index, idx)."""
    half = 0.5 / fps + 0.005
    plans = []
    for wi, w in enumerate(windows):
        if not w['valid']:
            continue
        ts = acc[w['i0']:w['i0'] + kine_len, 0]
        targets = np.linspace(ts[0], ts[-1], NUM_FRAMES) + offset
        if targets[0] < frame_times[0] or targets[-1] > frame_times[-1]:
            continue
        idx = np.clip(np.searchsorted(frame_times, targets), 1, len(frame_times) - 1)
        left = (targets - frame_times[idx - 1]) <= (frame_times[idx] - targets)
        idx = np.where(left, idx - 1, idx)
        if np.abs(frame_times[idx] - targets).max() > half:
            continue
        lo, hi = idx.min(), idx.max()
        if np.diff(frame_times[lo:hi + 1]).max() > 3.0 / fps:
            continue
        plans.append((wi, idx))
    return plans


def _prep(f):
    h, w = f.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    f = cv2.resize(f[y0:y0 + s, x0:x0 + s], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)


def _thumb(f):
    t = cv2.resize(f, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', t, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf.tobytes()).decode('ascii') if ok else None


def extract_frames(path, plans, frames_path, progress=None):
    """Giải mã tuần tự, ghi frame 224x224 uint8 của mọi cửa sổ vào memmap (N, 77, 224, 224, 3)
    và tạo ảnh thu nhỏ (frame giữa mỗi cửa sổ). Trả về (danh sách cửa sổ đủ frame, thumbs)."""
    n = len(plans)
    mm = np.lib.format.open_memmap(frames_path, mode='w+', dtype=np.uint8, shape=(n, NUM_FRAMES, SIZE, SIZE, 3))
    need = {}
    for pi, (_, idx) in enumerate(plans):
        for k, fi in enumerate(idx):
            need.setdefault(int(fi), []).append((pi, k))
    filled = np.zeros((n, NUM_FRAMES), dtype=bool)
    thumbs = [None] * n
    mid = NUM_FRAMES // 2
    order = sorted(need)
    cap = cv2.VideoCapture(path)
    fi, j = -1, 0
    while j < len(order):
        if not cap.grab():
            break
        fi += 1
        if fi != order[j]:
            continue
        ok, f = cap.retrieve()
        if ok:
            img = _prep(f)
            for pi, k in need[fi]:
                mm[pi, k] = img
                filled[pi, k] = True
                if k == mid:
                    thumbs[pi] = _thumb(f)
        j += 1
        if progress and j % 2000 == 0:
            progress(f'Đang lấy frame video: {100 * j // len(order)}%')
    cap.release()
    mm.flush()
    del mm
    complete = [bool(filled[pi].all()) for pi in range(n)]
    return complete, thumbs


@torch.no_grad()
def backbone_cls(encoder, frames_path, rows, device='cpu', batch_frames=256, progress=None):
    """CLS token DINOv3 cho các cửa sổ `rows` trong memmap frame. Trả về float16 (len(rows), 77, 384)."""
    mm = np.load(frames_path, mmap_mode='r')
    enc = encoder.to(device).eval()
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    out = np.zeros((len(rows), NUM_FRAMES, enc.hidden_size), dtype=np.float16)
    use_amp = device != 'cpu'
    for r, row in enumerate(rows):
        x = torch.from_numpy(np.ascontiguousarray(mm[row])).to(device)          # (77, 224, 224, 3) uint8
        x = x.permute(0, 3, 1, 2).float().div_(255.0).sub_(mean).div_(std)       # (77, 3, 224, 224)
        feats = []
        for i in range(0, NUM_FRAMES, batch_frames):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                feats.append(enc._backbone_cls(x[i:i + batch_frames]).float())
        out[r] = torch.cat(feats).cpu().numpy().astype(np.float16)
        if progress and (r + 1) % 10 == 0:
            progress(f'Đang chạy DINOv3: {r + 1}/{len(rows)} đoạn')
    return out
