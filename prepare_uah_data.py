"""
Làm sạch + chuyển UAH-DriveSet v1 sang định dạng pickle của pipeline DuST-DINOv3.

NGUYÊN TẮC: không sinh dữ liệu. Mọi giá trị đầu ra là mẫu thật:
  - Kinematic: 51 mẫu RAW_ACCELEROMETERS liên tiếp, nguyên giá trị (không nội suy),
    3 kênh gia tốc đã lọc Kalman X, Y, Z (cột 6-8), chỉ chuẩn hoá z-score theo tập train.
  - Video: 77 frame thật, mỗi frame là frame gần nhất với thời điểm đích
    (sai lệch <= nửa chu kỳ frame), thời điểm đích trải đều trên đúng khoảng thời gian
    thật của 51 mẫu kinematic, quy đổi bằng offset đã được KIỂM CHỨNG.
  - Cửa sổ nào không đạt kiểm tra thì BỎ, không sửa, không lấp. Lý do được ghi vào report.

Các bước:
  1. Kiểm tra cấu trúc chuyến: tên thư mục, đúng 1 file .mp4, video mở được, đủ file cảm biến.
  2. Làm sạch cảm biến (README_DATA.md):
       - Timestamp quay ngược (D6-DROWSY-MOTORWAY): chỉ giữ phần trước lần quay ngược đầu tiên.
       - Các dòng trùng timestamp (mẫu bị dồn buffer, không biết thời điểm thật): bỏ cả nhóm.
       - Dòng có cờ "hệ thống kích hoạt" = 0: cửa sổ chứa dòng đó bị loại.
       - File *.txt~ (backup editor) không bao giờ được đọc.
  3. Kiểm chứng đồng bộ video <-> cảm biến bằng dữ liệu, không chỉ tin tên file:
       tín hiệu A = median optical flow ngang ở dải chân trời của video (xoay xe -> cảnh trôi ngang),
       tín hiệu B = yaw rate từ cột yaw (RAD) của RAW_ACCELEROMETERS.
       Tương quan chéo tìm offset có tương quan cao nhất. Chuyến chỉ được dùng khi
       offset ước lượng rõ ràng (đỉnh nổi bật) VÀ khớp offset theo tên file trong ngưỡng.
       Chuyến không khớp bị loại, trừ khi người dùng ghi đè có chủ đích bằng --offset_override.
  4. Cắt cửa sổ, kiểm tra từng cửa sổ (khoảng trống mẫu, cờ kích hoạt, nằm trọn trong video,
     frame có tồn tại, frame đen, video đứng hình khi xe đang chạy).
  5. Chia train/val THEO TÀI XẾ, ghi pickle + file CSV mô tả từng mẫu (cùng thứ tự với pickle)
     + report từng chuyến.

Ví dụ:
  # Chỉ kiểm tra, không ghi pickle (nên chạy trước và đọc report):
  python prepare_uah_data.py --uah_root ../UAH-DRIVESET-v1 --out_dir ../data --check_only
  # Tạo dữ liệu:
  python prepare_uah_data.py --uah_root ../UAH-DRIVESET-v1 --out_dir ../data --max_windows_per_trip 8
"""

import argparse
import csv
import datetime as dt
import glob
import json
import os
import pickle
import re
import sys
import time

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(errors='backslashreplace')
except AttributeError:
    pass

LABELS = {'NORMAL': 0, 'AGGRESSIVE': 1, 'DROWSY': 2}
TRIP_RE = re.compile(r'^(\d{14})-(\d+)km-(D\d)-(NORMAL|AGGRESSIVE|DROWSY)(\d?)-(MOTORWAY|SECONDARY)$')
REQUIRED_FILES = ['RAW_ACCELEROMETERS.txt', 'RAW_GPS.txt']
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SYNC_FS = 5.0  # Hz, tần số tín hiệu dùng để kiểm chứng đồng bộ


# ----------------------------------------------------------------------------
# Tiện ích
# ----------------------------------------------------------------------------
def parse_ts(s):
    return dt.datetime.strptime(s[:14], '%Y%m%d%H%M%S')


def find_uah_root(path):
    """Chấp nhận thư mục chứa D1..D6 hoặc thư mục cha (UAH-DRIVESET-v1/UAH-DRIVESET-v1)."""
    cands = [path, os.path.join(path, 'UAH-DRIVESET-v1')]
    for c in cands:
        if os.path.isdir(os.path.join(c, 'D1')):
            return c
    raise FileNotFoundError(f'Không tìm thấy thư mục D1..D6 trong {path}')


def load_table(path):
    """Đọc file .txt cách nhau bởi khoảng trắng, bỏ dòng trống. Trả về ndarray float (n, k)."""
    rows = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.split()
            if parts:
                rows.append([float(x) for x in parts])
    ncol = {len(r) for r in rows}
    if len(ncol) != 1:
        raise ValueError(f'{path}: số cột không đồng nhất {sorted(ncol)}')
    return np.asarray(rows, dtype=np.float64)


def clean_timeseries(a):
    """Làm sạch theo timestamp (cột 0). Trả về (mảng sạch, thống kê)."""
    stats = {'rows_raw': int(len(a)), 'dropped_after_backjump': 0, 'dropped_duplicate_ts': 0,
             'ambiguous_from': None, 'ambiguous_to': None}
    t = a[:, 0]
    back = np.where(np.diff(t) < 0)[0]
    if len(back):
        cut = back[0] + 1
        stats['dropped_after_backjump'] = int(len(a) - cut)
        # Đoạn sau lần quay ngược ghi đè lên khoảng [min(t_sau), t_trước_khi_quay] với giá trị khác:
        # không biết bản nào đúng -> cả khoảng này là MƠ HỒ, cửa sổ chạm vào sẽ bị loại.
        stats['ambiguous_from'] = float(t[cut:].min())
        stats['ambiguous_to'] = float(t[cut - 1])
        a = a[:cut]
        t = a[:, 0]
    _, idx, cnt = np.unique(t, return_index=True, return_counts=True)
    keep = np.zeros(len(t), dtype=bool)
    keep[idx[cnt == 1]] = True
    stats['dropped_duplicate_ts'] = int((~keep).sum())
    a = a[keep]
    stats['rows_clean'] = int(len(a))
    return a, stats


# ----------------------------------------------------------------------------
# Kiểm chứng đồng bộ
# ----------------------------------------------------------------------------
def scan_video(mp4):
    """1 lượt giải mã: timestamp thật của MỌI frame + tín hiệu optical flow ngang ở SYNC_FS Hz."""
    cap = cv2.VideoCapture(mp4)
    if not cap.isOpened():
        raise IOError(f'Không mở được video {mp4}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_times, flow_t, flow_x = [], [], []
    prev, next_t = None, 0.0
    while True:
        if not cap.grab():
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        frame_times.append(t)
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


def reference_signals(acc, gps):
    """Tín hiệu cảm biến tỉ lệ với tốc độ quay của xe (cùng dấu với optical flow ngang của video).
    1) yaw rate từ attitude của điện thoại (10 Hz, chính xác nhất);
    2) dự phòng: -(tốc độ đổi hướng GPS) (1 Hz), dùng khi yaw của điện thoại hỏng
       (vd. D1-DROWSY-MOTORWAY: yaw rate gần như không tương quan với hướng GPS).
       Chỉ tính khi xe chạy > 10 km/h vì hướng GPS lúc đứng yên là nhiễu."""
    refs = [('yaw_rate', acc[:, 0], np.gradient(np.unwrap(acc[:, 10]), acc[:, 0]))]
    if len(gps) > 30:
        tg = gps[:, 0]
        rate = -np.gradient(np.unwrap(np.deg2rad(gps[:, 7])), tg)
        rate[gps[:, 1] < 10] = 0.0
        refs.append(('gps_course_rate', tg, rate))
    return refs


def correlation_scan(flow_t, flow_x, ref_t, ref_x, lags, min_overlap):
    """corr(lag) giữa video(t_data + lag) và tín hiệu tham chiếu cảm biến(t_data)."""
    # Cả hai tín hiệu nằm trên lưới bước 1/SYNC_FS: lưới cảm biến bắt đầu ở giây nguyên g0,
    # lưới video bắt đầu ở 0, nên lag là bội của 1/SYNC_FS tương ứng một phép dịch chỉ số nguyên.
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
        a = v[i0 + k:i1 + k]
        b = s[i0:i1]
        a = a - a.mean()
        b = b - b.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        if den > 0:
            out[i] = (a * b).sum() / den
    return out


def peak_info(lags, corr, exclude_sec=10.0):
    c = np.nan_to_num(corr, nan=-np.inf)
    i = int(np.argmax(c))  # chỉ xét tương quan DƯƠNG (dấu đã kiểm chứng trên dữ liệu)
    far = np.abs(lags - lags[i]) > exclude_sec
    second = float(np.nanmax(np.where(far, np.nan_to_num(np.abs(corr), nan=0), np.nan))) if far.any() else 0.0
    return float(lags[i]), float(corr[i]), second


def verify_offset(name_offset, flow_t, flow_x, acc, gps, args):
    """Kiểm chứng offset. Một chuyến được coi là đồng bộ đúng khi, với một tín hiệu tham chiếu:
      (a) quanh offset gốc (±sync_search s) có đỉnh tương quan dương đủ cao và nổi bật, VÀ
      (b) quét TOÀN BỘ lag khả dĩ cũng ra đỉnh tại cùng chỗ (±2 s) -> không phải đỉnh cục bộ ngẫu nhiên.
    Thử yaw rate trước, GPS course rate sau. Trả về dict kết quả."""
    step = 0.2
    near = np.arange(name_offset - args.sync_search, name_offset + args.sync_search + 1e-9, step)
    wide = np.arange(-flow_t[-1], flow_t[-1] + 1e-9, step)
    res = dict(name_offset=name_offset, verified=False)
    for ref_name, rt, rx in reference_signals(acc, gps):
        lag_n, corr_n, second_n = peak_info(near, correlation_scan(flow_t, flow_x, rt, rx, near, 0.0))
        lag_w, corr_w, second_w = peak_info(wide, correlation_scan(flow_t, flow_x, rt, rx, wide, 0.5))
        r = dict(ref=ref_name, near_lag=lag_n, near_corr=corr_n, near_ratio=corr_n / max(second_n, 1e-6),
                 wide_lag=lag_w, wide_corr=corr_w, wide_ratio=corr_w / max(second_w, 1e-6))
        near_ok = corr_n >= args.sync_min_corr and r['near_ratio'] >= args.sync_min_ratio
        r['verified'] = bool(near_ok and abs(lag_w - lag_n) <= 2.0)
        res.update(r)
        if r['verified']:
            break
    return res


# ----------------------------------------------------------------------------
# Cửa sổ
# ----------------------------------------------------------------------------
def candidate_windows(acc, gps, offset, frame_times, fps, args, ambiguous=None):
    """Sinh cửa sổ ứng viên trên lưới thời gian, kiểm tra phía cảm biến + phạm vi video.
    Trả về (list cửa sổ hợp lệ, dict đếm lý do loại)."""
    t = acc[:, 0]
    n = args.kine_len
    reasons = {}
    valid = []
    video_end = frame_times[-1] if len(frame_times) else 0.0
    start = t[0]
    while True:
        i0 = int(np.searchsorted(t, start))
        if i0 + n > len(t):
            break
        seg = acc[i0:i0 + n]
        ts = seg[:, 0]
        nxt = ts[0] + args.stride_sec
        span = ts[-1] - ts[0]
        if args.stride_sec >= span:
            nxt = max(nxt, ts[-1] + 1e-6)  # không chồng lấn: cửa sổ sau không dùng lại mẫu cuối của cửa sổ trước
        reason = None
        if np.diff(ts).max() > args.max_sample_gap:
            reason = 'sensor_gap'
        elif not (args.min_span <= span <= args.max_span):
            reason = 'sensor_span'
        elif (seg[:, 1] != 1).any():
            reason = 'system_inactive'
        elif ambiguous and any(a0 is not None and ts[0] <= a1 and ts[-1] >= a0 for a0, a1 in ambiguous):
            reason = 'ambiguous_time'
        else:
            v0, v1 = ts[0] + offset, ts[-1] + offset
            if v0 < 0 or v1 > video_end:
                reason = 'outside_video'
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
        else:
            gsel = (gps[:, 0] >= ts[0] - 1) & (gps[:, 0] <= ts[-1] + 1)
            speed = float(np.median(gps[gsel, 1])) if gsel.sum() >= 2 else float('nan')
            valid.append(dict(i0=i0, t_start=float(ts[0]), t_end=float(ts[-1]), gps_speed=speed))
        start = nxt
    return valid, reasons


def select_evenly(items, k):
    if k <= 0 or len(items) <= k:
        return list(items)
    idx = np.unique(np.round(np.linspace(0, len(items) - 1, k)).astype(int))
    return [items[i] for i in idx]


def preprocess_frame(f, size, mode):
    if mode == 'center_crop':
        h, w = f.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        f = f[y0:y0 + s, x0:x0 + s]
    f = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)


def extract_windows(mp4, frame_times, fps, windows, offset, acc, args):
    """Lượt giải mã thứ 2: lấy frame cho các cửa sổ đã chọn. Trả về list (window, frames uint8 (T,H,W,3))
    và dict lý do loại."""
    T = args.num_frames
    half = 0.5 / fps + 0.005
    plan = {}  # frame_index -> list of (win_idx, k)
    reasons = {}
    ok_windows = {}
    for wi, w in enumerate(windows):
        ts = acc[w['i0']:w['i0'] + args.kine_len, 0]
        targets = np.linspace(ts[0], ts[-1], T) + offset
        idx = np.clip(np.searchsorted(frame_times, targets), 1, len(frame_times) - 1)
        left_closer = (targets - frame_times[idx - 1]) <= (frame_times[idx] - targets)
        idx = np.where(left_closer, idx - 1, idx)
        err = np.abs(frame_times[idx] - targets)
        if err.max() > half:
            reasons['frame_time_mismatch'] = reasons.get('frame_time_mismatch', 0) + 1
            continue
        lo, hi = idx.min(), idx.max()
        if np.diff(frame_times[lo:hi + 1]).max() > 3.0 / fps:
            reasons['dropped_frames'] = reasons.get('dropped_frames', 0) + 1
            continue
        ok_windows[wi] = dict(win=w, idx=idx, frames=np.zeros((T, args.size, args.size, 3), np.uint8),
                              filled=np.zeros(T, bool), video_t_start=float(targets[0]))
        for k, fi in enumerate(idx):
            plan.setdefault(int(fi), []).append((wi, k))

    if plan:
        cap = cv2.VideoCapture(mp4)
        need = sorted(plan)
        j, fi = 0, -1
        while j < len(need):
            if not cap.grab():
                break
            fi += 1
            if fi != need[j]:
                continue
            t_now = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            ok, f = cap.retrieve()
            if ok and abs(t_now - frame_times[fi]) < 1e-3:  # cùng frame với lượt 1
                img = preprocess_frame(f, args.size, args.resize_mode)
                for wi, k in plan[fi]:
                    ok_windows[wi]['frames'][k] = img
                    ok_windows[wi]['filled'][k] = True
            j += 1
        cap.release()

    out = []
    for wi, o in ok_windows.items():
        if not o['filled'].all():
            reasons['frame_read_failed'] = reasons.get('frame_read_failed', 0) + 1
            continue
        fr = o['frames']
        gray = fr.mean(axis=3)
        if gray.mean(axis=(1, 2)).min() < args.min_brightness:
            reasons['dark_frame'] = reasons.get('dark_frame', 0) + 1
            continue
        diffs = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))
        frozen_frac = float((diffs < 0.5).mean())
        speed = o['win']['gps_speed']
        if frozen_frac > 0.5 and (np.isnan(speed) or speed > 5.0):
            reasons['frozen_video'] = reasons.get('frozen_video', 0) + 1
            continue
        w = dict(o['win'])
        w['video_t_start'] = o['video_t_start']
        w['frame_diff_mean'] = float(diffs.mean())
        out.append((w, fr))
    return out, reasons


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description='Làm sạch UAH-DriveSet và tạo pickle cho DuST-DINOv3')
    p.add_argument('--uah_root', type=str, default='../UAH-DRIVESET-v1')
    p.add_argument('--out_dir', type=str, default='../data')
    p.add_argument('--check_only', action='store_true', help='chỉ kiểm tra + report, không ghi pickle')
    p.add_argument('--val_drivers', type=str, default='D6', help='tài xế dùng cho val, cách nhau bởi dấu phẩy')
    p.add_argument('--exclude_drivers', type=str, default='', help='tài xế bỏ hẳn (vd. để làm test sau này)')
    p.add_argument('--only', action='append', default=[],
                   help='chỉ xử lý chuyến có tên chứa chuỗi này (lặp lại được), dùng để thử nhanh')
    p.add_argument('--num_frames', type=int, default=77)
    p.add_argument('--kine_len', type=int, default=51)
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--resize_mode', choices=['center_crop', 'squash'], default='center_crop')
    p.add_argument('--stride_sec', type=float, default=5.0, help='bước giữa các cửa sổ ứng viên (5.0 = không chồng lấn)')
    p.add_argument('--max_windows_per_trip', type=int, default=8,
                   help='số cửa sổ tối đa mỗi chuyến, chọn trải đều. <=0: lấy hết (rất tốn RAM)')
    p.add_argument('--video_dtype', choices=['float16', 'float32'], default='float16')
    p.add_argument('--no_kine_norm', action='store_true', help='không chuẩn hoá z-score kinematic')
    # Ngưỡng kiểm tra
    p.add_argument('--max_sample_gap', type=float, default=0.5, help='khoảng trống tối đa giữa 2 mẫu acc (s)')
    p.add_argument('--min_span', type=float, default=4.6)
    p.add_argument('--max_span', type=float, default=5.6)
    p.add_argument('--min_brightness', type=float, default=15.0)
    p.add_argument('--sync_search', type=float, default=10.0, help='phạm vi tìm offset quanh offset theo tên file (s)')
    p.add_argument('--sync_min_corr', type=float, default=0.15)
    p.add_argument('--sync_min_ratio', type=float, default=1.5)
    p.add_argument('--offset_override', action='append', default=[],
                   help='TRIP_NAME=OFFSET_GIAY, dùng offset này thay cho offset theo tên file (vẫn phải qua kiểm chứng)')
    args = p.parse_args()

    root = find_uah_root(args.uah_root)
    os.makedirs(args.out_dir, exist_ok=True)
    val_drivers = {d.strip() for d in args.val_drivers.split(',') if d.strip()}
    excl_drivers = {d.strip() for d in args.exclude_drivers.split(',') if d.strip()}
    overrides = {}
    for o in args.offset_override:
        k, v = o.split('=')
        overrides[k.strip()] = float(v)

    trip_dirs = sorted(d for d in glob.glob(os.path.join(root, 'D*', '*')) if os.path.isdir(d))
    if args.only:
        trip_dirs = [d for d in trip_dirs if any(s in os.path.basename(d) for s in args.only)]
    print(f'UAH root: {root}  ({len(trip_dirs)} chuyến)')
    trip_report, selected = [], []

    for d in trip_dirs:
        name = os.path.basename(d)
        rep = dict(trip=name, status='', note='')
        trip_report.append(rep)
        t0 = time.time()
        m = TRIP_RE.match(name)
        if not m:
            rep['status'] = 'EXCLUDED'; rep['note'] = 'tên thư mục sai định dạng'; continue
        _, km, driver, behavior, _, road = m.groups()
        rep.update(driver=driver, behavior=behavior, road=road, label=LABELS[behavior])
        if driver in excl_drivers:
            rep['status'] = 'EXCLUDED'; rep['note'] = 'driver bị loại theo tham số'; continue
        mp4s = glob.glob(os.path.join(d, '*.mp4'))
        missing = [f for f in REQUIRED_FILES if not os.path.isfile(os.path.join(d, f))]
        if len(mp4s) != 1 or missing:
            rep['status'] = 'EXCLUDED'; rep['note'] = f'mp4={len(mp4s)} thiếu={missing}'; continue
        mp4 = mp4s[0]

        acc, acc_stats = clean_timeseries(load_table(os.path.join(d, 'RAW_ACCELEROMETERS.txt')))
        gps, gps_stats = clean_timeseries(load_table(os.path.join(d, 'RAW_GPS.txt')))
        rep.update({f'acc_{k}': v for k, v in acc_stats.items()})
        rep['gps_dropped_after_backjump'] = gps_stats['dropped_after_backjump']

        fps, frame_times, flow_t, flow_x = scan_video(mp4)
        if len(frame_times) < 100 or fps <= 0:
            rep['status'] = 'EXCLUDED'; rep['note'] = 'video không giải mã được'; continue
        rep.update(video_fps=round(fps, 3), video_frames=len(frame_times), video_duration=round(frame_times[-1], 2),
                   video_max_frame_gap=round(float(np.diff(frame_times).max()), 4))

        name_offset = (parse_ts(name) - parse_ts(os.path.basename(mp4))).total_seconds()
        base_offset = overrides.get(name, name_offset)
        rep['name_offset'] = name_offset
        rep['override_offset'] = overrides.get(name, '')
        sync = verify_offset(base_offset, flow_t, flow_x, acc, gps, args)
        rep.update({f'sync_{k}': (round(v, 3) if isinstance(v, float) else v) for k, v in sync.items() if k != 'name_offset'})
        if not sync['verified']:
            rep['status'] = 'EXCLUDED'
            rep['note'] = (f'đồng bộ không kiểm chứng được (tham chiếu cuối: {sync["ref"]}): '
                           f'quanh offset {base_offset:+.0f}s đỉnh {sync["near_lag"]:+.1f}s '
                           f'corr={sync["near_corr"]:.2f} ratio={sync["near_ratio"]:.2f}; quét rộng: {sync["wide_lag"]:+.1f}s '
                           f'corr={sync["wide_corr"]:.2f} ratio={sync["wide_ratio"]:.2f}')
            print(f'[EXCLUDED] {name}: {rep["note"]}')
            continue
        offset = sync['near_lag']  # offset tinh chỉnh từ dữ liệu, đã kiểm chứng
        rep['offset_used'] = round(offset, 2)

        ambiguous = [(acc_stats['ambiguous_from'], acc_stats['ambiguous_to']),
                     (gps_stats['ambiguous_from'], gps_stats['ambiguous_to'])]
        cands, rej = candidate_windows(acc, gps, offset, frame_times, fps, args, ambiguous)
        chosen = select_evenly(cands, args.max_windows_per_trip)
        rep['windows_valid'] = len(cands)
        rep['windows_chosen'] = len(chosen)
        rep['reject_sensor'] = json.dumps(rej)
        rep['status'] = 'OK'
        selected.append(dict(rep=rep, dir=d, mp4=mp4, acc=acc, fps=fps, frame_times=frame_times,
                             offset=offset, windows=chosen, split='val' if driver in val_drivers else 'train'))
        print(f'[OK] {name}: offset tên={name_offset:+.0f}s dùng={offset:+.1f}s ({sync["ref"]}) corr={sync["near_corr"]:.2f} '
              f'ratio={sync["near_ratio"]:.2f} | cửa sổ hợp lệ={len(cands)} chọn={len(chosen)} loại={rej} ({time.time()-t0:.0f}s)')

    # ---- ghi report chuyến
    keys = []
    for r in trip_report:
        for k in r:
            if k not in keys:
                keys.append(k)
    rep_path = os.path.join(args.out_dir, 'uah_trip_report.csv')
    os.makedirs(args.out_dir, exist_ok=True)  # phòng thư mục bị xoá trong lúc chạy dài
    with open(rep_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(trip_report)
    n_ok = sum(r['status'] == 'OK' for r in trip_report)
    print(f'\nReport chuyến: {rep_path}  ({n_ok}/{len(trip_report)} chuyến OK)')

    per_split = {'train': 0, 'val': 0}
    for s in selected:
        per_split[s['split']] += len(s['windows'])
    bytes_per = 3 * args.num_frames * args.size * args.size * (2 if args.video_dtype == 'float16' else 4)
    for sp, n in per_split.items():
        print(f'  {sp}: tối đa {n} cửa sổ, video ~{n * bytes_per / 1e9:.1f} GB')
    if args.check_only:
        print('check_only: không ghi pickle.')
        return

    # ---- trích frame + ghi pickle
    stats_path = os.path.join(args.out_dir, 'kine_norm_stats.json')
    kine_mean = kine_std = None
    for split in ['train', 'val']:
        trips = [s for s in selected if s['split'] == split]
        nmax = sum(len(s['windows']) for s in trips)
        if nmax == 0:
            print(f'[CẢNH BÁO] split {split} không có cửa sổ nào')
            continue
        tmp = os.path.join(args.out_dir, f'_video_{split}.tmp.npy')
        vid = np.lib.format.open_memmap(tmp, mode='w+', dtype=args.video_dtype,
                                        shape=(nmax, 3, args.num_frames, args.size, args.size))
        kine, labels, meta = [], [], []
        n = 0
        for s in trips:
            got, rej = extract_windows(s['mp4'], s['frame_times'], s['fps'], s['windows'], s['offset'], s['acc'], args)
            s['rep']['reject_video'] = json.dumps(rej)
            s['rep']['windows_written'] = len(got)
            for w, fr in got:
                x = (fr.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD      # (T,H,W,3)
                vid[n] = x.transpose(3, 0, 1, 2).astype(args.video_dtype)              # (3,T,H,W)
                seg = s['acc'][w['i0']:w['i0'] + args.kine_len]
                kine.append(seg[:, 5:8].T.astype(np.float32))                          # (3, 51)
                labels.append(s['rep']['label'])
                meta.append(dict(row=n, trip=s['rep']['trip'], driver=s['rep']['driver'], behavior=s['rep']['behavior'],
                                 road=s['rep']['road'], label=s['rep']['label'], data_t_start=w['t_start'],
                                 data_t_end=w['t_end'], offset_used=s['offset'], video_t_start=w['video_t_start'],
                                 gps_speed=w['gps_speed'], frame_diff_mean=round(w['frame_diff_mean'], 3)))
                n += 1
            print(f'  [{split}] {s["rep"]["trip"]}: ghi {len(got)}/{len(s["windows"])} cửa sổ, loại={rej}')
        vid.flush()
        kine = np.stack(kine)
        labels = np.asarray(labels, dtype=np.int64)
        if not args.no_kine_norm:
            if split == 'train':
                kine_mean = kine.mean(axis=(0, 2), keepdims=True)
                kine_std = kine.std(axis=(0, 2), keepdims=True)
                with open(stats_path, 'w') as f:
                    json.dump(dict(channels=['accX_kf', 'accY_kf', 'accZ_kf'], mean=kine_mean.ravel().tolist(),
                                   std=kine_std.ravel().tolist()), f, indent=2)
            if kine_mean is None:
                raise RuntimeError('Cần split train để tính thống kê chuẩn hoá kinematic')
            kine = (kine - kine_mean) / kine_std

        arr = np.asarray(vid[:n])  # ndarray view trên memmap, không copy
        with open(os.path.join(args.out_dir, f'video_{split}.pkl'), 'wb') as f:
            pickle.dump(arr, f, protocol=5)  # protocol 5 ghi thẳng buffer, không nhân đôi RAM
        del arr, vid
        os.remove(tmp)
        with open(os.path.join(args.out_dir, f'kine_{split}.pkl'), 'wb') as f:
            pickle.dump(kine, f, protocol=4)
        with open(os.path.join(args.out_dir, f'label_{split}.pkl'), 'wb') as f:
            pickle.dump(labels, f, protocol=4)
        with open(os.path.join(args.out_dir, f'windows_{split}.csv'), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(meta[0].keys()))
            w.writeheader()
            w.writerows(meta)
        counts = {b: int((labels == l).sum()) for b, l in LABELS.items()}
        print(f'==> {split}: {n} mẫu, nhãn {counts}, video {args.video_dtype} {(n,) + (3, args.num_frames, args.size, args.size)}')

    with open(rep_path, 'w', newline='', encoding='utf-8') as f:  # ghi lại report kèm số liệu lượt 2
        keys = []
        for r in trip_report:
            for k in r:
                if k not in keys:
                    keys.append(k)
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(trip_report)
    print('Xong.')


if __name__ == '__main__':
    main()
