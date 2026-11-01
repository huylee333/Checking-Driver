"""
Suy luận DuST-DINOv3 cho một chuyến đi, gồm 3 model:
  - sensor : 1D Swin (Stage 2) trên 3 kênh gia tốc
  - video  : DINOv3 + Temporal Transformer (Stage 1) trên 77 frame mỗi đoạn
  - fusion : 1D Swin fusion (Stage 4) trên gia tốc + vector video 1024 chiều

Tiền xử lý PHẢI giống hệt lúc tạo dữ liệu train (dust_dinov3/prepare_uah_data.py):
  - làm sạch timestamp: bỏ phần sau lần quay ngược, bỏ cả nhóm dòng trùng timestamp,
    loại cửa sổ chạm khoảng thời gian bị ghi trùng;
  - cửa sổ = 51 mẫu RAW_ACCELEROMETERS liên tiếp, không chồng lấn, không nội suy;
  - loại cửa sổ có khoảng trống > 0.5 s, độ dài ngoài [4.6, 5.6] s, hoặc cờ kích hoạt != 1;
  - 3 kênh gia tốc đã lọc Kalman (cột 6-8), z-score bằng kine_norm_stats.json của tập train.
"""

import json
import os
import tempfile

import numpy as np
import torch

import video_pipeline as vp
from dinov3_encoder import DINOv3VideoEncoder
from swin1d import SwinTransformer_1D
from swin_fusion import SwinTransformer_1D as SwinTransformer_1D_fusion

CLASSES = ['NORMAL', 'AGGRESSIVE', 'DROWSY']
MODELS = ['sensor', 'video', 'fusion']
KINE_LEN = 51
STRIDE_SEC = 5.0
MAX_SAMPLE_GAP = 0.5
MIN_SPAN, MAX_SPAN = 4.6, 5.6
MIN_TRIP_SEC = 60            # chuyến ngắn hơn 1 phút: từ chối
MAX_TRIP_SEC = 2 * 3600      # chuyến dài hơn 2 giờ: từ chối (bảo vệ máy chủ)
EVENT_MIN_PROB = 0.6         # cửa sổ bất thường có xác suất >= ngưỡng này mới được gom thành "đoạn cần xem lại"
EVENT_HIGH_PROB = 0.8        # mức độ chắc chắn "Cao"
VIDEO_CPU_MAX_WINDOWS = int(os.environ.get('VIDEO_CPU_MAX_WINDOWS', 24))  # giới hạn số đoạn video khi chạy CPU


class InputError(ValueError):
    """Lỗi do file người dùng tải lên (thông báo tiếng Việt, hiển thị thẳng cho người dùng)."""


# ----------------------------------------------------------------------------
# Đọc + kiểm tra file
# ----------------------------------------------------------------------------
def decode_text(data, name):
    if not data or not data.strip():
        raise InputError(f'{name} trống.')
    if b'\x00' in data[:8192]:
        raise InputError(f'{name} không phải file văn bản (có thể là file nhị phân hoặc video).')
    for enc in ('utf-8-sig', 'utf-8'):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    raise InputError(f'{name} không đọc được (sai mã hoá ký tự).')


def parse_table(text, expected_cols, name, swapped_hint=None):
    rows = []
    for ln, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != expected_cols:
            hint = ''
            if swapped_hint and len(parts) == swapped_hint[0]:
                hint = ' ' + swapped_hint[1]
            elif ',' in line or ';' in line:
                hint = ' Các cột phải cách nhau bằng khoảng trắng (định dạng UAH-DriveSet), không dùng dấu phẩy.'
            raise InputError(f'{name}: dòng {ln} có {len(parts)} cột, cần {expected_cols} cột.{hint}')
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            raise InputError(f'{name}: dòng {ln} có giá trị không phải số.')
    if len(rows) < KINE_LEN:
        raise InputError(f'{name}: quá ít dữ liệu ({len(rows)} dòng).')
    a = np.asarray(rows, dtype=np.float64)
    bad = np.where(~np.isfinite(a).all(axis=1))[0]
    if len(bad):
        raise InputError(f'{name}: dòng {bad[0] + 1} có giá trị không hợp lệ (NaN hoặc vô cực).')
    return a


def clean_timeseries(a):
    info = {'rows_raw': int(len(a)), 'dropped_after_backjump': 0, 'dropped_duplicate_ts': 0, 'ambiguous': None}
    t = a[:, 0]
    back = np.where(np.diff(t) < 0)[0]
    if len(back):
        cut = back[0] + 1
        info['dropped_after_backjump'] = int(len(a) - cut)
        info['ambiguous'] = (float(t[cut:].min()), float(t[cut - 1]))
        a = a[:cut]
        t = a[:, 0]
    _, idx, cnt = np.unique(t, return_index=True, return_counts=True)
    keep = np.zeros(len(t), dtype=bool)
    keep[idx[cnt == 1]] = True
    info['dropped_duplicate_ts'] = int((~keep).sum())
    return a[keep], info


def make_windows(acc, ambiguous):
    """Trả về list cửa sổ (cả hợp lệ lẫn không) để vẽ kín dòng thời gian."""
    t = acc[:, 0]
    out = []
    start = t[0]
    while True:
        i0 = int(np.searchsorted(t, start))
        if i0 + KINE_LEN > len(t):
            break
        seg = acc[i0:i0 + KINE_LEN]
        ts = seg[:, 0]
        span = ts[-1] - ts[0]
        reason = None
        if np.diff(ts).max() > MAX_SAMPLE_GAP or not (MIN_SPAN <= span <= MAX_SPAN):
            reason = 'mất dữ liệu cảm biến'
        elif (seg[:, 1] != 1).any():
            reason = 'cảm biến chưa sẵn sàng'
        elif any(a is not None and ts[0] <= a[1] and ts[-1] >= a[0] for a in ambiguous):
            reason = 'dữ liệu ghi trùng, không rõ ràng'
        out.append(dict(i0=i0, t_start=float(ts[0]), t_end=float(ts[-1]), valid=reason is None, reason=reason))
        nxt = ts[0] + STRIDE_SEC
        if STRIDE_SEC >= span:
            nxt = max(nxt, ts[-1] + 1e-6)
        start = nxt
    return out


def load_sensors(acc_bytes, gps_bytes=None):
    """Kiểm tra + làm sạch file cảm biến (nhanh, chạy ngay khi upload). Ném InputError nếu không hợp lệ."""
    acc_text = decode_text(acc_bytes, 'File cảm biến chuyển động')
    acc, acc_info = clean_timeseries(parse_table(
        acc_text, 11, 'File cảm biến chuyển động',
        swapped_hint=(12, 'Có vẻ đây là file GPS. Hãy chọn RAW_ACCELEROMETERS.txt.')))
    gps, gps_info = None, None
    if gps_bytes:
        gps_text = decode_text(gps_bytes, 'File GPS')
        gps, gps_info = clean_timeseries(parse_table(
            gps_text, 12, 'File GPS',
            swapped_hint=(11, 'Có vẻ đây là file cảm biến chuyển động. Hãy chọn RAW_GPS.txt.')))
    if len(acc) < KINE_LEN:
        raise InputError('File cảm biến chuyển động: còn quá ít dữ liệu sau khi làm sạch.')
    dur = float(acc[-1, 0] - acc[0, 0])
    if dur < MIN_TRIP_SEC:
        raise InputError(f'Chuyến đi quá ngắn ({dur:.0f} giây). Cần ít nhất {MIN_TRIP_SEC // 60} phút dữ liệu.')
    if dur > MAX_TRIP_SEC:
        raise InputError(f'Chuyến đi quá dài ({dur / 3600:.1f} giờ). Hệ thống chỉ nhận tối đa {MAX_TRIP_SEC // 3600} giờ.')
    ambiguous = [acc_info['ambiguous'], gps_info['ambiguous'] if gps_info else None]
    windows = make_windows(acc, ambiguous)
    if not any(w['valid'] for w in windows):
        raise InputError('Không có đoạn dữ liệu nào đủ chất lượng để phân tích (mất dữ liệu hoặc cảm biến chưa sẵn sàng).')
    return dict(acc=acc, gps=gps, acc_info=acc_info, gps_info=gps_info, windows=windows)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def _softmax3(logits):
    """Model có 4 đầu ra như bản gốc (lớp 3 không dùng khi train) -> softmax trên 3 lớp thật."""
    return torch.softmax(logits[:, :3].float(), dim=1).numpy()


class Analyzer:
    def __init__(self, weights_dir, demo_dir=None):
        self.error = None
        self.model_1d = self.video = self.fusion = None
        self.demo = None
        w = lambda f: os.path.join(weights_dir, f)
        if not (os.path.isfile(w('final_1d.pth')) and os.path.isfile(w('kine_norm_stats.json'))):
            self.error = f'Chưa có model: cần {w("final_1d.pth")} và {w("kine_norm_stats.json")}'
            return
        stats = json.load(open(w('kine_norm_stats.json')))
        self.mean = np.asarray(stats['mean'], dtype=np.float32)[:, None]
        self.std = np.asarray(stats['std'], dtype=np.float32)[:, None]
        # Cấu hình giống dust_dinov3/step_fusion_1d.py và step_fusion.py
        cfg = dict(seq_len=KINE_LEN, in_chans=3, num_classes=4, window_size=8, patch_size=1,
                   num_heads=[16, 16, 16, 16], depths=[2, 2, 6, 2], embed_dim=256)
        self.model_1d = SwinTransformer_1D(drop_path_rate=0.1, **cfg)
        self.model_1d.load_state_dict(torch.load(w('final_1d.pth'), map_location='cpu'))
        self.model_1d.eval()
        if os.path.isfile(w('video_3d.pth')):
            self.video = DINOv3VideoEncoder(num_classes=4, num_frames=vp.NUM_FRAMES, output_dim=1024,
                                            freeze_backbone=True, pretrained=False)
            self.video.load_state_dict(torch.load(w('video_3d.pth'), map_location='cpu'))
            self.video.eval()
        if self.video is not None and os.path.isfile(w('fusion.pth')):
            self.fusion = SwinTransformer_1D_fusion(**cfg)
            self.fusion.load_state_dict(torch.load(w('fusion.pth'), map_location='cpu'))
            self.fusion.eval()
        if demo_dir and os.path.isfile(os.path.join(demo_dir, 'meta.json')):
            self.demo = dict(dir=demo_dir, meta=json.load(open(os.path.join(demo_dir, 'meta.json'), encoding='utf-8')))

    @property
    def ready(self):
        return self.model_1d is not None

    def status(self):
        return dict(ready=self.ready, video=self.video is not None, fusion=self.fusion is not None,
                    demo=self.demo['meta'] if self.demo else None, message=self.error or 'Sẵn sàng')

    def demo_sensors(self):
        d = self.demo['dir']
        acc = open(os.path.join(d, 'RAW_ACCELEROMETERS.txt'), 'rb').read()
        gps_p = os.path.join(d, 'RAW_GPS.txt')
        gps = open(gps_p, 'rb').read() if os.path.isfile(gps_p) else None
        return load_sensors(acc, gps)

    # ---- tiện ích
    def _kine(self, sens, idx):
        acc, wins = sens['acc'], sens['windows']
        x = np.stack([acc[wins[i]['i0']:wins[i]['i0'] + KINE_LEN, 5:8].T for i in idx]).astype(np.float32)
        return (x - self.mean) / self.std

    @torch.no_grad()
    def _predict_1d(self, x):
        out = [_softmax3(self.model_1d(torch.from_numpy(x[i:i + 64]))) for i in range(0, len(x), 64)]
        return np.concatenate(out)

    @torch.no_grad()
    def _video_and_fusion(self, cls, kine):
        """cls: (N, 77, 384) CLS token DINOv3; kine: (N, 3, 51) đã chuẩn hoá."""
        pv, pf = [], []
        for i in range(0, len(cls), 32):
            feat = self.video._temporal_aggregate(torch.from_numpy(cls[i:i + 32].astype(np.float32)))
            pv.append(_softmax3(self.video.head(feat)))
            if self.fusion is not None:
                pf.append(_softmax3(self.fusion(torch.from_numpy(kine[i:i + 32]), feat)))
        return np.concatenate(pv), (np.concatenate(pf) if pf else None)

    # ---- nhánh video
    def _video_branch(self, sens, video, progress, gpu_fn):
        """Trả về (cls theo chỉ số cửa sổ, thumbs theo chỉ số cửa sổ, thông tin video)."""
        acc, gps, wins = sens['acc'], sens['gps'], sens['windows']
        info = dict(status='none', message='')
        path = video['path']
        progress('Đang đồng bộ video với dữ liệu cảm biến…')
        fps, frame_times, flow_t, flow_x = vp.scan_video(path, progress)
        base = vp.name_offset(video.get('trip_name'), video.get('video_name'))
        sync = vp.verify_offset(base, flow_t, flow_x, acc, gps)
        info.update(sync=dict((k, v) for k, v in sync.items() if k != 'reason'))
        if not sync['verified']:
            info.update(status='sync_failed',
                        message=f'Không đồng bộ được video với cảm biến ({sync.get("reason")}). '
                                f'Chỉ hiện kết quả từ dữ liệu cảm biến.')
            return {}, {}, info
        offset = sync['offset']
        plans = vp.plan_frames(wins, acc, offset, frame_times, fps)
        if not plans:
            info.update(status='no_overlap', message='Video không trùng thời gian với đoạn dữ liệu cảm biến nào.')
            return {}, {}, info
        with tempfile.TemporaryDirectory() as tmp:
            fpath = os.path.join(tmp, 'frames.npy')
            progress('Đang lấy frame video…')
            complete, thumbs = vp.extract_frames(path, plans, fpath, progress)
            rows = [i for i, ok in enumerate(complete) if ok]
            device = 'cpu'
            cls = None
            if gpu_fn is not None:
                try:
                    progress(f'Đang chạy DINOv3 trên GPU cho {len(rows)} đoạn…')
                    cls = gpu_fn(fpath, rows)
                    device = 'gpu'
                except Exception as e:  # hết hạn mức ZeroGPU, lỗi GPU... -> chạy CPU
                    info['gpu_error'] = str(e)[:300]
            if cls is None:
                total = len(rows)
                if len(rows) > VIDEO_CPU_MAX_WINDOWS:
                    pick = np.unique(np.round(np.linspace(0, len(rows) - 1, VIDEO_CPU_MAX_WINDOWS)).astype(int))
                    rows = [rows[i] for i in pick]
                progress(f'Đang chạy DINOv3 trên CPU cho {len(rows)} đoạn…')
                cls = vp.backbone_cls(self.video, fpath, rows, device='cpu', progress=progress)
                if len(rows) < total:
                    info.update(status='cpu_limited',
                                message=f'Không có GPU nên chỉ phân tích video {len(rows)}/{total} đoạn, '
                                        f'trải đều trong chuyến. Các đoạn còn lại chỉ có kết quả cảm biến.')
        cls_by_win = {plans[r][0]: cls[k] for k, r in enumerate(rows)}
        thumb_by_win = {plans[pi][0]: thumbs[pi] for pi in range(len(plans)) if thumbs[pi]}
        if info['status'] == 'none':
            info['status'] = 'ok'
        info.update(offset=offset, device=device, windows_analysed=len(cls_by_win), windows_possible=len(plans))
        return cls_by_win, thumb_by_win, info

    def _demo_branch(self, sens):
        d = self.demo['dir']
        z = np.load(os.path.join(d, 'cache.npz'))
        thumbs = json.load(open(os.path.join(d, 'thumbs.json'), encoding='utf-8'))
        key = {round(float(t), 2): k for k, t in enumerate(z['t_start'])}
        cls_by_win, thumb_by_win = {}, {}
        for wi, w in enumerate(sens['windows']):
            k = key.get(round(w['t_start'], 2))
            if k is not None and w['valid']:
                cls_by_win[wi] = z['cls'][k]
                if thumbs[k]:
                    thumb_by_win[wi] = thumbs[k]
        meta = self.demo['meta']
        info = dict(status='ok', device='tính sẵn (chuyến mẫu)', offset=meta.get('offset'), sync=meta.get('sync'),
                    windows_analysed=len(cls_by_win), windows_possible=len(cls_by_win),
                    message='Đặc trưng DINOv3 của chuyến mẫu được tính sẵn; Temporal Transformer, '
                            'nhánh video và fusion chạy trực tiếp.')
        return cls_by_win, thumb_by_win, info

    # ---- toàn bộ
    def analyze(self, sens, video=None, demo=False, progress=None, gpu_fn=None):
        if not self.ready:
            raise RuntimeError(self.error)
        progress = progress or (lambda m: None)
        wins = sens['windows']
        valid_idx = [i for i, w in enumerate(wins) if w['valid']]

        progress('Đang phân tích dữ liệu cảm biến…')
        kine = self._kine(sens, valid_idx)
        p1 = self._predict_1d(kine)
        for i, p in zip(valid_idx, p1):
            wins[i]['sensor'] = _pack(p)

        cls_by_win, thumb_by_win = {}, {}
        vinfo = dict(status='none', message='Không có video. Chỉ phân tích dữ liệu cảm biến.')
        if self.video is not None and (demo or video):
            if demo:
                cls_by_win, thumb_by_win, vinfo = self._demo_branch(sens)
            else:
                cls_by_win, thumb_by_win, vinfo = self._video_branch(sens, video, progress, gpu_fn)
        elif video and self.video is None:
            vinfo = dict(status='none', message='Máy chủ chưa có model video. Chỉ phân tích dữ liệu cảm biến.')

        if cls_by_win:
            progress('Đang chạy nhánh video và fusion…')
            vidx = sorted(cls_by_win)
            cls = np.stack([cls_by_win[i] for i in vidx])
            pv, pf = self._video_and_fusion(cls, self._kine(sens, vidx))
            for k, i in enumerate(vidx):
                wins[i]['video'] = _pack(pv[k])
                if pf is not None:
                    wins[i]['fusion'] = _pack(pf[k])

        thumbs, out_w = [], []
        for i, w in enumerate(wins):
            ow = {k: w.get(k) for k in ('t_start', 't_end', 'valid', 'reason', 'sensor', 'video', 'fusion')}
            if i in thumb_by_win:
                ow['thumb'] = len(thumbs)
                thumbs.append(thumb_by_win[i])
            out_w.append(ow)

        return dict(
            demo=bool(demo),
            trip=trip_info(sens),
            models={m: summarize(out_w, m) for m in MODELS},
            windows=out_w,
            thumbs=thumbs,
            video=vinfo,
            series=make_series(sens['acc'], sens['gps']),
            cleaning=dict(acc=sens['acc_info'], gps=sens['gps_info']),
        )


def _pack(p):
    return dict(probs=[round(float(v), 4) for v in p], label=CLASSES[int(np.argmax(p))])


# ----------------------------------------------------------------------------
# Tổng hợp kết quả cho giao diện
# ----------------------------------------------------------------------------
def trip_info(sens):
    acc, gps, wins = sens['acc'], sens['gps'], sens['windows']
    res = dict(start=float(acc[0, 0]), end=float(acc[-1, 0]), duration=float(acc[-1, 0] - acc[0, 0]),
               windows_total=len(wins), windows_valid=sum(w['valid'] for w in wins))
    if gps is not None and len(gps) > 1:
        v = gps[:, 1] / 3.6
        res['distance_km'] = float(np.sum((v[1:] + v[:-1]) / 2 * np.diff(gps[:, 0])) / 1000)
        res['max_speed'] = float(gps[:, 1].max())
    return res


def summarize(windows, key):
    have = [w for w in windows if w.get(key)]
    if not have:
        return None
    dur = {c: 0.0 for c in CLASSES}
    for w in have:
        dur[w[key]['label']] += w['t_end'] - w['t_start']
    analysed = sum(dur.values())
    probs = np.array([w[key]['probs'] for w in have])
    trip_probs = {c: float(probs[:, i].mean()) for i, c in enumerate(CLASSES)}
    return dict(windows=len(have), analysed_seconds=analysed, seconds=dur,
                share={c: (dur[c] / analysed if analysed else 0.0) for c in CLASSES},
                trip_probs=trip_probs, trip_label=max(trip_probs, key=trip_probs.get),
                events=find_events(windows, key))


def find_events(windows, key):
    """Gom các cửa sổ liền nhau cùng nhãn bất thường (xác suất >= EVENT_MIN_PROB) thành một đoạn."""
    events, cur = [], None
    for w in windows:
        r = w.get(key)
        hit = bool(r) and r['label'] != 'NORMAL' and max(r['probs']) >= EVENT_MIN_PROB
        if hit and cur and cur['label'] == r['label'] and w['t_start'] - cur['t_end'] <= STRIDE_SEC + 1.0:
            cur['t_end'] = w['t_end']
            cur['probs'].append(max(r['probs']))
            continue
        if cur:
            events.append(cur)
            cur = None
        if hit:
            cur = dict(label=r['label'], t_start=w['t_start'], t_end=w['t_end'], probs=[max(r['probs'])])
    if cur:
        events.append(cur)
    for e in events:
        e['confidence'] = float(np.mean(e['probs']))
        e['level'] = 'high' if e['confidence'] >= EVENT_HIGH_PROB else 'medium'
        e['n_windows'] = len(e.pop('probs'))
    return events


def make_series(acc, gps):
    """Dữ liệu vẽ biểu đồ: độ mạnh chuyển động (g, 10 Hz) và tốc độ GPS (km/h, 1 Hz)."""
    mag = np.sqrt((acc[:, 5:8] ** 2).sum(axis=1))
    out = dict(t=np.round(acc[:, 0], 2).tolist(), motion=np.round(mag, 4).tolist())
    if gps is not None:
        out['gps_t'] = np.round(gps[:, 0], 2).tolist()
        out['speed'] = np.round(gps[:, 1], 1).tolist()
    return out
