"""
Tạo dữ liệu "chuyến mẫu" cho web (chạy 1 lần trên máy, rồi upload lên dataset PRIVATE trên Hugging Face).

Dùng lại đúng các hàm của web (inference.load_sensors, video_pipeline.*) để cửa sổ, đồng bộ
và frame khớp tuyệt đối với lúc web chạy. Phần tốn thời gian nhất (DINOv3 trên ~14k frame)
được tính sẵn ở đây; khi bấm "Phân tích chuyến mẫu", web vẫn chạy thật Temporal Transformer,
nhánh video và fusion trên các đặc trưng này.

Usage (từ thư mục web/):
  python tools/precompute_demo.py --trip "<thư mục chuyến UAH>" --out demo
Kết quả trong demo/: RAW_ACCELEROMETERS.txt, RAW_GPS.txt, cache.npz, thumbs.json, meta.json
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import inference as inf          # noqa: E402
import video_pipeline as vp      # noqa: E402

try:
    sys.stdout.reconfigure(errors='backslashreplace')
except AttributeError:
    pass

TRIP_RE = re.compile(r'(\d{14})-(\d+)km-(D\d)-(NORMAL|AGGRESSIVE|DROWSY)(\d?)-(MOTORWAY|SECONDARY)')
LABEL_VI = {'NORMAL': 'Bình thường', 'AGGRESSIVE': 'Lái xe hung hăng', 'DROWSY': 'Dấu hiệu buồn ngủ'}
ROAD_VI = {'MOTORWAY': 'cao tốc', 'SECONDARY': 'đường thường'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trip', required=True, help='thư mục 1 chuyến UAH (chứa .mp4, RAW_ACCELEROMETERS.txt, RAW_GPS.txt)')
    p.add_argument('--out', default=os.path.join(os.path.dirname(HERE), 'demo'))
    p.add_argument('--weights', default=os.path.join(os.path.dirname(HERE), 'weights'))
    args = p.parse_args()

    trip = os.path.normpath(args.trip)
    name = os.path.basename(trip)
    mp4 = glob.glob(os.path.join(trip, '*.mp4'))
    assert len(mp4) == 1, f'cần đúng 1 file .mp4 trong {trip}'
    mp4 = mp4[0]
    t0 = time.time()
    log = lambda m: print(f'[{time.time() - t0:6.0f}s] {m}', flush=True)

    acc_b = open(os.path.join(trip, 'RAW_ACCELEROMETERS.txt'), 'rb').read()
    gps_b = open(os.path.join(trip, 'RAW_GPS.txt'), 'rb').read()
    sens = inf.load_sensors(acc_b, gps_b)
    analyzer = inf.Analyzer(args.weights)
    assert analyzer.video is not None, 'thiếu weights/video_3d.pth'

    log('Đồng bộ video…')
    fps, frame_times, flow_t, flow_x = vp.scan_video(mp4)
    base = vp.name_offset(name, os.path.basename(mp4))
    sync = vp.verify_offset(base, flow_t, flow_x, sens['acc'], sens['gps'])
    log(f'sync: {sync}')
    assert sync['verified'], 'chuyến này không kiểm chứng được đồng bộ, hãy chọn chuyến khác'
    plans = vp.plan_frames(sens['windows'], sens['acc'], sync['offset'], frame_times, fps)

    with tempfile.TemporaryDirectory() as tmp:
        fpath = os.path.join(tmp, 'frames.npy')
        log(f'Lấy frame cho {len(plans)} đoạn…')
        complete, thumbs = vp.extract_frames(mp4, plans, fpath, log)
        rows = [i for i, ok in enumerate(complete) if ok]
        log(f'DINOv3 (CPU) cho {len(rows)} đoạn…')
        cls = vp.backbone_cls(analyzer.video, fpath, rows, device='cpu', progress=log)

    os.makedirs(args.out, exist_ok=True)
    t_start = np.array([sens['windows'][plans[r][0]]['t_start'] for r in rows])
    np.savez_compressed(os.path.join(args.out, 'cache.npz'), t_start=t_start, cls=cls)
    json.dump([thumbs[r] for r in rows], open(os.path.join(args.out, 'thumbs.json'), 'w', encoding='utf-8'))
    shutil.copy2(os.path.join(trip, 'RAW_ACCELEROMETERS.txt'), args.out)
    shutil.copy2(os.path.join(trip, 'RAW_GPS.txt'), args.out)

    m = TRIP_RE.search(name)
    meta = dict(trip_name=name, offset=sync['offset'],
                sync={k: v for k, v in sync.items() if k != 'reason'},
                windows=len(rows), duration=float(sens['acc'][-1, 0] - sens['acc'][0, 0]))
    if m:
        _, km, drv, beh, _, road = m.groups()
        meta.update(driver=drv, behavior=beh, road=road, km=int(km),
                    title=f'Tài xế {drv} · {ROAD_VI[road]} · {km} km',
                    reference=LABEL_VI[beh])
    json.dump(meta, open(os.path.join(args.out, 'meta.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    log(f'Xong: {args.out} ({len(rows)} đoạn)')


if __name__ == '__main__':
    main()
