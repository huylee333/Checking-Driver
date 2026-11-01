"""
Web kiểm tra hành vi lái xe (bản thử nghiệm DuST-DINOv3: cảm biến + video + fusion).

Chạy trên máy:   python app.py              (mở http://localhost:7860)
Deploy:          Hugging Face Spaces (SDK Gradio, phần cứng ZeroGPU), xem README.md

Luồng xử lý:
  POST /api/analyze  -> kiểm tra file ngay (dung lượng, loại file, nội dung); hợp lệ thì đưa vào hàng đợi
  POST /api/demo     -> đưa "chuyến mẫu" vào hàng đợi (không cần upload)
  GET  /api/job/{id} -> trạng thái, tiến độ, vị trí trong hàng đợi, kết quả
Mỗi lúc chỉ xử lý 1 chuyến (1 luồng worker), tối đa MAX_PENDING yêu cầu đang chờ.
"""

import os

# Trên Space ZeroGPU, `spaces` phải được import trước mọi thứ có thể khởi tạo CUDA.
try:
    import spaces
except ImportError:
    spaces = None

import queue
import shutil
import tempfile
import threading
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

import video_pipeline as vp
from inference import MAX_TRIP_SEC, MIN_TRIP_SEC, Analyzer, InputError, load_sensors

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.environ.get('WEIGHTS_DIR', os.path.join(HERE, 'weights'))
DEMO_DIR = os.environ.get('DEMO_DIR', os.path.join(HERE, 'demo'))
DEMO_REPO = os.environ.get('DEMO_REPO', 'huy-le/checking-driver-demo')   # dataset PRIVATE chứa chuyến mẫu
# Space miễn phí giới hạn 1 GB, không chứa nổi 3 model (~1.6 GB) -> model nằm ở model repo PRIVATE riêng
WEIGHTS_REPO = os.environ.get('WEIGHTS_REPO', 'huy-le/checking-driver-weights')
WEIGHT_FILES = ['final_1d.pth', 'kine_norm_stats.json', 'video_3d.pth', 'fusion.pth']

TEXT_MAX_MB = 20
VIDEO_MAX_MB = 300
TOTAL_MAX_BYTES = (2 * TEXT_MAX_MB + VIDEO_MAX_MB + 2) * 1024 * 1024
TEXT_EXT = {'.txt', '.csv'}
VIDEO_EXT = {'.mp4', '.mov', '.avi', '.mkv', '.m4v'}
MAX_PENDING = 3          # số yêu cầu tối đa đang chờ + đang chạy
JOB_TTL_SEC = 1800       # kết quả giữ trong 30 phút


def _fetch_demo():
    """Tải chuyến mẫu từ dataset private (cần secret HF_TOKEN trên Space). Lỗi thì web vẫn chạy, chỉ tắt nút demo."""
    if os.path.isfile(os.path.join(DEMO_DIR, 'meta.json')):
        return
    token = os.environ.get('HF_TOKEN')
    if not token:
        return
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=DEMO_REPO, repo_type='dataset', local_dir=DEMO_DIR, token=token)
    except Exception as e:
        print(f'[demo] không tải được chuyến mẫu: {e}')


def _fetch_weights():
    """Tải model từ model repo private nếu thư mục weights/ còn thiếu file (luôn đúng khi chạy trên Space)."""
    if all(os.path.isfile(os.path.join(WEIGHTS_DIR, f)) for f in WEIGHT_FILES):
        return
    token = os.environ.get('HF_TOKEN')
    if not token:
        print('[weights] thiếu file model và không có HF_TOKEN để tải')
        return
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=WEIGHTS_REPO, repo_type='model', local_dir=WEIGHTS_DIR, token=token,
                          allow_patterns=WEIGHT_FILES)
    except Exception as e:
        print(f'[weights] không tải được model: {e}')


_fetch_weights()
_fetch_demo()
app = FastAPI(title='Kiểm tra hành vi lái xe')
analyzer = Analyzer(WEIGHTS_DIR, demo_dir=DEMO_DIR)

# ZeroGPU: chạy DINOv3 trên GPU. Space ZeroGPU cũng bắt buộc có ít nhất 1 hàm @spaces.GPU.
GPU_FN = None
if spaces is not None:
    @spaces.GPU(duration=150)
    def _gpu_backbone(frames_path, rows):
        return vp.backbone_cls(analyzer.video, frames_path, rows, device='cuda')

    if analyzer.video is not None:
        GPU_FN = _gpu_backbone


# ----------------------------------------------------------------------------
# Hàng đợi xử lý
# ----------------------------------------------------------------------------
JOBS = {}
LOCK = threading.Lock()
QUEUE = queue.Queue()


def _cleanup():
    now = time.time()
    with LOCK:
        for jid in [j for j, v in JOBS.items() if v['status'] in ('done', 'error') and now - v['created'] > JOB_TTL_SEC]:
            JOBS.pop(jid, None)


def _worker():
    while True:
        jid = QUEUE.get()
        job = JOBS.get(jid)
        if job is None:
            continue
        job['status'] = 'running'
        progress = lambda msg: job.__setitem__('progress', msg)
        try:
            if job['kind'] == 'demo':
                sens = analyzer.demo_sensors()
                job['result'] = analyzer.analyze(sens, demo=True, progress=progress)
            else:
                job['result'] = analyzer.analyze(job['sens'], video=job.get('video'), progress=progress, gpu_fn=GPU_FN)
            job['status'] = 'done'
            job['progress'] = 'Xong'
        except InputError as e:
            job.update(status='error', error=str(e))
        except Exception as e:  # lỗi bất ngờ: báo chung, ghi chi tiết ra log của Space
            print(f'[job {jid}] lỗi: {e!r}')
            job.update(status='error', error='Có lỗi khi phân tích chuyến đi này. Hãy thử lại hoặc chọn file khác.')
        finally:
            job.pop('sens', None)
            if job.get('tmp'):
                shutil.rmtree(job['tmp'], ignore_errors=True)
            job['created'] = time.time()
            QUEUE.task_done()


threading.Thread(target=_worker, daemon=True).start()


def _submit(kind, **payload):
    _cleanup()
    with LOCK:
        pending = sum(1 for v in JOBS.values() if v['status'] in ('queued', 'running'))
        if pending >= MAX_PENDING:
            raise HTTPException(429, 'Hệ thống đang bận phân tích các chuyến khác. Vui lòng thử lại sau vài phút.')
        jid = uuid.uuid4().hex
        JOBS[jid] = dict(kind=kind, status='queued', progress='Đang chờ đến lượt…', created=time.time(), **payload)
    QUEUE.put(jid)
    return {'job_id': jid}


# ----------------------------------------------------------------------------
# Giới hạn dung lượng
# ----------------------------------------------------------------------------
@app.middleware('http')
async def limit_body(request: Request, call_next):
    if request.method == 'POST':
        cl = request.headers.get('content-length')
        if cl and cl.isdigit() and int(cl) > TOTAL_MAX_BYTES:
            return JSONResponse(status_code=413, content={
                'detail': f'Dữ liệu tải lên quá lớn. Giới hạn: file văn bản {TEXT_MAX_MB} MB, video {VIDEO_MAX_MB} MB.'})
    return await call_next(request)


def _ext(f):
    return os.path.splitext(f.filename or '')[1].lower()


async def _read_text(f, name):
    ext = _ext(f)
    if ext in VIDEO_EXT:
        raise HTTPException(400, f'{name}: đây là file video. Hãy chọn file video ở ô "Video hành trình".')
    if ext not in TEXT_EXT:
        raise HTTPException(400, f'{name}: chỉ nhận file .txt (định dạng UAH-DriveSet).')
    data = bytearray()
    while chunk := await f.read(1 << 20):
        data.extend(chunk)
        if len(data) > TEXT_MAX_MB * 1024 * 1024:
            raise HTTPException(413, f'{name} quá lớn (tối đa {TEXT_MAX_MB} MB).')
    return bytes(data)


async def _save_video(f, folder):
    ext = _ext(f)
    if ext not in VIDEO_EXT:
        raise HTTPException(400, f'Video: định dạng "{ext or "không rõ"}" không được hỗ trợ. Hãy chọn file .mp4.')
    path = os.path.join(folder, 'video' + ext)
    size = 0
    with open(path, 'wb') as out:
        while chunk := await f.read(1 << 20):
            size += len(chunk)
            if size > VIDEO_MAX_MB * 1024 * 1024:
                raise HTTPException(413, f'Video quá lớn (tối đa {VIDEO_MAX_MB} MB).')
            out.write(chunk)
    if size == 0:
        raise HTTPException(400, 'Video trống.')
    try:
        _, _, dur = await run_in_threadpool(vp.probe_video, path)
    except ValueError as e:
        raise HTTPException(400, f'Video: {e}')
    if dur < MIN_TRIP_SEC:
        raise HTTPException(400, f'Video quá ngắn ({dur:.0f} giây). Cần ít nhất {MIN_TRIP_SEC // 60} phút.')
    if dur > MAX_TRIP_SEC + 600:
        raise HTTPException(400, f'Video quá dài ({dur / 3600:.1f} giờ). Hệ thống chỉ nhận tối đa {MAX_TRIP_SEC // 3600} giờ.')
    return path


# ----------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------
@app.get('/')
def index():
    return FileResponse(os.path.join(HERE, 'static', 'index.html'))


@app.get('/api/status')
def status():
    s = analyzer.status()
    s.update(gpu=GPU_FN is not None, limits=dict(text_mb=TEXT_MAX_MB, video_mb=VIDEO_MAX_MB))
    return s


@app.post('/api/analyze')
async def analyze(acc_file: UploadFile = File(...),
                  gps_file: UploadFile | None = File(None),
                  video_file: UploadFile | None = File(None),
                  trip_name: str = Form('')):
    if not analyzer.ready:
        raise HTTPException(503, 'Hệ thống chưa có model. ' + (analyzer.error or ''))
    acc = await _read_text(acc_file, 'File cảm biến chuyển động')
    gps = await _read_text(gps_file, 'File GPS') if gps_file is not None and gps_file.filename else None
    try:
        sens = await run_in_threadpool(load_sensors, acc, gps)
    except InputError as e:
        raise HTTPException(400, str(e))
    tmp, video = None, None
    if video_file is not None and video_file.filename:
        tmp = tempfile.mkdtemp(prefix='trip_')
        try:
            path = await _save_video(video_file, tmp)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        video = dict(path=path, trip_name=trip_name, video_name=video_file.filename)
    try:
        return _submit('upload', sens=sens, video=video, tmp=tmp)
    except HTTPException:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
        raise


@app.post('/api/demo')
def demo():
    if not analyzer.ready:
        raise HTTPException(503, 'Hệ thống chưa có model.')
    if not analyzer.demo:
        raise HTTPException(404, 'Máy chủ chưa có chuyến mẫu.')
    return _submit('demo')


@app.get('/api/job/{jid}')
def job_status(jid: str):
    job = JOBS.get(jid)
    if job is None:
        raise HTTPException(404, 'Không tìm thấy yêu cầu phân tích (có thể đã hết hạn). Hãy phân tích lại.')
    out = dict(status=job['status'], progress=job.get('progress', ''))
    if job['status'] == 'queued':
        with LOCK:
            out['queue_position'] = sum(1 for v in JOBS.values()
                                        if v['status'] in ('queued', 'running') and v['created'] < job['created'])
    elif job['status'] == 'done':
        out['result'] = job['result']
    elif job['status'] == 'error':
        out['error'] = job.get('error')
    return out


if __name__ == '__main__':
    # Space SDK Gradio khởi động bằng `python app.py` và chờ web ở cổng 7860.
    import uvicorn

    # ZeroGPU chỉ "báo cáo khởi động" bên trong gr.Blocks.launch(); web này không gọi launch()
    # nên gọi trực tiếp bước khởi động của spaces trước khi chạy uvicorn.
    if spaces is not None:
        try:
            import spaces.zero as _zero
            if hasattr(_zero, 'startup'):
                _zero.startup()
        except ImportError:
            pass

    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 7860)))
