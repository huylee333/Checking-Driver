"""A09. Dữ liệu bất thường và độc hại (Model AI).

Model thật, qua API thật (TestClient, trong tiến trình, giống T09) -- không cần Docker. Hai nhóm:
  1. NaN/Inf ở TẦNG VĂN BẢN: phải bị `inference.parse_table` chặn trước khi tới model (đã có kiểm tra
     `np.isfinite` -- test này xác nhận lại bằng HTTP thật, không chỉ đọc code).
  2. Dữ liệu CÓ SỐ (không NaN/Inf) nhưng phi vật lý/bất thường (toàn 0, hằng số lặp, gia tốc 10^8,
     nhiễu cực đoan): phải phân tích xong, không sập, không ra NaN/Inf trong XÁC SUẤT trả về.
"""
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as web
from inference import Analyzer
from synthetic import to_bytes, wait_for_job

WEIGHTS_DIR = os.environ.get('CD_WEIGHTS_DIR')
DEMO_DIR = os.environ.get('CD_DEMO_DIR')
pytestmark = pytest.mark.skipif(not (WEIGHTS_DIR and DEMO_DIR), reason='set CD_WEIGHTS_DIR and CD_DEMO_DIR')


@pytest.fixture(scope='module')
def client():
    analyzer = Analyzer(WEIGHTS_DIR)
    assert analyzer.ready, analyzer.error
    patch = pytest.MonkeyPatch()
    patch.setattr(web, 'analyzer', analyzer)
    yield TestClient(web.app)
    patch.undo()


def base_trip(n=701, hz=10.0):
    a = np.zeros((n, 11))
    a[:, 0] = np.arange(n) / hz
    a[:, 1] = 1
    return a


def upload_raw_text(client, text_bytes):
    return client.post('/api/analyze', files={'acc_file': ('RAW_ACCELEROMETERS.txt', text_bytes, 'text/plain')})


def upload_array(client, acc):
    return client.post('/api/analyze', files={'acc_file': ('RAW_ACCELEROMETERS.txt', to_bytes(acc), 'text/plain')})


def run_to_completion(client, response):
    assert response.status_code == 200, response.text
    body, _ = wait_for_job(client, response.json()['job_id'], timeout=120)
    assert body['status'] == 'done', body
    return body['result']


@pytest.mark.parametrize('token', ['nan', 'inf', '-inf', 'NaN', 'Infinity'])
def test_nan_and_inf_tokens_are_rejected_before_reaching_the_model(client, token):
    acc = base_trip()
    lines = []
    for i, row in enumerate(acc):
        cells = [f'{v:.6f}' for v in row]
        if i == 5:
            cells[5] = token          # một ô trong cột gia tốc Kalman (cột đưa vào model)
        lines.append(' '.join(cells))
    text = ('\n'.join(lines) + '\n').encode()
    res = upload_raw_text(client, text)
    assert res.status_code == 400, f'token={token!r}: mong 400, được {res.status_code}: {res.text[:200]}'
    assert 'không hợp lệ' in res.text or 'NaN' in res.text or 'vô cực' in res.text


@pytest.mark.parametrize('name,build', [
    ('toàn số 0', lambda acc: acc),
    ('hằng số lặp (không phải 0)', lambda acc: _set_const(acc, 0.5)),
    ('gia tốc phi vật lý (1e8)', lambda acc: _set_const(acc, 1e8)),
    ('nhiễu ngẫu nhiên cực đoan', lambda acc: _set_noise(acc, 1e4)),
])
def test_physically_implausible_but_finite_data_never_produces_nan_probabilities(client, name, build):
    acc = build(base_trip())
    result = run_to_completion(client, upload_array(client, acc))
    sensor = result['models']['sensor']
    assert sensor is not None, f'{name}: không có kết quả nhánh cảm biến'
    probs = sensor['trip_probs']
    assert all(np.isfinite(v) for v in probs.values()), f'{name}: xác suất không hữu hạn: {probs}'
    assert abs(sum(probs.values()) - 1.0) < 1e-3, f'{name}: tổng xác suất không phải 1: {probs}'
    for w in result['windows']:
        if w.get('sensor'):
            assert all(np.isfinite(p) for p in w['sensor']['probs']), f'{name}: một cửa sổ có xác suất NaN/Inf: {w}'


def _set_const(acc, value):
    acc = acc.copy()
    acc[:, 5:8] = value
    return acc


def _set_noise(acc, scale):
    acc = acc.copy()
    acc[:, 5:8] = np.random.default_rng(0).normal(0, scale, (len(acc), 3))
    return acc
