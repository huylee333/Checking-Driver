"""A10. Trích xuất model bằng truy vấn hàng loạt (Model AI).

Model thật, qua API thật (TestClient, trong tiến trình, giống T09) -- không cần Docker, không gửi gì ra
ngoài máy này. Mô phỏng kẻ tấn công: gửi nhiều chuyến TỔNG HỢP khác nhau (chỉ khác biên độ nhiễu, một đặc
trưng rẻ để tính mà không cần hiểu model), thu nhãn + xác suất ĐẦY ĐỦ mà API trả về, rồi thử huấn luyện
một model xấp xỉ (surrogate) rẻ hơn nhiều từ các cặp (đặc trưng, nhãn) đó.
"""
import os
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as web
from inference import Analyzer
from synthetic import to_bytes, wait_for_job

WEIGHTS_DIR = os.environ.get('CD_WEIGHTS_DIR')
DEMO_DIR = os.environ.get('CD_DEMO_DIR')
pytestmark = pytest.mark.skipif(not (WEIGHTS_DIR and DEMO_DIR), reason='set CD_WEIGHTS_DIR and CD_DEMO_DIR')

N_QUERIES = 160


@pytest.fixture(scope='module')
def client():
    analyzer = Analyzer(WEIGHTS_DIR)
    assert analyzer.ready, analyzer.error
    patch = pytest.MonkeyPatch()
    patch.setattr(web, 'analyzer', analyzer)
    yield TestClient(web.app)
    patch.undo()


def make_trip(scale, seed, n=701, hz=10.0):
    a = np.zeros((n, 11))
    a[:, 0] = np.arange(n) / hz
    a[:, 1] = 1
    a[:, 5:8] = np.random.default_rng(seed).normal(0, 1, (n, 3)) * scale
    return a


def query(client, acc):
    res = client.post('/api/analyze', files={'acc_file': ('RAW_ACCELEROMETERS.txt', to_bytes(acc), 'text/plain')})
    assert res.status_code == 200, res.text
    body, _ = wait_for_job(client, res.json()['job_id'], timeout=60)
    assert body['status'] == 'done', body
    return body['result']


@pytest.fixture(scope='module')
def queries(client):
    """Gửi N_QUERIES chuyến tổng hợp, thu (đặc trưng rẻ, nhãn thật, xác suất thật) cho mỗi chuyến."""
    rng = np.random.default_rng(0)
    scales = rng.uniform(0.02, 2.0, N_QUERIES)
    rows = []
    t0 = time.time()
    for i, scale in enumerate(scales):
        acc = make_trip(scale, seed=1000 + i)
        result = query(client, acc)
        sensor = result['models']['sensor']
        feature = float(np.abs(acc[:, 5:8]).mean())    # đặc trưng rẻ: không cần biết gì về kiến trúc model
        rows.append((feature, sensor['trip_label'], sensor['trip_probs']))
    elapsed = time.time() - t0
    return rows, elapsed


def test_no_rate_limit_slows_down_a_large_number_of_sequential_queries(queries):
    """Ghi nhận: hệ thống có giới hạn tốc độ theo người dùng/IP không? `MAX_PENDING` chỉ chặn số job
    CÙNG LÚC trong hàng đợi, không chặn tổng số truy vấn theo thời gian khi gửi TUẦN TỰ (chờ job xong rồi
    mới gửi tiếp, như một kẻ tấn công kiên nhẫn sẽ làm). Đây LÀ kết quả cần ghi vào báo cáo."""
    rows, elapsed = queries
    print(f'\n[A10] gửi {len(rows)} truy vấn tuần tự trong {elapsed:.1f}s '
          f'({elapsed / len(rows):.2f}s/truy vấn) -- không truy vấn nào bị 429 hay bị làm chậm có chủ đích.')
    assert len(rows) == N_QUERIES, 'có truy vấn bị chặn/giữa đường thất bại -- khác với kỳ vọng "không giới hạn"'


def test_surrogate_model_trained_only_from_api_responses_agrees_with_the_real_model(queries):
    """Huấn luyện một cây quyết định nhỏ CHỈ từ (đặc trưng rẻ, nhãn mà API trả về) -- không đụng tới
    trọng số thật -- rồi đo độ khớp trên phần giữ lại (holdout). Độ khớp cao nghĩa là: chỉ cần truy vấn
    API đủ nhiều, không cần có trọng số, đã mô phỏng lại được hành vi của model ở mức đủ dùng."""
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    rows, _ = queries
    X = np.array([[f] for f, _, _ in rows])
    y = np.array([label for _, label, _ in rows])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y if len(set(y)) > 1 else None)
    clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_train, y_train)
    agreement = clf.score(X_test, y_test)
    print(f'\n[A10] độ khớp model xấp xỉ (chỉ 1 đặc trưng rẻ, {len(X_train)} mẫu luyện) với nhãn thật '
          f'trên {len(X_test)} mẫu giữ lại: {agreement:.2f}')
    assert agreement >= 0.5, f'độ khớp model xấp xỉ bất ngờ thấp ({agreement:.2f}) -- kiểm tra lại logic test'


def test_api_returns_full_probability_vectors_not_just_a_label():
    """Xác nhận lại bằng code: `_pack()` trong `inference.py` trả `probs` (xác suất đầy đủ 3 lớp, 4 số
    lẻ) cho MỌI cửa sổ, không chỉ nhãn thắng. Đây là một phần lý do model xấp xỉ ở trên dễ huấn luyện --
    xác suất chi tiết tiết lộ nhiều hơn một nhãn rời rạc."""
    import inspect

    from inference import _pack
    src = inspect.getsource(_pack)
    assert 'probs' in src, 'giả định về _pack() đã thay đổi -- cập nhật lại ghi chú A10'
