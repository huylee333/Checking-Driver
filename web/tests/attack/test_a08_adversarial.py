"""A08. Mẫu đối nghịch, evasion (Model AI).

Dùng model thật (1D Swin, nhánh cảm biến), chạy trực tiếp trong tiến trình (giống T09, không cần Docker)
với trọng số thật trên máy -- không gửi gì ra ngoài Internet. Thử FGSM (Fast Gradient Sign Method): với
một đoạn mà model đang chấm "AGGRESSIVE" tin cậy cao, cộng vào tín hiệu một nhiễu nhỏ THEO HƯỚNG GRADIENT
(không phải nhiễu ngẫu nhiên như T09) để tìm biên độ nhiễu NHỎ NHẤT làm model đổi nhãn.

Đây là phép đo độ bền, không có "đạt/không đạt" tuyệt đối -- xem CHECKPOINT_TEST_VA_TAN_CONG.md mục A08.
"""
import os

import numpy as np
import pytest
import torch

from inference import KINE_LEN, Analyzer, load_sensors

WEIGHTS_DIR = os.environ.get('CD_WEIGHTS_DIR')
DEMO_DIR = os.environ.get('CD_DEMO_DIR')
pytestmark = pytest.mark.skipif(not (WEIGHTS_DIR and DEMO_DIR), reason='set CD_WEIGHTS_DIR and CD_DEMO_DIR')


@pytest.fixture(scope='module')
def analyzer():
    a = Analyzer(WEIGHTS_DIR)
    assert a.ready, a.error
    return a


@pytest.fixture(scope='module')
def aggressive_window(analyzer):
    """Cửa sổ đầu tiên của chuyến mẫu mà model cảm biến chấm AGGRESSIVE với xác suất >= 0.9."""
    acc = np.loadtxt(os.path.join(DEMO_DIR, 'RAW_ACCELEROMETERS.txt'))
    sens = load_sensors(open(os.path.join(DEMO_DIR, 'RAW_ACCELEROMETERS.txt'), 'rb').read())
    valid_idx = [i for i, w in enumerate(sens['windows']) if w['valid']]
    kine = analyzer._kine(sens, valid_idx)                           # (N, 3, 51) đã chuẩn hoá
    probs = analyzer._predict_1d(kine)
    agg = np.where((np.argmax(probs, axis=1) == 1) & (probs[:, 1] >= 0.9))[0]
    assert len(agg), 'không có cửa sổ AGGRESSIVE tin cậy cao nào trong chuyến mẫu để thử'
    i = agg[0]
    return torch.from_numpy(kine[i:i + 1]), float(probs[i, 1]), analyzer.std


def _softmax3(logits):
    return torch.softmax(logits[:, :3].float(), dim=1)


def fgsm(model, x0, true_class, epsilon):
    """Một bước FGSM: x0 + epsilon * sign(grad of CE loss wrt x, tăng loss của true_class)."""
    x = x0.clone().requires_grad_(True)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits[:, :3], torch.tensor([true_class]))
    loss.backward()
    with torch.no_grad():
        return (x + epsilon * x.grad.sign()).detach()


def test_fgsm_minimum_perturbation_that_flips_the_label(analyzer, aggressive_window):
    """Tăng dần epsilon (đơn vị: độ lệch chuẩn đã chuẩn hoá của kênh) tới khi nhãn đổi khỏi AGGRESSIVE.
    Ghi lại epsilon nhỏ nhất và tỉ lệ so với biên độ tín hiệu gốc -- đây LÀ kết quả cần báo cáo, không
    phải một ngưỡng pass/fail."""
    x0, p0, _ = aggressive_window
    model = analyzer.model_1d.eval()
    true_class = 1  # AGGRESSIVE

    flip_eps = None
    for eps in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0):
        x_adv = fgsm(model, x0, true_class, eps)
        with torch.no_grad():
            p = _softmax3(model(x_adv))[0]
        label = int(torch.argmax(p))
        if label != true_class:
            flip_eps = eps
            flip_label, flip_prob = label, float(p[label])
            break

    rel = (flip_eps / x0.abs().mean().item()) if flip_eps else None
    print(f'\n[A08] xác suất AGGRESSIVE gốc: {p0:.3f}; epsilon nhỏ nhất làm đổi nhãn: {flip_eps}; '
          f'tỉ lệ so với biên độ tín hiệu trung bình: {rel}')

    # Đây KHÔNG phải một ngưỡng an toàn tuyệt đối -- chỉ là ghi nhận: model lật nhãn ở mức nhiễu nào.
    # Bất kỳ epsilon hữu hạn nào lật được nhãn đều đáng ghi vào báo cáo (xem CHECKPOINT mục A08).
    assert flip_eps is not None, 'không lật được nhãn dù eps lên tới 2.0 (đáng mừng, nhưng hiếm, nên kiểm tra lại logic test)'


def test_fgsm_perturbation_is_detectable_as_out_of_distribution(analyzer, aggressive_window):
    """Nhiễu đủ để lật nhãn có "giống dữ liệu cảm biến thật" không, hay đã rõ ràng bất thường (biên độ
    chuẩn hoá vượt xa [-6, 6], tức hơn 6 độ lệch chuẩn của tập train)? Nếu nhiễu nhỏ mà vẫn trong khoảng
    bình thường -> đáng lo hơn (khó phát hiện); nếu phải nhiễu lớn, bất thường rõ -> một bộ lọc ngoại lai
    đơn giản (chặn |x| chuẩn hoá > ngưỡng) có thể giảm nhẹ rủi ro này."""
    x0, _, _ = aggressive_window
    model = analyzer.model_1d.eval()
    x_adv = fgsm(model, x0, 1, 0.5)                                   # eps=0.5: đủ lớn để thường lật được
    with torch.no_grad():
        label = int(torch.argmax(_softmax3(model(x_adv))[0]))
    max_abs = float(x_adv.abs().max())
    print(f'\n[A08] |x| lớn nhất sau nhiễu (đơn vị độ lệch chuẩn): {max_abs:.2f}; nhãn sau nhiễu: {label}')
    # Chỉ ghi nhận số liệu để đưa vào báo cáo -- không assert ngưỡng cụ thể (phụ thuộc chuyến mẫu).
