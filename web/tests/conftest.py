"""Shared test setup.

Tests must never depend on the 1.6 GB model weights or on the network, so the app is pointed at
empty folders and HF_TOKEN is removed BEFORE `app` is imported (app.py downloads weights at import time).
"""
import os
import sys
import tempfile

WEB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WEB_DIR not in sys.path:
    sys.path.insert(0, WEB_DIR)

_EMPTY = tempfile.mkdtemp(prefix='cd_test_')
os.environ['WEIGHTS_DIR'] = os.path.join(_EMPTY, 'weights')
os.environ['DEMO_DIR'] = os.path.join(_EMPTY, 'demo')
os.environ.pop('HF_TOKEN', None)


# ---- shared fixtures for the tests that need a "model" without any weights ----------------------------------
import threading  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402


class FakeModel:
    """Stands in for the three networks. `pattern` is a list of [normal, aggressive, drowsy] probabilities that is
    cycled over the valid windows, so a test can decide exactly what the "model" says and compute the expected
    result by hand."""

    def __init__(self):
        self.pattern = [[0.1, 0.8, 0.1]]
        self.calls = 0


@pytest.fixture
def fake_model(monkeypatch):
    """Pretend a sensor model is loaded. Only the neural network call is replaced; windowing, summary, events and
    the API run for real."""
    import numpy as np

    import inference
    fake = FakeModel()

    def predict(self, x):
        fake.calls += 1
        return np.array([fake.pattern[i % len(fake.pattern)] for i in range(len(x))], dtype=np.float64)

    monkeypatch.setattr(inference.Analyzer, 'ready', property(lambda self: True))
    monkeypatch.setattr(inference.Analyzer, '_kine', lambda self, sens, idx: np.zeros((len(idx), 3, 51), np.float32))
    monkeypatch.setattr(inference.Analyzer, '_predict_1d', predict)
    return fake


@pytest.fixture(autouse=True)
def clean_jobs():
    """Every test starts and ends with an empty job table so the queue limit never leaks between tests."""
    import app as web
    web.JOBS.clear()
    yield
    deadline = time.time() + 10
    while time.time() < deadline and any(v['status'] in ('queued', 'running') for v in web.JOBS.values()):
        time.sleep(0.02)
    web.JOBS.clear()


@pytest.fixture
def gate(monkeypatch):
    """Replace the analyzer's analyze() with one that blocks until the test opens the gate (deterministic, no sleeps)."""
    import app as web
    ev = threading.Event()
    state = {'started': 0}

    def slow(sens, video=None, demo=False, progress=None, gpu_fn=None):
        state['started'] += 1
        ev.wait(timeout=15)
        return {'demo': False, 'trip': {}, 'models': {}, 'windows': [], 'thumbs': [], 'video': {}, 'series': {}, 'cleaning': {}}

    monkeypatch.setattr(web.Analyzer, 'ready', property(lambda self: True))
    monkeypatch.setattr(web.analyzer, 'analyze', slow)
    ev.state = state
    yield ev
    ev.set()
