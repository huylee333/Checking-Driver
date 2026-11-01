"""HTTP API of the web app (app.py), run without model weights.

With no weights the app must still start and answer, reporting that no model is available.
"""
import pytest
from fastapi.testclient import TestClient

import app as web


@pytest.fixture(scope='module')
def client():
    return TestClient(web.app)


@pytest.fixture
def ready(monkeypatch):
    """Pretend the model is loaded so the upload validation (400 errors) can be exercised."""
    monkeypatch.setattr(web.Analyzer, 'ready', property(lambda self: True))


def test_home_page_is_served(client):
    r = client.get('/')
    assert r.status_code == 200
    assert 'text/html' in r.headers['content-type']


def test_status_reports_no_model_instead_of_crashing(client):
    r = client.get('/api/status')
    assert r.status_code == 200
    body = r.json()
    assert body['ready'] is False
    assert body['video'] is False and body['fusion'] is False
    assert body['limits'] == {'text_mb': 20, 'video_mb': 300}


def test_demo_is_unavailable_without_model(client):
    assert client.post('/api/demo').status_code == 503


def test_analyze_without_a_file_is_a_validation_error(client):
    assert client.post('/api/analyze').status_code == 422


def test_analyze_without_model_returns_503(client):
    files = {'acc_file': ('RAW_ACCELEROMETERS.txt', b'1 2 3\n', 'text/plain')}
    r = client.post('/api/analyze', files=files)
    assert r.status_code == 503
    assert 'chưa có model' in r.json()['detail']


def test_unknown_job_is_404(client):
    assert client.get('/api/job/does-not-exist').status_code == 404


def test_oversized_upload_is_rejected_before_it_is_read(client, monkeypatch):
    monkeypatch.setattr(web, 'TOTAL_MAX_BYTES', 10)
    files = {'acc_file': ('RAW_ACCELEROMETERS.txt', b'x' * 1000, 'text/plain')}
    assert client.post('/api/analyze', files=files).status_code == 413


def test_upload_with_a_non_txt_extension_is_rejected(client, ready):
    files = {'acc_file': ('data.exe', b'1 2 3\n', 'application/octet-stream')}
    r = client.post('/api/analyze', files=files)
    assert r.status_code == 400
    assert '.txt' in r.json()['detail']


def test_video_sent_in_the_sensor_slot_is_rejected(client, ready):
    files = {'acc_file': ('trip.mp4', b'\x00\x00', 'video/mp4')}
    r = client.post('/api/analyze', files=files)
    assert r.status_code == 400
    assert 'video' in r.json()['detail']


def test_malformed_sensor_file_gets_a_readable_error(client, ready):
    files = {'acc_file': ('RAW_ACCELEROMETERS.txt', b'not a table\n', 'text/plain')}
    r = client.post('/api/analyze', files=files)
    assert r.status_code == 400
    assert 'cột' in r.json()['detail']
