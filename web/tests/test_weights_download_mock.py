"""T06. Hugging Face fails while the app starts (Mock, API).

`huggingface_hub.snapshot_download` is replaced by a fake that fails the way the real service can fail: wrong token,
repo not found, timeout, half-written file. In every case the web app must still start and say what is wrong in
words, and the token must never appear in the log or in an answer.
"""
import json
import os
import urllib.error

import pytest
from fastapi.testclient import TestClient

import app as web
from inference import Analyzer

FAKE_TOKEN = 'hf_FAKEtokenFAKEtokenFAKEtoken0123456789'


class FakeHub:
    """Records what the app asked for and raises the configured error."""

    def __init__(self):
        self.error = None
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        if self.error is not None:
            raise self.error
        return kw.get('local_dir')


@pytest.fixture
def hub(monkeypatch, tmp_path):
    import huggingface_hub
    fake = FakeHub()
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', fake)
    monkeypatch.setattr(web, 'WEIGHTS_DIR', str(tmp_path / 'weights'))
    monkeypatch.setattr(web, 'DEMO_DIR', str(tmp_path / 'demo'))
    monkeypatch.setenv('HF_TOKEN', FAKE_TOKEN)
    return fake


# Standard-library exceptions only: the CI environment does not have `requests`, and the app catches any Exception.
URL = 'https://huggingface.co/api/models/x'
ERRORS = {
    'wrong token (401)': urllib.error.HTTPError(URL, 401, 'Unauthorized', {}, None),
    'repo not found (404)': urllib.error.HTTPError(URL, 404, 'Repository Not Found', {}, None),
    'timeout': TimeoutError('Connection to huggingface.co timed out'),
    'no network': ConnectionError('Failed to resolve huggingface.co'),
    'disk full': OSError(28, 'No space left on device'),
}


@pytest.mark.parametrize('name', ERRORS)
def test_a_download_failure_never_stops_the_app_and_never_prints_the_token(hub, capsys, name):
    hub.error = ERRORS[name]
    web._fetch_weights()                    # must not raise
    web._fetch_demo()
    out = capsys.readouterr().out
    assert '[weights]' in out and '[demo]' in out
    assert FAKE_TOKEN not in out


def test_the_token_is_sent_to_hugging_face_and_only_the_four_weight_files_are_requested(hub):
    web._fetch_weights()
    assert len(hub.calls) == 1
    call = hub.calls[0]
    assert call['token'] == FAKE_TOKEN
    assert call['repo_type'] == 'model'
    assert call['allow_patterns'] == web.WEIGHT_FILES


def test_without_a_token_nothing_is_downloaded_and_the_reason_is_logged(hub, monkeypatch, capsys):
    monkeypatch.delenv('HF_TOKEN')
    web._fetch_weights()
    web._fetch_demo()
    assert hub.calls == []
    assert 'HF_TOKEN' in capsys.readouterr().out


def test_complete_weights_on_disk_are_not_downloaded_again(hub):
    os.makedirs(web.WEIGHTS_DIR)
    for f in web.WEIGHT_FILES:
        open(os.path.join(web.WEIGHTS_DIR, f), 'wb').close()
    web._fetch_weights()
    assert hub.calls == []


def test_status_without_weights_is_ready_false_with_a_readable_reason(tmp_path):
    a = Analyzer(str(tmp_path))
    s = a.status()
    assert s['ready'] is False and s['video'] is False and s['fusion'] is False
    assert 'Chưa có model' in s['message']


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(data)


@pytest.mark.parametrize('bad', ['stats_not_json', 'weights_truncated', 'weights_empty'])
def test_corrupt_weight_files_leave_the_app_up_with_ready_false(tmp_path, bad):
    """A half-downloaded file must not crash the import of app.py (then the container would restart forever)."""
    stats = json.dumps({'mean': [0, 0, 0], 'std': [1, 1, 1]}).encode()
    _write(str(tmp_path / 'kine_norm_stats.json'), b'{not json' if bad == 'stats_not_json' else stats)
    _write(str(tmp_path / 'final_1d.pth'), b'' if bad == 'weights_empty' else b'PK\x03\x04 truncated')
    a = Analyzer(str(tmp_path))                  # must not raise
    s = a.status()
    assert s['ready'] is False
    assert s['message'] and s['message'] != 'Sẵn sàng'


def test_the_api_of_an_app_without_a_model_still_answers(tmp_path):
    client = TestClient(web.app)
    assert client.get('/api/status').json()['ready'] is False
    assert client.get('/').status_code == 200
