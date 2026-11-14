"""T07. A fake Hugging Face server: weights are downloaded into the volume once (Mock, System).

The container is told to use a fake Hub (HF_ENDPOINT) running on this machine, with a fake token. No real token,
no 1.6 GB download, no Internet. The fake "weights" are a few bytes, so the model cannot be loaded: the web app must
still start and explain it.
"""
import socket

import pytest
from docker_helpers import docker
from fake_hf_server import FAKE_TOKEN, FILES, FakeHuggingFace

WEIGHTS = '/home/user/app/weights'


def free_port():
    with socket.socket() as s:
        s.bind(('', 0))
        return s.getsockname()[1]


@pytest.fixture
def hub():
    with FakeHuggingFace(free_port()) as fake:
        yield fake


def hub_env(hub, token=FAKE_TOKEN):
    return ['--add-host=host.docker.internal:host-gateway',
            '-e', f'HF_ENDPOINT=http://host.docker.internal:{hub.server.server_address[1]}',
            '-e', f'HF_TOKEN={token}']


def test_the_first_start_downloads_the_four_files_into_the_volume(start_container, volume, hub):
    c = start_container(*hub_env(hub), '-v', f'{volume()}:{WEIGHTS}')
    c.wait_web(timeout=180)
    assert hub.downloads() == len(FILES)
    for name, data in FILES.items():
        assert c.exec('cat', f'{WEIGHTS}/{name}') == data.decode()
    # the library also pings /api/agent-harnesses without a token; every model request must carry it
    model_requests = [r for r in hub.requests if '/api/agent-harnesses' not in r[1]]
    assert model_requests and all(ok for _, _, ok in model_requests), 'every model request must carry the token'


def test_fake_weights_leave_the_web_up_with_a_clear_message_and_no_token_in_the_log(start_container, volume, hub):
    c = start_container(*hub_env(hub), '-v', f'{volume()}:{WEIGHTS}')
    c.wait_web(timeout=180)
    status = c.status()
    assert status['ready'] is False
    assert 'Không nạp được model' in status['message']
    assert FAKE_TOKEN not in c.logs()
    assert FAKE_TOKEN not in str(status)


def test_a_restart_uses_the_volume_and_does_not_download_again(start_container, volume, hub):
    c = start_container(*hub_env(hub), '-v', f'{volume()}:{WEIGHTS}')
    c.wait_web(timeout=180)
    assert hub.downloads() == len(FILES)
    weight_requests = lambda: [r for r in hub.requests if 'checking-driver-weights' in r[1]]   # the demo repo is separate
    after_first_start = len(weight_requests())
    docker('restart', c.name)
    c.wait_web(timeout=180)
    assert hub.downloads() == len(FILES)
    # with all four files already in the volume the app must not even ask the Hub about the weights
    assert len(weight_requests()) == after_first_start


def test_a_wrong_token_is_refused_and_the_web_still_starts(start_container, volume, hub):
    c = start_container(*hub_env(hub, token='hf_WRONGtokenWRONGtoken0123456789'), '-v', f'{volume()}:{WEIGHTS}')
    c.wait_web(timeout=180)
    assert hub.downloads() == 0
    assert c.status()['ready'] is False
    assert 'Chưa có model' in c.status()['message']
    log = c.logs()
    assert '[weights]' in log
    assert 'hf_WRONGtoken' not in log
    assert c.exec('sh', '-c', f'ls {WEIGHTS}/*.pth 2>/dev/null | wc -l') == '0'
