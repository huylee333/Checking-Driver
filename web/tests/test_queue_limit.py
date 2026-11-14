"""T04. The queue is full: the next request is refused, and the service recovers (Automation, API).

`gate` (conftest.py) makes the analysis block until the test releases it, so the test is deterministic and has no sleeps.
"""
import glob
import tempfile
import time

import pytest
from fastapi.testclient import TestClient

import app as web
from synthetic import upload_files, wait_for_job


@pytest.fixture(scope='module')
def client():
    return TestClient(web.app)


def _submit(client):
    return client.post('/api/analyze', files=upload_files())


def _wait_until(cond, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return
        time.sleep(0.01)
    raise AssertionError('condition not reached in time')


def test_the_request_after_max_pending_is_refused_with_429(client, gate):
    accepted = [_submit(client) for _ in range(web.MAX_PENDING)]
    assert [r.status_code for r in accepted] == [200] * web.MAX_PENDING
    refused = _submit(client)
    assert refused.status_code == 429
    assert 'bận' in refused.json()['detail']


def test_only_one_job_runs_the_others_wait_with_a_queue_position(client, gate):
    ids = [_submit(client).json()['job_id'] for _ in range(web.MAX_PENDING)]
    _wait_until(lambda: gate.state['started'] == 1)
    states = [client.get(f'/api/job/{j}').json() for j in ids]
    assert [s['status'] for s in states] == ['running', 'queued', 'queued']
    assert [s['queue_position'] for s in states[1:]] == [1, 2]
    assert gate.state['started'] == 1                           # one worker thread, so one analysis at a time


def test_after_the_queue_drains_new_requests_are_accepted_again(client, gate):
    ids = [_submit(client).json()['job_id'] for _ in range(web.MAX_PENDING)]
    assert _submit(client).status_code == 429
    gate.set()
    for j in ids:
        assert wait_for_job(client, j)[0]['status'] == 'done'
    assert _submit(client).status_code == 200


def test_a_refused_request_does_not_use_up_a_slot_or_leave_files_behind(client, gate):
    before = set(glob.glob(f'{tempfile.gettempdir()}/trip_*'))
    for _ in range(web.MAX_PENDING):
        _submit(client)
    for _ in range(5):
        assert _submit(client).status_code == 429
    assert len(web.JOBS) == web.MAX_PENDING
    assert set(glob.glob(f'{tempfile.gettempdir()}/trip_*')) == before


def test_the_limit_matches_the_documented_value():
    assert web.MAX_PENDING == 3
