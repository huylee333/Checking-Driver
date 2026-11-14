"""T02. Life of one job, from upload to result (Automation, API).

Only the neural network call is faked (see `fake_model` in conftest.py). Upload, validation, queue, worker thread,
windowing, summary and the JSON the web page receives all run for real.
"""
import pytest
from fastapi.testclient import TestClient

import app as web
from inference import InputError
from synthetic import upload_files, wait_for_job

RESULT_KEYS = {'demo', 'trip', 'models', 'windows', 'thumbs', 'video', 'series', 'cleaning'}


@pytest.fixture(scope='module')
def client():
    return TestClient(web.app)


def test_a_trip_goes_queued_running_done_with_a_complete_result(client, fake_model):
    job_id = client.post('/api/analyze', files=upload_files()).json()['job_id']
    body, seen = wait_for_job(client, job_id)
    assert body['status'] == 'done'
    assert seen[-1] == 'done' and set(seen) <= {'queued', 'running', 'done'}
    result = body['result']
    assert RESULT_KEYS <= set(result)
    assert result['trip']['windows_valid'] > 0
    assert result['models']['sensor'] is not None
    assert result['models']['video'] is None and result['models']['fusion'] is None
    assert result['video']['status'] == 'none'
    assert result['cleaning']['acc']['rows_raw'] == 701


def test_an_input_error_inside_the_worker_is_shown_to_the_user(client, fake_model, monkeypatch):
    def reject(*a, **k):
        raise InputError('Chuyến đi này không phân tích được.')
    monkeypatch.setattr(web.analyzer, 'analyze', reject)
    job_id = client.post('/api/analyze', files=upload_files()).json()['job_id']
    body, _ = wait_for_job(client, job_id)
    assert body['status'] == 'error'
    assert body['error'] == 'Chuyến đi này không phân tích được.'


def test_an_unexpected_crash_is_reported_generically_and_hides_the_detail(client, fake_model, monkeypatch, capsys):
    def crash(*a, **k):
        raise RuntimeError('secret internal detail C:\\srv\\app\\weights')
    monkeypatch.setattr(web.analyzer, 'analyze', crash)
    job_id = client.post('/api/analyze', files=upload_files()).json()['job_id']
    body, _ = wait_for_job(client, job_id)
    assert body['status'] == 'error'
    assert 'secret internal detail' not in body['error']
    assert 'Có lỗi' in body['error']
    assert 'secret internal detail' in capsys.readouterr().out      # the detail goes to the server log only


def test_the_worker_survives_a_crash_and_serves_the_next_job(client, fake_model, monkeypatch):
    real = web.analyzer.analyze

    def boom(*a, **k):
        raise RuntimeError('boom')
    monkeypatch.setattr(web.analyzer, 'analyze', boom)
    first = client.post('/api/analyze', files=upload_files()).json()['job_id']
    assert wait_for_job(client, first)[0]['status'] == 'error'
    monkeypatch.setattr(web.analyzer, 'analyze', real)
    second = client.post('/api/analyze', files=upload_files()).json()['job_id']
    assert wait_for_job(client, second)[0]['status'] == 'done'


def test_a_finished_job_is_forgotten_after_its_time_to_live(client, fake_model):
    old = client.post('/api/analyze', files=upload_files()).json()['job_id']
    wait_for_job(client, old)
    web.JOBS[old]['created'] -= web.JOB_TTL_SEC + 60            # pretend it finished long ago
    client.post('/api/analyze', files=upload_files())             # any new submit runs the cleanup
    r = client.get(f'/api/job/{old}')
    assert r.status_code == 404
    assert 'hết hạn' in r.json()['detail']


def test_the_uploaded_data_is_dropped_from_memory_when_the_job_ends(client, fake_model):
    job_id = client.post('/api/analyze', files=upload_files()).json()['job_id']
    wait_for_job(client, job_id)
    assert 'sens' not in web.JOBS[job_id]
