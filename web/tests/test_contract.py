"""T01. Contract of every HTTP endpoint (Automation, API).

Each endpoint has a row in CONTRACT. The first test fails when a route is added to the app without a row, so a new
endpoint cannot ship untested. The other tests check the status code, content type and JSON fields that the web page
and users rely on.
"""
import re

import pytest
from fastapi.testclient import TestClient

import app as web

# (method, path, status without a model, content-type prefix)
CONTRACT = [
    ('GET', '/', 200, 'text/html'),
    ('GET', '/api/status', 200, 'application/json'),
    ('POST', '/api/analyze', 422, 'application/json'),     # no file sent
    ('POST', '/api/demo', 503, 'application/json'),        # no model loaded
    ('GET', '/api/job/{jid}', 404, 'application/json'),    # unknown job
]
STATUS_FIELDS = {'ready', 'video', 'fusion', 'demo', 'message', 'gpu', 'limits'}


@pytest.fixture(scope='module')
def client():
    return TestClient(web.app)


def _app_routes(client):
    spec = client.get('/openapi.json').json()
    return {(m.upper(), p) for p, ops in spec['paths'].items() for m in ops}


def test_every_route_of_the_app_has_a_contract_row(client):
    documented = {(m, p) for m, p, _, _ in CONTRACT}
    # "/" is a FileResponse route that FastAPI may leave out of the schema, so compare only the API routes
    api = {(m, p) for m, p in _app_routes(client) if p != '/'}
    assert api == {(m, p) for m, p in documented if p != '/'}


@pytest.mark.parametrize('method,path,status,ctype', CONTRACT)
def test_endpoint_status_and_content_type(client, method, path, status, ctype):
    r = client.request(method, path.replace('{jid}', 'no-such-job'))
    assert r.status_code == status
    assert r.headers['content-type'].startswith(ctype)


@pytest.mark.parametrize('method,path,status,ctype', [c for c in CONTRACT if c[2] >= 400])
def test_errors_carry_a_human_readable_detail(client, method, path, status, ctype):
    detail = client.request(method, path.replace('{jid}', 'no-such-job')).json()['detail']
    assert detail, 'an error without a message cannot be shown to the user'


def test_status_has_every_field_the_page_reads(client):
    body = client.get('/api/status').json()
    assert STATUS_FIELDS <= set(body)
    assert isinstance(body['ready'], bool) and isinstance(body['message'], str)
    assert set(body['limits']) == {'text_mb', 'video_mb'}


def test_wrong_method_is_405_not_a_crash(client):
    assert client.get('/api/analyze').status_code == 405
    assert client.post('/api/status').status_code == 405
    assert client.delete('/api/job/x').status_code == 405


def test_unknown_path_is_404(client):
    assert client.get('/api/nothing-here').status_code == 404


def test_a_successful_submit_returns_a_job_id(client, fake_model):
    from synthetic import upload_files
    r = client.post('/api/analyze', files=upload_files())
    assert r.status_code == 200
    assert re.fullmatch(r'[0-9a-f]{32}', r.json()['job_id'])
