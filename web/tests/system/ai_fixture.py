"""Fixture for the AI system tests: the real image with the real weights (mounted read-only, no token needed).

Needs CD_IMAGE, CD_WEIGHTS_DIR (folder with the four weight files) and CD_DEMO_DIR (the demo trip folder).
It starts the image, runs the demo trip once and measures start-up time, analysis time and peak memory.
"""
import json
import os
import re
import time
import urllib.request
import uuid

import pytest
from docker_helpers import IMAGE, Container, docker, http_get, wait_until

WEIGHTS_DIR = os.environ.get('CD_WEIGHTS_DIR')
DEMO_DIR = os.environ.get('CD_DEMO_DIR')
UNITS_IN_GIB = {'KiB': 1024 ** 2, 'MiB': 1024, 'GiB': 1, 'KB': 1e6, 'MB': 1000, 'GB': 1}


def to_gib(text):
    m = re.match(r'([\d.]+)\s*([KMG]i?B)', text)
    return float(m.group(1)) / UNITS_IN_GIB[m.group(2)]


@pytest.fixture(scope='session')
def demo_run():
    if not (IMAGE and WEIGHTS_DIR and DEMO_DIR):
        pytest.skip('set CD_IMAGE, CD_WEIGHTS_DIR and CD_DEMO_DIR to run the AI system tests')
    name = 'cdai-' + uuid.uuid4().hex[:8]
    docker('run', '-d', '--name', name, '-p', '127.0.0.1::7860',
           '-v', f'{WEIGHTS_DIR}:/home/user/app/weights:ro', '-v', f'{DEMO_DIR}:/home/user/app/demo:ro', IMAGE)
    c = Container(name)
    try:
        t0 = time.time()
        try:
            wait_until(lambda: http_get(c.url + '/api/status')[0] == 200, 300, what='web up')
        except AssertionError as e:
            raise AssertionError(f"{e}; state: {docker('inspect', '--format', '{{json .State}}', name, check=False)}; "
                                 f"log tail: {c.logs()[-800:]}") from None
        startup = time.time() - t0
        status = c.status()
        req = urllib.request.Request(c.url + '/api/demo', method='POST')
        job_id = json.loads(urllib.request.urlopen(req, timeout=30).read())['job_id']
        t1, peak, body = time.time(), 0.0, {}
        while time.time() - t1 < 600:
            body = json.loads(http_get(f'{c.url}/api/job/{job_id}')[1])
            usage = docker('stats', '--no-stream', '--format', '{{.MemUsage}}', name).split('/')[0].strip()
            peak = max(peak, to_gib(usage))
            if body['status'] in ('done', 'error'):
                break
            time.sleep(1.0)
        yield dict(container=c, startup=startup, analysis=time.time() - t1, peak_gib=peak, status=status, job=body,
                   result=body.get('result'))
    finally:
        docker('rm', '-f', name, check=False)
