"""Fixtures for the attack tests (A01-A10).

A01, A03-A07 run the REAL image in Docker with the REAL weights mounted read-only (no HF_TOKEN needed),
the same way web/tests/system/ai_fixture.py does for T08-T10 -- they request `ready_container`,
`start_limited_container` or `demo_run` below, each of which calls `pytest.skip()` on its own when
CD_IMAGE/CD_WEIGHTS_DIR/CD_DEMO_DIR are missing, for example:
    CD_IMAGE=checking-driver:ci CD_WEIGHTS_DIR=...\\weights CD_DEMO_DIR=...\\demo pytest web/tests/attack -q

A02, A08-A10 need only the real weights (no Docker, no CD_IMAGE): they run the model directly in this
process, like web/tests/ai/test_model_properties.py, and skip themselves via their own module-level
`pytestmark` when CD_WEIGHTS_DIR/CD_DEMO_DIR are missing. There is deliberately no blanket
"skip everything unless CD_IMAGE is set" here, so those do not get skipped for the wrong reason.

Safety: every test here only talks to a container started by this fixture (127.0.0.1, a port Docker picked)
or to the model in-process, using synthetic or locally-generated files. Nothing is sent to Docker Hub,
GitHub, Hugging Face or any other host, and no resource limit is removed -- some tests set
--memory/--cpus to make a crash cheap to reproduce and cheap to clean up.
"""
import os
import sys
import uuid

import pytest

SYSTEM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'system')
if SYSTEM_DIR not in sys.path:
    sys.path.insert(0, SYSTEM_DIR)
ATTACK_DIR = os.path.dirname(os.path.abspath(__file__))
if ATTACK_DIR not in sys.path:
    sys.path.insert(0, ATTACK_DIR)

from ai_fixture import demo_run  # noqa: F401,E402  (session fixture: real weights + demo trip, already running)
from docker_helpers import IMAGE, Container, docker  # noqa: E402

WEIGHTS_DIR = os.environ.get('CD_WEIGHTS_DIR')


@pytest.fixture(autouse=True)
def clean_jobs():
    """Overrides the fixture of the same name in tests/conftest.py: attack tests never import the app in-process."""
    yield


@pytest.fixture(scope='session')
def ready_container():
    """One container shared by every attack test that only needs to send a few harmless-looking requests
    (real weights mounted read-only, so /api/analyze is reachable; no demo trip needed). Started once per
    session and removed at the end: these tests never change server-side state across requests (the job
    table is per-request and short-lived), so sharing it is safe and ~10x faster than one container each."""
    if not (IMAGE and WEIGHTS_DIR):
        pytest.skip('set CD_IMAGE and CD_WEIGHTS_DIR to run this attack test')
    name = 'cdatk-shared-' + uuid.uuid4().hex[:8]
    docker('run', '-d', '--name', name, '-p', '127.0.0.1::7860', '-v', f'{WEIGHTS_DIR}:/home/user/app/weights:ro', IMAGE)
    c = Container(name)
    c.wait_web(180)
    try:
        yield c
    finally:
        docker('rm', '-f', name, check=False)


@pytest.fixture
def start_limited_container():
    """start_limited_container(mem='1g', cpus=2) -> Container, real weights mounted, resource-limited.
    For the denial-of-service test: a small, explicit limit makes the crash point reproducible."""
    if not (IMAGE and WEIGHTS_DIR):
        pytest.skip('set CD_IMAGE and CD_WEIGHTS_DIR to run this attack test')
    started = []

    def start(mem='1g', cpus=2):
        name = 'cdatk-' + uuid.uuid4().hex[:8]
        docker('run', '-d', '--name', name, '--memory', mem, '--cpus', str(cpus), '-p', '127.0.0.1::7860',
               '-v', f'{WEIGHTS_DIR}:/home/user/app/weights:ro', IMAGE)
        started.append(name)
        c = Container(name)
        c.wait_web(180)
        return c

    yield start
    for name in started:
        docker('rm', '-f', name, check=False)
