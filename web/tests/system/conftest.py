"""Fixtures for the system tests: they run the REAL image in Docker.

They are skipped unless the environment variable CD_IMAGE names an image, for example:
    CD_IMAGE=checking-driver:ci pytest web/tests/system -q
"""
import uuid

import pytest

from ai_fixture import demo_run  # noqa: F401  (session fixture shared by the AI system tests)
from docker_helpers import IMAGE, Container, docker


def pytest_collection_modifyitems(config, items):
    if IMAGE:
        return
    skip = pytest.mark.skip(reason='set CD_IMAGE to the image name to run the system tests')
    for item in items:
        if 'tests/system' in item.nodeid.replace(chr(92), '/'):
            item.add_marker(skip)


@pytest.fixture
def start_container():
    """start_container(*docker_run_args) -> Container. Every container is removed at the end of the test."""
    started = []

    def start(*args):
        name = 'cdtest-' + uuid.uuid4().hex[:8]
        docker('run', '-d', '--name', name, '-p', '127.0.0.1::7860', *args, IMAGE)
        started.append(name)
        return Container(name)

    yield start
    for name in started:
        docker('rm', '-f', name, check=False)


@pytest.fixture
def volume():
    """A new named volume, deleted at the end of the test."""
    names = []

    def make():
        name = 'cdtest-vol-' + uuid.uuid4().hex[:8]
        docker('volume', 'create', name)
        names.append(name)
        return name

    yield make
    for name in names:
        docker('volume', 'rm', '-f', name, check=False)


@pytest.fixture(autouse=True)
def clean_jobs():
    """Overrides the fixture of the same name in tests/conftest.py: system tests never import the app in-process."""
    yield
