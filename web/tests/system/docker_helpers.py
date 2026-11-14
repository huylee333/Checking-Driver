"""Helpers for the system tests (they drive the real image through the docker command line)."""
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

IMAGE = os.environ.get('CD_IMAGE')


def docker(*args, check=True, timeout=120):
    r = subprocess.run(['docker', *args], capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise AssertionError(f'docker {" ".join(args)} failed: {r.stderr.strip()[:500]}')
    return r.stdout.strip()


def http_get(url, timeout=5):
    """(status, body bytes). Never raises on an HTTP error status."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception:
        return 0, b''


def wait_until(cond, timeout, interval=1.0, what='condition'):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = cond()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError(f'{what} not reached within {timeout}s')


class Container:
    def __init__(self, name):
        self.name = name

    def host_port(self):
        # the port mapping appears a moment after `docker run -d` returns
        out = wait_until(lambda: docker('port', self.name, '7860/tcp', check=False), 30, interval=0.5, what='port mapping')
        return int(out.splitlines()[0].rsplit(':', 1)[1])

    @property
    def url(self):
        return f'http://127.0.0.1:{self.host_port()}'

    def wait_web(self, timeout=120):
        return wait_until(lambda: http_get(self.url + '/')[0] == 200, timeout, what='home page 200')

    def status(self):
        return json.loads(http_get(self.url + '/api/status')[1])

    def exec(self, *cmd, check=True):
        return docker('exec', self.name, *cmd, check=check)

    def logs(self):
        r = subprocess.run(['docker', 'logs', self.name], capture_output=True, text=True)
        return r.stdout + r.stderr

    def inspect(self, fmt):
        return docker('inspect', '--format', fmt, self.name)
