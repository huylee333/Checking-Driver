"""Synthetic sensor trips for the tests. They are generated here, NOT taken from UAH-DriveSet.

Accelerometer file: 11 whitespace-separated columns.
  0 = timestamp (s), 1 = activation flag, 2-4 = raw acceleration, 5-7 = Kalman-filtered acceleration (the model input),
  8-10 = roll, pitch, yaw.
"""
import time

import numpy as np


def make_trip(seconds=70.0, hz=10.0, amp=0.1, flag=1, seed=0):
    """A valid trip: `seconds` long at `hz` Hz. A 51-sample window then spans exactly 5.0 s, as the model expects."""
    n = int(seconds * hz) + 1
    rng = np.random.default_rng(seed)
    a = np.zeros((n, 11))
    a[:, 0] = np.arange(n) / hz
    a[:, 1] = flag
    a[:, 2:] = rng.normal(0.0, amp, (n, 9))
    return a


def to_bytes(a):
    return ('\n'.join(' '.join(f'{x:.6f}' for x in row) for row in a) + '\n').encode()


def upload_files(trip=None):
    """Multipart `files` argument for POST /api/analyze."""
    data = to_bytes(make_trip() if trip is None else trip)
    return {'acc_file': ('RAW_ACCELEROMETERS.txt', data, 'text/plain')}


def wait_for_job(client, job_id, timeout=20.0):
    """Poll /api/job/{id} until the job ends. Returns (final body, every distinct status seen in order)."""
    seen, deadline = [], time.time() + timeout
    while time.time() < deadline:
        body = client.get(f'/api/job/{job_id}').json()
        if not seen or seen[-1] != body['status']:
            seen.append(body['status'])
        if body['status'] in ('done', 'error'):
            return body, seen
        time.sleep(0.02)
    raise AssertionError(f'job {job_id} did not finish in {timeout}s, statuses seen: {seen}')
