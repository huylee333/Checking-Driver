"""T09. Behaviour of the real sensor model through the HTTP API (AI Test, API).

These are metamorphic tests: we cannot say what the "right" label of a trip is, but we know how the answer must
change, or not change, when the input is changed in a known way.
  1. the same input gives the same answer (deterministic)
  2. a tiny amount of noise does not flip the answer
  3. a calm signal and a violent signal get clearly different answers, in the right direction

Needs the real weights and a real trip, so it only runs when these variables are set (CI does not set them):
  CD_WEIGHTS_DIR = folder with final_1d.pth and kine_norm_stats.json
  CD_DEMO_DIR    = folder with the demo trip (RAW_ACCELEROMETERS.txt). The data stays on your machine.
Measured on the development machine (sensor model, demo trip of 182 windows):
  noise 0.1% of the channel std -> 0.0% of window labels flip;  1% -> 0.5%;  5% -> 6.6%
  calm signal (noise 0.2 std) -> NORMAL 0.87;  violent signal (noise 8 std) -> AGGRESSIVE 0.79
"""
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as web
from inference import Analyzer
from synthetic import to_bytes, wait_for_job

WEIGHTS_DIR = os.environ.get('CD_WEIGHTS_DIR')
DEMO_DIR = os.environ.get('CD_DEMO_DIR')

pytestmark = pytest.mark.skipif(not (WEIGHTS_DIR and DEMO_DIR), reason='set CD_WEIGHTS_DIR and CD_DEMO_DIR')


@pytest.fixture(scope='module')
def client():
    analyzer = Analyzer(WEIGHTS_DIR)
    assert analyzer.ready, analyzer.error
    patch = pytest.MonkeyPatch()
    patch.setattr(web, 'analyzer', analyzer)            # the app's real endpoints, with the real model behind them
    yield TestClient(web.app)
    patch.undo()


@pytest.fixture(scope='module')
def trip():
    return np.loadtxt(os.path.join(DEMO_DIR, 'RAW_ACCELEROMETERS.txt'))


def analyse(client, acc):
    files = {'acc_file': ('RAW_ACCELEROMETERS.txt', to_bytes(acc), 'text/plain')}
    response = client.post('/api/analyze', files=files)
    assert response.status_code == 200, response.text
    body, _ = wait_for_job(client, response.json()['job_id'], timeout=300)
    assert body['status'] == 'done', body
    result = body['result']
    labels = [w['sensor']['label'] if w.get('sensor') else None for w in result['windows']]
    return result['models']['sensor'], labels


def noisy(acc, fraction, seed=1):
    """Add Gaussian noise to the three filtered acceleration channels (columns 5-7), `fraction` of each channel's std."""
    out = acc.copy()
    std = acc[:, 5:8].std(axis=0)
    out[:, 5:8] += np.random.default_rng(seed).normal(0, 1, (len(acc), 3)) * std * fraction
    return out


def replaced(acc, scale_in_std, seed=2):
    """Replace the three channels by pure noise of `scale_in_std` times the original std."""
    out = acc.copy()
    std = acc[:, 5:8].std(axis=0)
    out[:, 5:8] = np.random.default_rng(seed).normal(0, 1, (len(acc), 3)) * std * scale_in_std
    return out


@pytest.fixture(scope='module')
def base(client, trip):
    return analyse(client, trip)


def test_the_same_input_gives_the_same_answer(client, trip, base):
    again = analyse(client, trip)
    assert again[1] == base[1]
    assert again[0]['trip_probs'] == base[0]['trip_probs']


def test_a_tiny_amount_of_noise_does_not_change_the_answer(client, trip, base):
    model, labels = analyse(client, noisy(trip, 0.001))
    flips = np.mean([a != b for a, b in zip(base[1], labels)])
    assert model['trip_label'] == base[0]['trip_label']
    assert flips <= 0.02, f'{flips:.1%} of the window labels flipped with 0.1% noise'


def test_one_percent_noise_flips_only_a_few_windows(client, trip, base):
    _, labels = analyse(client, noisy(trip, 0.01))
    flips = np.mean([a != b for a, b in zip(base[1], labels)])
    assert flips <= 0.05, f'{flips:.1%} of the window labels flipped with 1% noise'


def test_a_calm_signal_is_normal_and_a_violent_signal_is_aggressive(client, trip):
    calm, _ = analyse(client, replaced(trip, 0.2))
    wild, _ = analyse(client, replaced(trip, 8.0))
    assert calm['trip_label'] == 'NORMAL' and wild['trip_label'] == 'AGGRESSIVE'
    assert calm['trip_probs']['NORMAL'] > wild['trip_probs']['NORMAL'] + 0.5
    assert wild['trip_probs']['AGGRESSIVE'] > calm['trip_probs']['AGGRESSIVE'] + 0.5
    assert wild['share']['AGGRESSIVE'] > calm['share']['AGGRESSIVE'] + 0.5


def test_probabilities_are_valid_for_every_window(base):
    model, labels = base
    assert len(labels) > 100 and model['windows'] > 100
    assert abs(sum(model['trip_probs'].values()) - 1.0) < 1e-3
