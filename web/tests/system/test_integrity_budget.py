"""T10. The right weights are running, within the time and memory budget, and the model is not worse (AI Test, System).

Budgets are the numbers measured on the development machine (CPU only), with headroom:
  start-up   measured about 35 s    -> budget 90 s
  analysis   measured about 47 s    -> budget 120 s    (demo trip: 181 windows, 15 minutes)
  memory     measured about 2.3 GiB -> budget 4.5 GiB  (right after a first download it was 4.23 GiB)
Accuracy: the full validation set is not in this repository, so the proxy is the demo trip (driver D6, labelled
AGGRESSIVE by the dataset): the share of windows each model calls AGGRESSIVE must not drop by more than 5 points.
Create or refresh the checksum list with CD_UPDATE_GOLDEN=1.
"""
import hashlib
import json
import os

import pytest
from ai_fixture import DEMO_DIR, WEIGHTS_DIR
from docker_helpers import docker

HERE = os.path.dirname(__file__)
CHECKSUMS = os.path.join(HERE, 'weights.sha256.json')
GOLDEN = os.path.join(HERE, 'golden.json')
WEIGHT_FILES = ['final_1d.pth', 'kine_norm_stats.json', 'video_3d.pth', 'fusion.pth']
START_BUDGET_S, ANALYSIS_BUDGET_S, MEMORY_BUDGET_GIB = 90, 120, 4.5


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope='module')
def checksums(demo_run):
    got = {f: sha256(os.path.join(WEIGHTS_DIR, f)) for f in WEIGHT_FILES}
    if os.environ.get('CD_UPDATE_GOLDEN') == '1':
        with open(CHECKSUMS, 'w', encoding='utf-8') as f:
            json.dump(got, f, indent=1)
        pytest.skip('weights.sha256.json written')
    if not os.path.isfile(CHECKSUMS):
        pytest.fail('weights.sha256.json is missing: run once with CD_UPDATE_GOLDEN=1')
    with open(CHECKSUMS, encoding='utf-8') as f:
        return json.load(f), got


def test_the_weight_files_on_disk_are_the_recorded_ones(checksums):
    want, got = checksums
    assert got == want


def test_the_container_sees_exactly_the_same_bytes(demo_run, checksums):
    _, host = checksums
    for name in WEIGHT_FILES:
        out = docker('exec', demo_run['container'].name, 'sha256sum', f'/home/user/app/weights/{name}')
        assert out.split()[0] == host[name]


def test_startup_is_within_budget(demo_run):
    assert demo_run['startup'] <= START_BUDGET_S, f"start-up took {demo_run['startup']:.0f}s"


def test_analysis_of_the_demo_trip_is_within_budget(demo_run):
    assert demo_run['job']['status'] == 'done'
    assert demo_run['analysis'] <= ANALYSIS_BUDGET_S, f"analysis took {demo_run['analysis']:.0f}s"


def test_memory_is_within_budget(demo_run):
    assert 0 < demo_run['peak_gib'] <= MEMORY_BUDGET_GIB, f"peak memory {demo_run['peak_gib']:.2f} GiB"


def test_the_container_still_runs_after_the_analysis(demo_run):
    assert demo_run['container'].inspect('{{.State.Running}}') == 'true'
    assert demo_run['container'].inspect('{{.State.OOMKilled}}') == 'false'


@pytest.mark.parametrize('model', ['sensor', 'video', 'fusion'])
def test_accuracy_on_the_demo_trip_did_not_drop_by_more_than_five_points(demo_run, model):
    assert os.path.isfile(GOLDEN), 'run T08 with CD_UPDATE_GOLDEN=1 first'
    with open(GOLDEN, encoding='utf-8') as f:
        baseline = json.load(f)['aggressive_share'][model]
    windows = [w for w in demo_run['result']['windows'] if w.get(model)]
    share = sum(1 for w in windows if w[model]['label'] == 'AGGRESSIVE') / len(windows)
    assert share >= baseline - 0.05, f'{model}: {share:.3f} vs baseline {baseline:.3f}'


def test_the_demo_trip_is_the_one_the_baseline_was_made_from(demo_run):
    with open(os.path.join(DEMO_DIR, 'meta.json'), encoding='utf-8') as f:
        meta = json.load(f)
    assert (meta['driver'], meta['behavior'], meta['windows']) == ('D6', 'AGGRESSIVE', 181)
