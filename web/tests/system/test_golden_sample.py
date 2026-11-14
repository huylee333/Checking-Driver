"""T08. Golden sample: the demo trip analysed by the real models must give the recorded answer (AI Test, System).

The "golden" file holds only the labels and mean probabilities the models produced for the demo trip, NOT the trip's
raw data. Create or refresh it on purpose with CD_UPDATE_GOLDEN=1 (after a model or preprocessing change that you
accept), then commit it.
"""
import json
import os

import pytest

GOLDEN = os.path.join(os.path.dirname(__file__), 'golden.json')
MODELS = ('sensor', 'video', 'fusion')


def summarise(result):
    labels = {m: [(w.get(m) or {}).get('label') for w in result['windows']] for m in MODELS}
    return {
        'windows_total': result['trip']['windows_total'],
        'windows_valid': result['trip']['windows_valid'],
        'trip_label': {m: result['models'][m]['trip_label'] for m in MODELS},
        'trip_probs': {m: {k: round(v, 4) for k, v in result['models'][m]['trip_probs'].items()} for m in MODELS},
        'window_labels': labels,
        'aggressive_share': {m: round(labels[m].count('AGGRESSIVE') / max(1, result['models'][m]['windows']), 4)
                             for m in MODELS},
    }


@pytest.fixture(scope='module')
def golden(demo_run):
    got = summarise(demo_run['result'])
    if os.environ.get('CD_UPDATE_GOLDEN') == '1':
        with open(GOLDEN, 'w', encoding='utf-8') as f:
            json.dump(got, f, ensure_ascii=False, indent=1)
        pytest.skip('golden.json written, run again without CD_UPDATE_GOLDEN to compare')
    if not os.path.isfile(GOLDEN):
        pytest.fail('golden.json is missing: run once with CD_UPDATE_GOLDEN=1 and review the file')
    with open(GOLDEN, encoding='utf-8') as f:
        return json.load(f), got


def test_the_job_finishes_and_all_three_models_answered(demo_run):
    assert demo_run['job']['status'] == 'done', demo_run['job'].get('error')
    assert demo_run['status']['ready'] and demo_run['status']['video'] and demo_run['status']['fusion']
    assert all(demo_run['result']['models'][m] is not None for m in MODELS)


def test_the_window_count_is_unchanged(golden):
    want, got = golden
    assert (got['windows_total'], got['windows_valid']) == (want['windows_total'], want['windows_valid'])


@pytest.mark.parametrize('model', MODELS)
def test_the_trip_label_is_unchanged(golden, model):
    want, got = golden
    assert got['trip_label'][model] == want['trip_label'][model]


@pytest.mark.parametrize('model', MODELS)
def test_at_least_95_percent_of_window_labels_are_unchanged(golden, model):
    want, got = golden
    pairs = list(zip(want['window_labels'][model], got['window_labels'][model]))
    assert len(pairs) == len(want['window_labels'][model])
    same = sum(1 for a, b in pairs if a == b)
    assert same / len(pairs) >= 0.95, f'{model}: only {same}/{len(pairs)} window labels match the golden sample'


@pytest.mark.parametrize('model', MODELS)
def test_mean_probabilities_stay_within_two_points(golden, model):
    want, got = golden
    for cls, p in want['trip_probs'][model].items():
        assert got['trip_probs'][model][cls] == pytest.approx(p, abs=0.02)
