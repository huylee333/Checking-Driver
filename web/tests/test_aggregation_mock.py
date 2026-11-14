"""T05. Result summary with a mocked model (Mock, API).

The model is replaced by fixed probabilities, so the expected trip label, shares and events can be worked out by hand.
This checks the code around the network (windows, summary, events, series), which does not need any weights.
"""
import pytest

import inference as inf
from inference import Analyzer, load_sensors
from synthetic import make_trip, to_bytes


@pytest.fixture
def analyze(tmp_path, fake_model):
    def run(pattern, trip=None):
        fake_model.pattern = pattern
        sens = load_sensors(to_bytes(make_trip() if trip is None else trip))
        return Analyzer(str(tmp_path)).analyze(sens)
    return run


def test_the_synthetic_trip_has_13_valid_windows_of_5_seconds(analyze):
    # 701 samples at 10 Hz, cut into consecutive blocks of 51 samples (13 blocks fit)
    r = analyze([[0.8, 0.1, 0.1]])
    assert r['trip']['windows_total'] == 13 and r['trip']['windows_valid'] == 13
    assert r['trip']['duration'] == pytest.approx(70.0)


def test_a_trip_of_confident_aggressive_windows_is_one_high_confidence_event(analyze):
    r = analyze([[0.05, 0.9, 0.05]])
    s = r['models']['sensor']
    assert s['trip_label'] == 'AGGRESSIVE'
    assert s['share']['AGGRESSIVE'] == pytest.approx(1.0)
    assert s['seconds']['AGGRESSIVE'] == pytest.approx(65.0)      # 13 windows x 5 s
    assert len(s['events']) == 1
    ev = s['events'][0]
    assert (ev['label'], ev['n_windows'], ev['level']) == ('AGGRESSIVE', 13, 'high')
    assert ev['confidence'] == pytest.approx(0.9)
    assert (ev['t_start'], ev['t_end']) == (r['windows'][0]['t_start'], r['windows'][-1]['t_end'])


def test_a_normal_trip_has_no_events(analyze):
    s = analyze([[0.9, 0.05, 0.05]])['models']['sensor']
    assert s['trip_label'] == 'NORMAL' and s['events'] == []


def test_the_trip_label_follows_the_mean_probability_not_the_time_share(analyze):
    # 7 windows say AGGRESSIVE with little confidence (0.40 vs normal 0.35), 6 windows say NORMAL with 0.90.
    # Time share: aggressive 7/13 = 0.538. Mean probability: normal (7*0.35 + 6*0.90)/13 = 0.604, aggressive 0.238.
    s = analyze([[0.35, 0.40, 0.25], [0.90, 0.05, 0.05]])['models']['sensor']
    assert s['share']['AGGRESSIVE'] == pytest.approx(7 / 13) and s['share']['NORMAL'] == pytest.approx(6 / 13)
    assert s['trip_probs']['NORMAL'] == pytest.approx(0.6038, abs=1e-3)
    assert s['trip_probs']['AGGRESSIVE'] == pytest.approx(0.2385, abs=1e-3)
    assert s['trip_label'] == 'NORMAL'


def test_separated_aggressive_windows_are_separate_medium_events(analyze):
    s = analyze([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])['models']['sensor']
    assert len(s['events']) == 7
    assert {e['n_windows'] for e in s['events']} == {1}
    assert {e['level'] for e in s['events']} == {'medium'}       # confidence 0.7 is below the 0.8 "high" level


def test_a_label_below_the_event_threshold_is_not_an_event(analyze):
    s = analyze([[0.30, 0.55, 0.15]])['models']['sensor']        # 0.55 < EVENT_MIN_PROB (0.6)
    assert s['trip_label'] == 'AGGRESSIVE'
    assert s['events'] == []


def test_every_window_probability_sums_to_one_and_uses_a_known_label(analyze):
    r = analyze([[0.2, 0.5, 0.3], [0.6, 0.3, 0.1], [0.1, 0.1, 0.8]])
    for w in r['windows']:
        assert sum(w['sensor']['probs']) == pytest.approx(1.0, abs=1e-3)
        assert w['sensor']['label'] in inf.CLASSES


def test_a_window_with_a_data_gap_is_reported_not_analysed(analyze):
    trip = make_trip()
    trip = trip[(trip[:, 0] < 20.0) | (trip[:, 0] >= 26.0)]      # a 6 s hole in the recording
    r = analyze([[0.9, 0.05, 0.05]], trip=trip)
    bad = [w for w in r['windows'] if not w['valid']]
    assert bad and all(w['reason'] == 'mất dữ liệu cảm biến' and w['sensor'] is None for w in bad)
    assert r['trip']['windows_valid'] < r['trip']['windows_total']
    assert r['models']['sensor']['windows'] == r['trip']['windows_valid']


def test_without_video_the_video_and_fusion_results_are_empty(analyze):
    r = analyze([[0.8, 0.1, 0.1]])
    assert r['models']['video'] is None and r['models']['fusion'] is None
    assert r['video']['status'] == 'none'
    assert all(w.get('video') is None for w in r['windows'])


def test_chart_series_are_aligned_and_gps_fields_are_absent_without_gps(analyze):
    r = analyze([[0.8, 0.1, 0.1]])
    assert len(r['series']['t']) == len(r['series']['motion']) == 701
    assert 'speed' not in r['series'] and 'distance_km' not in r['trip']
