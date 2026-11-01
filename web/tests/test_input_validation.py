"""Input validation of the sensor files (inference.py).

The fixtures are tiny synthetic tables written for these tests. They are NOT UAH-DriveSet data.
Accelerometer file: 11 whitespace-separated columns, column 0 = timestamp (s), column 1 = activation flag.
GPS file: 12 columns.
"""
import numpy as np
import pytest

import inference as inf
from inference import InputError, clean_timeseries, decode_text, load_sensors, make_windows, parse_table


def make_table(n=700, hz=10.0, flag=1, cols=11):
    """n rows at `hz` Hz: a 51-sample window then spans exactly 5.0 s, as the model expects."""
    rng = np.random.default_rng(0)
    a = np.zeros((n, cols))
    a[:, 0] = np.arange(n) / hz
    a[:, 1] = flag
    a[:, 2:] = rng.normal(0.0, 0.1, (n, cols - 2))
    return a


def to_bytes(a):
    return ('\n'.join(' '.join(f'{x:.6f}' for x in row) for row in a) + '\n').encode()


# ---- decode_text -----------------------------------------------------------------------------
def test_decode_text_rejects_empty():
    with pytest.raises(InputError, match='trống'):
        decode_text(b'  \n ', 'f')


def test_decode_text_rejects_binary():
    with pytest.raises(InputError, match='nhị phân'):
        decode_text(b'abc\x00def', 'f')


def test_decode_text_rejects_bad_encoding():
    with pytest.raises(InputError, match='mã hoá'):
        decode_text(b'\xff\xfe\xfd', 'f')


def test_decode_text_accepts_utf8_with_bom():
    assert decode_text('﻿1 2 3'.encode('utf-8'), 'f') == '1 2 3'


# ---- parse_table -----------------------------------------------------------------------------
def test_parse_table_wrong_column_count():
    with pytest.raises(InputError, match='cần 11 cột'):
        parse_table('1 2 3\n', 11, 'f')


def test_parse_table_hints_when_commas_are_used():
    with pytest.raises(InputError, match='khoảng trắng'):
        parse_table('1,2,3\n', 11, 'f')


def test_parse_table_non_numeric():
    row = ' '.join(['abc'] + ['1'] * 10)
    with pytest.raises(InputError, match='không phải số'):
        parse_table(row + '\n', 11, 'f')


def test_parse_table_too_few_rows():
    with pytest.raises(InputError, match='quá ít dữ liệu'):
        parse_table(to_bytes(make_table(n=10)).decode(), 11, 'f')


def test_parse_table_rejects_nan():
    a = make_table(n=60)
    a[5, 4] = np.nan
    with pytest.raises(InputError, match='NaN'):
        parse_table(to_bytes(a).decode().replace('nan', 'nan'), 11, 'f')


def test_parse_table_accepts_valid_table():
    out = parse_table(to_bytes(make_table(n=60)).decode(), 11, 'f')
    assert out.shape == (60, 11)


# ---- clean_timeseries ------------------------------------------------------------------------
def test_clean_timeseries_drops_everything_after_a_backjump():
    a = make_table(n=20)
    a[12:, 0] = np.arange(8) * 0.1          # time jumps back at row 12
    cleaned, info = clean_timeseries(a)
    assert len(cleaned) == 12
    assert info['dropped_after_backjump'] == 8
    assert info['ambiguous'] is not None


def test_clean_timeseries_drops_whole_groups_of_duplicate_timestamps():
    a = make_table(n=5)
    a[:, 0] = [0.0, 1.0, 1.0, 2.0, 3.0]     # rows 1 and 2 share a timestamp: both are removed
    cleaned, info = clean_timeseries(a)
    assert list(cleaned[:, 0]) == [0.0, 2.0, 3.0]
    assert info['dropped_duplicate_ts'] == 2


# ---- make_windows ----------------------------------------------------------------------------
def test_make_windows_rejects_a_gap_even_when_the_window_length_is_acceptable():
    """One 0.6 s hole (> 0.5 s allowed) keeps the window span at 5.5 s, inside [4.6, 5.6]:
    only the gap rule can reject it, so this test fails if that rule is removed."""
    a = make_table(n=400)
    a[200:, 0] += 0.5
    wins = make_windows(a, [None])
    crossing = [w for w in wins if w['t_start'] < a[199, 0] < w['t_end']]
    assert crossing and not any(w['valid'] for w in crossing)
    assert all(w['reason'] == 'mất dữ liệu cảm biến' for w in crossing)
    assert any(w['valid'] for w in wins)


def test_make_windows_rejects_a_long_hole():
    a = make_table(n=400)
    a[200:, 0] += 3.0                       # a 3 s hole
    wins = make_windows(a, [None])
    assert any((not w['valid']) and w['reason'] == 'mất dữ liệu cảm biến' for w in wins)
    assert any(w['valid'] for w in wins)


def test_make_windows_marks_inactive_sensor_as_invalid():
    wins = make_windows(make_table(n=200, flag=0), [None])
    assert wins and not any(w['valid'] for w in wins)
    assert all(w['reason'] == 'cảm biến chưa sẵn sàng' for w in wins)


# ---- load_sensors (end to end of the validation) ---------------------------------------------
def test_load_sensors_accepts_a_valid_trip():
    s = load_sensors(to_bytes(make_table(n=700)))
    assert s['acc'].shape[1] == 11
    assert sum(w['valid'] for w in s['windows']) >= 10
    assert s['gps'] is None


def test_load_sensors_rejects_a_trip_shorter_than_a_minute():
    with pytest.raises(InputError, match='quá ngắn'):
        load_sensors(to_bytes(make_table(n=300)))        # 30 s


def test_load_sensors_rejects_a_trip_longer_than_two_hours():
    with pytest.raises(InputError, match='quá dài'):
        load_sensors(to_bytes(make_table(n=7300, hz=1.0)))   # 7299 s


def test_load_sensors_rejects_when_no_window_is_usable():
    with pytest.raises(InputError, match='Không có đoạn dữ liệu'):
        load_sensors(to_bytes(make_table(n=700, flag=0)))


def test_load_sensors_detects_a_gps_file_passed_as_accelerometer_file():
    with pytest.raises(InputError, match='GPS'):
        load_sensors(to_bytes(make_table(n=700, cols=12)))


def test_limits_match_the_documented_values():
    assert inf.MIN_TRIP_SEC == 61   # DELIBERATELY WRONG (real value is 60): proves that CI turns red. Do not merge.
    assert inf.MAX_TRIP_SEC == 2 * 3600
    assert inf.KINE_LEN == 51
