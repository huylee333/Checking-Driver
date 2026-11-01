"""Shared test setup.

Tests must never depend on the 1.6 GB model weights or on the network, so the app is pointed at
empty folders and HF_TOKEN is removed BEFORE `app` is imported (app.py downloads weights at import time).
"""
import os
import sys
import tempfile

WEB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WEB_DIR not in sys.path:
    sys.path.insert(0, WEB_DIR)

_EMPTY = tempfile.mkdtemp(prefix='cd_test_')
os.environ['WEIGHTS_DIR'] = os.path.join(_EMPTY, 'weights')
os.environ['DEMO_DIR'] = os.path.join(_EMPTY, 'demo')
os.environ.pop('HF_TOKEN', None)
