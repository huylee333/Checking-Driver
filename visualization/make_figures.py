"""
Data visualization for the DuST-DINOv3 training data (UAH-DriveSet, after cleaning).

All figures are drawn from the exact files the model was trained on
(runs/<run>/data: video/kine/label pickles, windows_*.csv, uah_trip_report.csv),
except Figure 4a, which re-runs the sync check on one raw trip to show the curve.

Usage (from dust_dinov3/):
  python visualization/make_figures.py
  python visualization/make_figures.py --data runs/dinov3_vits16_run1/data --uah ../UAH-DRIVESET-v1/DRIVERSET_H

Every figure goes through check_layout(): text boxes that overlap each other or leave the
canvas, and axes that overlap each other, are reported before the PNG is written.
"""

import argparse
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # dust_dinov3/
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(errors='backslashreplace')
except AttributeError:
    pass

# ----------------------------------------------------------------------------
# Style (reference palette of the dataviz method; 3-class palette validated all-pairs)
# ----------------------------------------------------------------------------
CLASSES = ['NORMAL', 'AGGRESSIVE', 'DROWSY']
CLASS_NAME = {'NORMAL': 'Normal', 'AGGRESSIVE': 'Aggressive', 'DROWSY': 'Drowsy'}
CLASS_COLOR = {'NORMAL': '#2a78d6', 'AGGRESSIVE': '#eb6834', 'DROWSY': '#1baf7a'}
NEUTRAL = '#898781'          # single-series bars that are NOT a class
SURFACE = '#fcfcfb'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS = '#e1e0d9', '#c3c2b7'
SEQ = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']   # blue ramp 100..700
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

plt.rcParams.update({
    'font.family': ['Segoe UI', 'DejaVu Sans', 'sans-serif'],
    'font.size': 9, 'axes.titlesize': 10.5, 'axes.titleweight': 'semibold', 'axes.titlelocation': 'left',
    'axes.titlepad': 10, 'axes.labelsize': 9, 'axes.labelcolor': INK2,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.8, 'axes.facecolor': SURFACE, 'figure.facecolor': SURFACE,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'xtick.major.size': 0, 'ytick.major.size': 0, 'xtick.major.pad': 4, 'ytick.major.pad': 4,
    'axes.grid': False, 'grid.color': GRID, 'grid.linewidth': 0.7, 'grid.linestyle': '-',
    'legend.frameon': False, 'legend.fontsize': 8.5, 'savefig.facecolor': SURFACE,
})


def ygrid(ax):
    ax.grid(axis='y', color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def xgrid(ax):
    ax.grid(axis='x', color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------------
# Layout check
# ----------------------------------------------------------------------------
def check_layout(fig, name, tol=1.0):
    """Report overlapping text boxes, text leaving the canvas, and overlapping axes. Returns #issues."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    # tick labels of ticks outside the visible axis range are never drawn -> ignore them
    hidden = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not (lo - 1e-12 <= tick.get_loc() <= hi + 1e-12) or not tick.label1.get_visible():
                    hidden.update({id(tick.label1), id(tick.label2)})
            hidden.update(id(tl) for tl in axis.get_ticklabels(minor=True) if not tl.get_text())
    texts = [t for t in fig.findobj(mtext.Text)
             if t.get_visible() and t.get_text().strip() and id(t) not in hidden]
    boxes = []
    for t in texts:
        try:
            boxes.append((t, t.get_window_extent(r)))
        except RuntimeError:
            pass
    issues = []
    for i in range(len(boxes)):
        ti, bi = boxes[i]
        if bi.x0 < -tol or bi.y0 < -tol or bi.x1 > W + tol or bi.y1 > H + tol:
            issues.append(f'text outside canvas: "{ti.get_text()[:40]}"')
        for j in range(i + 1, len(boxes)):
            tj, bj = boxes[j]
            w = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
            h = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
            if w > tol and h > tol:
                issues.append(f'text overlap: "{ti.get_text()[:30]}" <-> "{tj.get_text()[:30]}"')
    axes = [a for a in fig.axes if a.get_visible()]
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            a, b = axes[i].get_window_extent(r), axes[j].get_window_extent(r)
            w = min(a.x1, b.x1) - max(a.x0, b.x0)
            h = min(a.y1, b.y1) - max(a.y0, b.y0)
            if w > tol and h > tol:
                issues.append(f'axes overlap: {axes[i].get_title()[:25] or i} <-> {axes[j].get_title()[:25] or j}')
    print(f'[layout] {name}: {len(texts)} text items, {len(axes)} axes -> '
          + ('OK' if not issues else f'{len(issues)} issue(s)'))
    for s in issues:
        print('    -', s)
    return len(issues)


def save(fig, out, name):
    n = check_layout(fig, name)
    fig.savefig(os.path.join(out, name), dpi=200, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    return n


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_meta(data_dir):
    rep = pd.read_csv(os.path.join(data_dir, 'uah_trip_report.csv'))
    wtr = pd.read_csv(os.path.join(data_dir, 'windows_train.csv'))
    wva = pd.read_csv(os.path.join(data_dir, 'windows_val.csv'))
    wtr['split'], wva['split'] = 'train', 'val'
    stats = json.load(open(os.path.join(data_dir, 'kine_norm_stats.json')))
    mean = np.asarray(stats['mean'])[:, None]
    std = np.asarray(stats['std'])[:, None]
    ktr = pd.read_pickle(os.path.join(data_dir, 'kine_train.pkl')) * std + mean   # back to g
    kva = pd.read_pickle(os.path.join(data_dir, 'kine_val.pkl')) * std + mean
    return rep, wtr, wva, ktr, kva


# ----------------------------------------------------------------------------
# Figure 1: dataset composition
# ----------------------------------------------------------------------------
def fig1_dataset(rep, wtr, wva, out):
    ok = rep[rep.status == 'OK']
    drivers = sorted(rep.driver.dropna().unique())
    val_drivers = set(wva.driver.unique())
    M = np.array([[((ok.driver == d) & (ok.behavior == c)).sum() for c in CLASSES] for d in drivers])

    fig = plt.figure(figsize=(12, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.45, top=0.80, bottom=0.2)

    ax = fig.add_subplot(gs[0])
    vmax = max(3, M.max())
    cmap = ListedColormap(SEQ[:vmax + 1])
    ax.imshow(M, cmap=cmap, vmin=-0.5, vmax=vmax + 0.5, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, str(M[i, j]), ha='center', va='center', fontsize=10,
                    color='white' if M[i, j] >= 3 else INK)
    ax.set_xticks(range(len(CLASSES)), [CLASS_NAME[c] for c in CLASSES])
    ax.set_yticks(range(len(drivers)), [f'{d} ({"val" if d in val_drivers else "train"})' for d in drivers])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f'(a) Usable trips per driver ({len(ok)} of {len(rep)})')
    excl = rep[rep.status != 'OK']
    if len(excl):
        e = excl.iloc[0]
        ax.text(0, -0.16, f'Excluded: {e.driver} {CLASS_NAME[e.behavior].lower()} {e.road.lower()} '
                f'(video sync not verified)', transform=ax.transAxes, fontsize=8, color=MUTED, va='top')

    for k, (df, title) in enumerate([(wtr, f'(b) Training windows (n = {len(wtr)})'),
                                     (wva, f'(c) Validation windows (n = {len(wva)})')]):
        ax = fig.add_subplot(gs[k + 1])
        counts = [int((df.behavior == c).sum()) for c in CLASSES]
        y = np.arange(len(CLASSES))[::-1]
        ax.barh(y, counts, height=0.62, color=[CLASS_COLOR[c] for c in CLASSES])
        top = max(counts)
        for yi, c in zip(y, counts):
            ax.text(c + top * 0.02, yi, str(c), va='center', fontsize=9, color=INK)
        ax.set_yticks(y, [CLASS_NAME[c] for c in CLASSES])
        ax.set_xlim(0, top * 1.18)
        ax.set_xlabel('5-s windows')
        xgrid(ax)
        ax.spines['left'].set_visible(False)
        ax.set_title(title)
    fig.suptitle('Dataset composition (UAH-DriveSet, split by driver)', x=0.06, ha='left',
                 fontsize=12, fontweight='semibold', y=0.98)
    return save(fig, out, 'fig1_dataset_composition.png')


# ----------------------------------------------------------------------------
# Figure 2: multimodal samples (exactly what the model sees)
# ----------------------------------------------------------------------------
def pick_windows(wva, kva):
    """One representative validation window per class: the one whose acceleration variability
    (mean std over the 3 axes) is closest to the class median."""
    pick = {}
    var = kva.std(axis=2).mean(axis=1)
    for c in CLASSES:
        idx = np.where(wva.behavior.values == c)[0]
        med = np.median(var[idx])
        pick[c] = int(idx[np.argmin(np.abs(var[idx] - med))])
    return pick


def fig2_samples(data_dir, wva, kva, out):
    pick = pick_windows(wva, kva)
    vid = pd.read_pickle(os.path.join(data_dir, 'video_val.pkl'))       # (N, 3, 77, 224, 224) float16, normalized
    frame_idx = [0, 19, 38, 57, 76]
    t = np.linspace(0, 5, kva.shape[2])

    fig = plt.figure(figsize=(13, 8.4))
    outer = fig.add_gridspec(len(CLASSES), 2, width_ratios=[5, 1.9], wspace=0.12, hspace=0.42, top=0.88, bottom=0.07)
    axis_name = ['X', 'Y', 'Z']
    lims = [(-0.0, 0.0)] * 3
    for k in range(3):
        v = kva[list(pick.values()), k]
        pad = 0.15 * (v.max() - v.min() + 1e-6)
        lims[k] = (v.min() - pad, v.max() + pad)

    for r, c in enumerate(CLASSES):
        i = pick[c]
        w = wva.iloc[i]
        img_gs = GridSpecFromSubplotSpec(1, len(frame_idx), subplot_spec=outer[r, 0], wspace=0.04)
        for j, f in enumerate(frame_idx):
            ax = fig.add_subplot(img_gs[j])
            x = vid[i][:, f].astype(np.float32).transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
            ax.imshow(np.clip(x, 0, 1))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(f'frame {f + 1}/77  ·  t = {f / 76 * 5:.1f} s', fontsize=8, color=INK2,
                         fontweight='normal', loc='center', pad=4)
            if j == 0:
                # row header: colour swatch, class name, then sample info, each placed right after the previous
                sw = ax.text(0, 1.22, '■', transform=ax.transAxes, fontsize=12, color=CLASS_COLOR[c], va='bottom')
                nm = ax.annotate(CLASS_NAME[c], xy=(1, 0), xycoords=sw, xytext=(5, 0), textcoords='offset points',
                                 fontsize=11, fontweight='semibold', color=INK, va='bottom')
                ax.annotate(f'driver {w.driver}, {w.road.lower()}, trip time {w.data_t_start:.0f}–{w.data_t_end:.0f} s',
                            xy=(1, 0), xycoords=nm, xytext=(10, 0), textcoords='offset points',
                            fontsize=8.5, color=INK2, va='bottom')
        sig_gs = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[r, 1], hspace=0.18)
        for k in range(3):
            ax = fig.add_subplot(sig_gs[k])
            ax.plot(t, kva[i, k], color=CLASS_COLOR[c], linewidth=1.6)
            ax.set_ylim(*lims[k]); ax.set_xlim(0, 5)
            ygrid(ax)
            ax.set_ylabel(axis_name[k], rotation=0, fontsize=9, color=INK, labelpad=10, va='center')
            ax.tick_params(labelsize=7.5)
            ax.locator_params(axis='y', nbins=3)
            if k < 2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('time in window (s)', fontsize=8)
            if k == 0:
                ax.set_title('Acceleration (g), 51 samples @ 10 Hz', fontsize=8.5, fontweight='normal', pad=4)
    fig.suptitle('One validation sample per class: 5 of the 77 video frames and the 3-axis '
                 'acceleration of the same 5-s window', x=0.06, ha='left', fontsize=12,
                 fontweight='semibold', y=0.98)
    return save(fig, out, 'fig2_multimodal_samples.png'), pick


# ----------------------------------------------------------------------------
# Figure 3: per-window acceleration variability by class
# ----------------------------------------------------------------------------
def fig3_variability(wtr, wva, ktr, kva, out):
    k = np.concatenate([ktr, kva])
    beh = np.concatenate([wtr.behavior.values, wva.behavior.values])
    sd = k.std(axis=2)                               # (N, 3) std over the 51 samples
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), sharey=True)
    fig.subplots_adjust(wspace=0.32, top=0.80, bottom=0.2)
    for a, ax in enumerate(axes):
        data = [sd[beh == c, a] for c in CLASSES]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=True,
                        medianprops=dict(color=INK, linewidth=1.4),
                        whiskerprops=dict(color=INK2, linewidth=0.9), capprops=dict(color=INK2, linewidth=0.9),
                        flierprops=dict(marker='o', markersize=3, markerfacecolor=MUTED, markeredgecolor='none', alpha=0.6))
        for patch, c in zip(bp['boxes'], CLASSES):
            patch.set_facecolor(CLASS_COLOR[c]); patch.set_alpha(0.85); patch.set_edgecolor(SURFACE)
        ax.set_xticks(range(1, 4), [f'{CLASS_NAME[c]}\n(n = {len(d)})' for c, d in zip(CLASSES, data)])
        ax.set_yscale('log')
        ticks = [0.005, 0.01, 0.02, 0.05, 0.1]
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylim(0.0045, 0.12)
        ygrid(ax)
        ax.set_title(f'Acceleration {"XYZ"[a]}')
        if a == 0:
            ax.set_ylabel('std within window (g, log scale)')
    fig.suptitle('Acceleration variability per 5-s window, by class (train + validation)',
                 x=0.06, ha='left', fontsize=12, fontweight='semibold', y=0.98)
    return save(fig, out, 'fig3_acceleration_variability.png')


# ----------------------------------------------------------------------------
# Figure 4: video <-> sensor synchronization check
# ----------------------------------------------------------------------------
def fig4_sync(rep, uah_root, out):
    import prepare_uah_data as pu
    ok = rep[rep.status == 'OK'].copy()
    base = ok.override_offset.where(ok.override_offset.notna(), ok.name_offset)
    ok['correction'] = ok.offset_used - base

    fig = plt.figure(figsize=(12, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.28, top=0.80, bottom=0.16)

    # (a) correlation vs lag for one validation trip
    ax = fig.add_subplot(gs[0])
    trip = None
    cands = sorted(glob.glob(os.path.join(uah_root, 'D6', '*NORMAL-SECONDARY')))
    if cands:
        trip = cands[0]
        name = os.path.basename(trip)
        mp4 = glob.glob(os.path.join(trip, '*.mp4'))[0]
        acc, _ = pu.clean_timeseries(pu.load_table(os.path.join(trip, 'RAW_ACCELEROMETERS.txt')))
        gps, _ = pu.clean_timeseries(pu.load_table(os.path.join(trip, 'RAW_GPS.txt')))
        print('[fig4] scanning video for the sync curve …')
        _, _, flow_t, flow_x = pu.scan_video(mp4)
        ref, rt, rx = pu.reference_signals(acc, gps)[0]
        lags = np.arange(-120, 120.01, 0.2)
        corr = pu.correlation_scan(flow_t, flow_x, rt, rx, lags, 0.5)
        row = rep[rep.trip == name].iloc[0]
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.plot(lags, corr, color=CLASS_COLOR['NORMAL'], linewidth=1.6)
        i = int(np.nanargmax(corr))
        ax.plot([lags[i]], [corr[i]], 'o', markersize=7, color=CLASS_COLOR['NORMAL'],
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f'verified offset {lags[i]:+.1f} s  (r = {corr[i]:.2f})', (lags[i], corr[i]),
                    xytext=(14, 2), textcoords='offset points', fontsize=8.5, color=INK, va='center')
        ax.axvline(row.name_offset, color=MUTED, linewidth=1.0)
        ax.text(row.name_offset - 3, corr[i], f'file-name offset {row.name_offset:+.0f} s',
                fontsize=8, color=INK2, ha='right', va='center')
        ax.set_ylim(np.nanmin(corr) - 0.05, corr[i] + 0.08)
        ax.set_xlim(lags[0], lags[-1])
        ax.set_xlabel('candidate offset: video time − sensor time (s)')
        ax.set_ylabel('correlation')
        ygrid(ax)
        ax.set_title(f'(a) Video horizontal motion vs. yaw rate, {row.driver} normal secondary')
    else:
        ax.text(0.5, 0.5, 'raw trip not found', ha='center', transform=ax.transAxes, color=MUTED)

    # (b) correction applied per trip
    ax = fig.add_subplot(gs[1])
    bins = np.arange(np.floor(ok.correction.min() * 2) / 2 - 0.25, np.ceil(ok.correction.max() * 2) / 2 + 0.5, 0.5)
    ax.hist(ok.correction, bins=bins, color=NEUTRAL, edgecolor=SURFACE, linewidth=2)
    ygrid(ax)
    ax.set_xlabel('verified offset − file-name offset (s)')
    ax.set_ylabel('trips')
    ax.set_title(f'(b) Correction to file-name timestamps ({len(ok)} trips)')
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ny = int((ok.sync_ref == 'yaw_rate').sum())
    ax.text(0.03, 0.95, f'reference signal:\nyaw rate {ny} trips\nGPS course {len(ok) - ny} trips',
            transform=ax.transAxes, ha='left', va='top', fontsize=8.5, color=INK2)
    fig.suptitle('Data-driven check of video–sensor synchronization', x=0.06, ha='left',
                 fontsize=12, fontweight='semibold', y=0.98)
    return save(fig, out, 'fig4_sync_verification.png')


# ----------------------------------------------------------------------------
# Figure 5: window filtering
# ----------------------------------------------------------------------------
REASON = {'system_inactive': 'Sensor not active', 'sensor_gap': 'Gap > 0.5 s between samples',
          'sensor_span': 'Window span outside 4.6–5.6 s', 'outside_video': 'Outside video coverage',
          'ambiguous_time': 'Duplicated recording (ambiguous)', 'frame_time_mismatch': 'No frame close enough',
          'dropped_frames': 'Dropped video frames', 'dark_frame': 'Dark frame', 'frozen_video': 'Frozen video',
          'frame_read_failed': 'Frame read failed'}


def fig5_filtering(rep, wtr, wva, out):
    ok = rep[rep.status == 'OK']
    reasons = {}
    for col in ['reject_sensor', 'reject_video']:
        for s in ok[col].dropna():
            for k, v in json.loads(s).items():
                reasons[k] = reasons.get(k, 0) + int(v)
    rej_sensor = sum(sum(json.loads(s).values()) for s in ok.reject_sensor.dropna())
    valid = int(ok.windows_valid.sum())
    cand = valid + rej_sensor
    used = len(wtr) + len(wva)

    fig = plt.figure(figsize=(12, 3.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.75, top=0.78, bottom=0.16)

    ax = fig.add_subplot(gs[0])
    labels = ['Candidate 5-s windows', 'Valid after cleaning', 'Used for training\n(8 per trip)']
    vals = [cand, valid, used]
    y = np.arange(3)[::-1]
    ax.barh(y, vals, height=0.6, color=NEUTRAL)
    for yi, v in zip(y, vals):
        ax.text(v + cand * 0.015, yi, f'{v:,}', va='center', fontsize=9, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, cand * 1.15)
    ax.set_xlabel('windows')
    xgrid(ax)
    ax.spines['left'].set_visible(False)
    ax.set_title(f'(a) From candidate windows to training data ({len(ok)} trips)')

    ax = fig.add_subplot(gs[1])
    items = sorted(reasons.items(), key=lambda kv: kv[1])
    y = np.arange(len(items))
    ax.barh(y, [v for _, v in items], height=0.6, color=NEUTRAL)
    top = max(v for _, v in items)
    for yi, (_, v) in zip(y, items):
        ax.text(v + top * 0.02, yi, str(v), va='center', fontsize=9, color=INK)
    ax.set_yticks(y, [REASON.get(k, k) for k, _ in items])
    ax.set_xlim(0, top * 1.18)
    ax.set_xlabel('windows removed')
    xgrid(ax)
    ax.spines['left'].set_visible(False)
    ax.set_title('(b) Why windows were removed')
    fig.suptitle('Window filtering (no sample is generated or repaired, only removed)',
                 x=0.06, ha='left', fontsize=12, fontweight='semibold', y=0.98)
    return save(fig, out, 'fig5_window_filtering.png')


# ----------------------------------------------------------------------------
# Figure 6: training curves, parsed from the training logs
# ----------------------------------------------------------------------------
EPOCH_RE = re.compile(r'Epoch: (\d+)/\d+, Train Loss: ([\d.]+)(?: \(CE ([\d.]+), SupCon ([\d.]+)\))?'
                      r', Val Loss: ([\d.]+), Val Acc: ([\d.]+)')


def parse_log(path):
    """Epoch lines of the last run in a log (a log can hold several restarts)."""
    lines = open(path, encoding='utf-8', errors='replace').read().splitlines()
    starts = [i for i, l in enumerate(lines) if 'Define Data Loader' in l]
    rows = []
    for l in lines[starts[-1] if starts else 0:]:
        m = EPOCH_RE.search(l)
        if m:
            e, tr, ce, sc, vl, va = m.groups()
            rows.append(dict(epoch=int(e), train=float(tr), ce=float(ce) if ce else np.nan,
                             supcon=float(sc) if sc else np.nan, val_loss=float(vl), val_acc=float(va)))
    return pd.DataFrame(rows)


def fig6_training(log_dir, out, batch=16):
    d1 = parse_log(os.path.join(log_dir, 'step_1d.log'))
    d3 = parse_log(os.path.join(log_dir, 'step_dinov3_3d.log'))
    c1, c3 = '#2a78d6', '#eb6834'

    fig = plt.figure(figsize=(13, 3.9))
    gs = fig.add_gridspec(1, 3, wspace=0.32, top=0.78, bottom=0.16)

    ax = fig.add_subplot(gs[0])
    ax.plot(d1.epoch, d1.train, color=c1, lw=1.6, label='train')
    ax.plot(d1.epoch, d1.val_loss, color=MUTED, lw=1.2, label='validation (D6)')
    ax.set_ylim(0, 3)
    ax.set_xlabel('epoch')
    ax.set_ylabel('cross-entropy')
    ygrid(ax)
    ax.legend(loc='upper right')
    ax.set_title('(a) Stage 2, 1D Swin: loss')

    ax = fig.add_subplot(gs[1])
    chance = np.log(batch - 1)
    ax.plot(d3.epoch, d3.supcon, color=c3, lw=1.6, marker='o', ms=3, label='SupCon (train)')
    ax.plot(d3.epoch, d3.ce, color=c1, lw=1.6, marker='o', ms=3, label='CE, smoothed (train)')
    ax.axhline(chance, color=INK2, lw=1, ls='--')
    ax.text(d3.epoch.max(), chance + 0.06, f'ln({batch - 1}) = {chance:.2f}: SupCon with no class signal',
            ha='right', va='bottom', fontsize=8, color=INK2)
    ax.set_ylim(0.8, 3.1)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ygrid(ax)
    ax.legend(loc='center right')
    ax.set_title('(b) Stage 1, fine-tune + SupCon: loss terms')

    ax = fig.add_subplot(gs[2])
    ax.plot(d1.epoch, d1.val_acc, color=c1, lw=1.2, label='Stage 2, 1D Swin')
    ax.plot(d3.epoch, d3.val_acc, color=c3, lw=1.6, marker='o', ms=3, label='Stage 1, fine-tune + SupCon')
    ax.axhline(40, color=INK2, lw=1, ls='--')
    ax.text(d1.epoch.max(), 41, 'majority class (40%)', ha='right', va='bottom', fontsize=8, color=INK2)
    ax.set_ylim(0, 70)
    ax.set_xlabel('epoch')
    ax.set_ylabel('validation accuracy (%)')
    ygrid(ax)
    ax.legend(loc='lower right')
    ax.set_title('(c) Validation accuracy on driver D6')
    fig.suptitle('Training curves from the logs (Stage 2: 200 epochs; Stage 1: stopped at epoch '
                 f'{int(d3.epoch.max())}, evaluated every 3 epochs)',
                 x=0.06, ha='left', fontsize=12, fontweight='semibold', y=0.98)
    return save(fig, out, 'fig6_training_curves.png')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=os.path.join(ROOT, 'runs', 'dinov3_vits16_run1', 'data'))
    p.add_argument('--logs', default=os.path.join(ROOT, 'runs', 'dinov3_vits16_run1', 'logs'))
    p.add_argument('--uah', default=os.path.join(os.path.dirname(ROOT), 'UAH-DRIVESET-v1', 'DRIVERSET_H'))
    p.add_argument('--out', default=os.path.join(HERE, 'figures'))
    p.add_argument('--only', type=int, nargs='*', help='only these figure numbers')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rep, wtr, wva, ktr, kva = load_meta(args.data)
    want = set(args.only or [1, 2, 3, 4, 5, 6])
    issues = 0
    if 1 in want:
        issues += fig1_dataset(rep, wtr, wva, args.out)
    if 2 in want:
        n, pick = fig2_samples(args.data, wva, kva, args.out)
        issues += n
    if 3 in want:
        issues += fig3_variability(wtr, wva, ktr, kva, args.out)
    if 4 in want:
        issues += fig4_sync(rep, args.uah, args.out)
    if 5 in want:
        issues += fig5_filtering(rep, wtr, wva, args.out)
    if 6 in want:
        issues += fig6_training(args.logs, args.out)
    print(f'total layout issues: {issues}')


if __name__ == '__main__':
    main()
