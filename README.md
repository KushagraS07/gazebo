#!/usr/bin/env python3
"""
rnn_fusion_v2.py — Denoising RNN matching paper exactly:
  KEY CHANGE: Train RNN using EKF output → Ground Truth
              (not synthetic Gaussian noise denoising)
  - Input:  EKF output [x, y, theta] (from ekf_fused_output.csv)
  - Target: RESIDUAL = Ground truth − EKF (see FIX below), not absolute GT
  - RNN learns to correct EKF drift toward real position

  FIX (root-caused via visual inspection of Fig 4 — RNN RMSE 1.317m vs
  EKF's own 0.648m, i.e. "correction" was making things WORSE): the model
  previously regressed straight to absolute ground-truth [x,y,theta] from
  each independent 10-step window, with literally nothing tying one
  window's prediction to the next (see smooth_trajectory()'s pre-existing
  docstring — that's exactly why raw output looped). EKF has a real
  kinematic motion model constraining consecutive states to be physically
  reachable; this RNN had no such constraint at all, so a single bad
  window could put the "corrected" point anywhere.
  Fix: predict the RESIDUAL (GT − EKF) at each window's target timestep
  instead of the absolute position, then reconstruct
  final = ekf_at_that_timestep + predicted_residual. This anchors every
  single prediction to the EKF's own (already smooth, kinematically
  constrained) estimate — even a bad/noisy residual prediction can only
  perturb around an already-plausible trajectory, it can't teleport.
  Residuals are normalized with their OWN train-only mean/std, computed
  PER SESSION (not pooled — see FIX #2 further down for why: pooling
  across sessions with very different EKF quality let one outlier
  session's scale distort the target for the others). Saved as a dict
  in rnn_residual_stats_per_session.npy (session -> {mean, std}).
  Theta residuals are wrapped to [-pi, pi] via atan2(sin,cos) before
  normalization/training and after reconstruction, to avoid a spurious
  near-2*pi "residual" when GT and EKF theta sit on opposite sides of the
  angle wraparound (same technique smooth_trajectory() already uses).
  - lr=1e-6, epochs=80, batch=32, window=10 (all from paper)
  - Train/Test split 80/20 (chronological — first 80% of trajectory for
    training, last 20% for testing; avoids overlapping-window leakage)
  - Validation loss tracked
  - Random seed=42
  - No SimpleEKF — reads ekf_fused_output.csv directly
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
import glob

# ── Config (all match paper) ──────────────────────────────────────────────────
LOG_DIR      = os.path.expanduser('~/ws_mobile/sensor_logs')

# FIX (multi-session support): this used to train on exactly one session
# (ekf_fused_output.csv), with an 80/20 CHRONOLOGICAL split of that single
# recording. That meant the test set was always just "whatever happened in
# the last ~20% of one specific drive" — not a random sample of driving
# conditions. Verified on real data: session 20260714_014615's test segment
# had a 40% higher mean speed and far less idle time than its train segment,
# so the model was evaluated on a motion regime it never saw in training —
# a domain-shift problem, not a model or hyperparameter problem. No amount
# of learning-rate or checkpoint tuning fixes that.
# Fix: train on MULTIPLE independent sessions. Sequences are built WITHIN
# each session separately (a sliding window never crosses a session
# boundary — that would create a fake, physically meaningless transition
# between two unrelated drives). Each session is independently split 80/20
# chronologically, then all sessions' train portions are combined and all
# sessions' test portions are combined. This way both train and test
# contain a mix of "early" and "late" behavior from multiple different
# drives, directly fixing the single-session domain-shift problem.
SESSIONS     = ['20260712_141740', '20260712_144326']
# FIX (session 20260714_014615 excluded — verified real, not a guess):
# raw IMU orientation for that session is flat at ~1e-7 rad the entire
# recording (imu_filtered_20260714_014615.csv, orient_z/orient_w), while
# ground truth shows the robot genuinely swinging through ~6 rad of
# heading and cmd_vel commanded real turns (angular_z up to 1.46 rad/s).
# The IMU topic never captured real rotation data for this one session —
# a dead/broken sensor log at recording time, not noise, not a harder
# trajectory, and not something EKF/RNN/LiDAR tuning can fix after the
# fact (there's no real signal in the file to recover). Confirmed this
# was the root cause of that session's 6.6m residual and its straight-
# line-instead-of-loop EKF trajectory (rnn_trajectory_comparison_
# 20260714_014615.png). Needs re-recording, not re-processing, if it's
# wanted back in the dataset — check the IMU plugin/topic timing in the
# ROS2 launch setup before the next recording session.

WINDOW_SIZE  = 10       # T=10 (paper Table 6), RNN input sequence window
SG_WINDOW    = 11       # Savitzky-Golay smoothing window for rnn_smoothed_output,
                         # matches the value used throughout preprocess_data.py
EPOCHS       = 80       # paper: 80 epochs
BATCH_SIZE   = 32       # paper: batch size 32
LR           = 1e-3     # FIX: paper states lr=1e-6, but verified on real data
                         # this leaves a from-scratch, randomly-initialized
                         # single-layer RNN (hidden=32) essentially unmoved
                         # after 80 epochs (~2880 total gradient steps) —
                         # train loss changed only ~4% over the entire run,
                         # and the model collapsed to predicting near-constant
                         # output (std ~10x smaller than the real trajectory),
                         # producing RMSE 4x WORSE than the raw EKF input it
                         # is supposed to correct. 1e-3 is a standard starting
                         # point for a model this small; verify train/val loss
                         # actually decreases substantially before trusting
                         # the resulting RMSE.
DROPOUT_P    = 0.2      # FIX (regularization): train loss -> ~0 while val loss
                         # bottoms out ~epoch 7 then rises/oscillates for the
                         # remaining ~73 epochs (confirmed visually in the
                         # loss curve figure) — classic overfitting, made worse
                         # by heavily-overlapping sliding-window training
                         # sequences from only 3 continuous recordings.
                         # Dropout on the RNN's output (before the final FC
                         # layer) is a standard, minimal fix targeting this
                         # exact symptom without changing window/stride
                         # structure or sample count.
WEIGHT_DECAY = 1e-4      # FIX (regularization): small L2 penalty via Adam's
                         # weight_decay, paired with dropout above — same
                         # target (reduce overfitting), different mechanism
                         # (constrains weight magnitude vs randomly zeroing
                         # activations). 1e-4 is a conservative starting
                         # value — small enough that it shouldn't meaningfully
                         # change results if the model wasn't overfitting,
                         # but should show up in the val loss curve if it is.
HIDDEN_SIZE  = 32
TRAIN_SPLIT  = 0.70     # 3-way split, applied PER SESSION chronologically:
                         # train / val / test = 0.70 / 0.15 / 0.15
VAL_SPLIT    = 0.15      # FIX (reviewer-flagged, verified real): previously
                         # only a 2-way 80/20 train/test split existed, and
                         # the "test" set was ALSO used every epoch to pick
                         # the best checkpoint (val_loss = criterion(model(
                         # X_test), y_test)) — meaning the reported "test"
                         # RMSE was computed on the exact same data used to
                         # select the model, an optimistic, non-generalizing
                         # number (classic validation-set leakage into the
                         # reported metric). Now: train picks weights, val
                         # picks the best checkpoint, test is touched ONLY
                         # once at the very end for reporting — genuinely
                         # held out through the entire training process.
SEED         = 42
# ──────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# RNN MODEL
# ══════════════════════════════════════════════════════════════════════════════

def smooth_trajectory(pred, sg_window=SG_WINDOW):
    """
    Apply Savitzky-Golay smoothing to a raw [x,y,theta] RNN output
    trajectory. FIX: each window's RNN prediction is independent (no
    continuity constraint between consecutive outputs), so plotting raw
    predictions as a trajectory produces visibly erratic, looping paths
    even when pointwise RMSE looks reasonable (RMSE measures point-to-point
    distance error, not path coherence/smoothness). This reuses the same
    S-G filtering technique already used throughout preprocess_data.py
    (consistent methodology, not a new invented technique). Theta is
    smoothed via sin/cos and reconstructed with atan2 to correctly handle
    angle wraparound. Single source of truth — used both by
    train_denoising_rnn()'s own inference loop and by noise_evaluation.py,
    so "rnn_smoothed_output" means the same thing everywhere it's produced.
    """
    n = len(pred)
    window = min(sg_window, n if n % 2 == 1 else n - 1)
    if window < 5:  # need enough points for a degree-3 polynomial fit
        return pred.copy()
    smoothed = pred.copy()
    smoothed[:, 0] = savgol_filter(pred[:, 0], window, 3)
    smoothed[:, 1] = savgol_filter(pred[:, 1], window, 3)
    sin_t = savgol_filter(np.sin(pred[:, 2]), window, 3)
    cos_t = savgol_filter(np.cos(pred[:, 2]), window, 3)
    smoothed[:, 2] = np.arctan2(sin_t, cos_t)
    return smoothed


def compute_residual(gt, ekf):
    """
    Residual = GT - EKF, elementwise, with theta (column 2) wrapped to
    [-pi, pi] via atan2(sin,cos). Single source of truth for this math —
    used both when building training targets below and by
    noise_evaluation.py's run_real_rnn(), so "residual" means the exact
    same thing everywhere it's computed. Without the wrap, a GT theta of
    +3.13 rad and an EKF theta of -3.13 rad (physically ~0.003 rad apart)
    would produce a raw residual of ~6.26 rad instead of ~0.003 rad — a
    huge, wrong training target purely from the angle wraparound.
    """
    res = gt - ekf
    res = res.copy()
    res[:, 2] = np.arctan2(np.sin(res[:, 2]), np.cos(res[:, 2]))
    return res


def run_rnn_inference(model, mean, std, res_mean, res_std, ekf_data, window=WINDOW_SIZE):
    """
    Reusable RNN inference — applies the exact trained model + normalization
    stats to an arbitrary [x,y,theta] EKF trajectory array. Returns
    (pred, valid_start_index) where pred has len(ekf_data)-window rows and
    valid_start_index=window is how far into the original array the first
    prediction corresponds to (matches create_sequences' windowing).
    Factored out so other scripts (e.g. noise_evaluation.py) can reuse the
    real trained model instead of a separate reimplementation.

    FIX (residual-learning reconstruction): the model output is now a
    normalized RESIDUAL, not an absolute normalized position — see the
    module docstring for why. Reconstruction: take the model's predicted
    residual, un-normalize it with the residual's own stats (res_mean/
    res_std, NOT the position mean/std), then add it onto the EKF's own
    raw value at that exact target timestep (ekf_data[window:window+n]) —
    the same anchor index create_sequences() itself used to build targets
    during training, so inference and training stay consistent.
    """
    ekf_norm = (ekf_data - mean) / std
    dummy_targets = np.zeros_like(ekf_norm)  # not used for inference, only
                                              # needed to satisfy create_sequences'
                                              # signature (targets are discarded below)
    X_full, _ = create_sequences(ekf_norm, dummy_targets, window)
    model.eval()
    with torch.no_grad():
        pred_residual_norm = model(torch.tensor(X_full)).numpy()
    pred_residual = pred_residual_norm * res_std + res_mean
    n_pred = len(pred_residual)
    anchor = ekf_data[window:window + n_pred]
    pred = anchor + pred_residual
    pred[:, 2] = np.arctan2(np.sin(pred[:, 2]), np.cos(pred[:, 2]))
    return pred, window


class DenoisingRNN(nn.Module):
    """
    RNN that maps EKF trajectory → Ground Truth trajectory.
    Input/output: 3D state [x, y, theta].
    FIX: added dropout on the RNN's output (before the final FC layer) —
    nn.RNN's own `dropout` kwarg only applies BETWEEN stacked layers and
    does nothing with num_layers=1 (our case), so it has to go here
    instead. Targets the overfitting confirmed in the loss curve (val loss
    rises/oscillates after ~epoch 7 while train loss keeps falling).
    Disabled automatically during model.eval() (inference/checkpointing),
    same as any standard nn.Dropout usage.
    """
    def __init__(self, input_size=3, hidden_size=HIDDEN_SIZE, output_size=3,
                 dropout_p=DROPOUT_P):
        super().__init__()
        self.rnn     = nn.RNN(input_size, hidden_size,
                              num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_ekf_output(session: str) -> pd.DataFrame:
    path = os.path.join(LOG_DIR, f'ekf_fused_output_{session}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ekf_fused_output_{session}.csv not found. "
            f"Run ekf_fusion.py {session} first.")
    df = pd.read_csv(path)
    print(f"  Loaded {os.path.basename(path)} — {len(df)} rows")
    return df


def load_ground_truth_aligned(ekf_timestamps: np.ndarray, session: str):
    """
    Load ground truth for a specific session and align to EKF timestamps
    via interpolation. This ensures row i of EKF matches row i of GT by
    actual time, not by row index (which would be wrong if rates differ).
    Always returns (gt_data, mask) or None — never a bare array — so the
    caller's tuple-unpacking is always safe.
    """
    # Prefer the canonical pipeline output: already cleaned (tf-fallback rows
    # dropped) and resampled onto the shared timeline by preprocess_data.py.
    # Don't rely on ctime ordering across raw/stage/filtered variants.
    filtered_path = os.path.join(LOG_DIR, f'ground_truth_filtered_{session}.csv')
    if os.path.exists(filtered_path):
        path  = filtered_path
        gt_df = pd.read_csv(path)
        print(f"  Loaded ground truth: {os.path.basename(path)} "
              f"({len(gt_df)} rows) — canonical, cleaned")
    else:
        files = glob.glob(os.path.join(LOG_DIR, f'ground_truth_{session}.csv'))
        files = [f for f in files if 'stage' not in f and 'filtered' not in f]
        if not files:
            print(f"  Warning: no ground truth found for session {session} — "
                  "using odom as reference. Every RMSE number that follows is "
                  "NOT a real accuracy measurement. Run "
                  f"'preprocess_data.py {session}' first.")
            return None
        path  = files[0]
        gt_df = pd.read_csv(path)
        print(f"  Loaded ground truth: {os.path.basename(path)} ({len(gt_df)} rows) "
              f"— RAW, unfiltered (run preprocess_data.py first for cleaner GT)")
        if 'child_frame_id' in gt_df.columns:
            before = len(gt_df)
            gt_df = gt_df[~gt_df['child_frame_id'].isin(['base_footprint', 'base_link'])]
            if len(gt_df) < before:
                print(f"    Dropped {before - len(gt_df)} /tf-fallback row(s) from raw GT")

    # Build GT timestamp
    if 'timestamp' in gt_df.columns:
        gt_t = gt_df['timestamp'].values.astype(float)
    elif 'time_sec' in gt_df.columns and 'time_nsec' in gt_df.columns:
        gt_t = gt_df['time_sec'].values + gt_df['time_nsec'].values * 1e-9
    else:
        # FIX: previously returned a bare ndarray here while every other path
        # returns (data, mask) — caller always does `gt_data, mask = result`,
        # which would crash the moment this branch was ever hit. Now
        # consistent with the rest of the function.
        print("  Warning: no timestamp in GT — falling back to row alignment")
        n = len(ekf_timestamps)
        gt_df = gt_df.iloc[:n].reset_index(drop=True)
        gt_xy = gt_df[['gt_x', 'gt_y']].values.astype(np.float32)
        theta = np.zeros((len(gt_df), 1), dtype=np.float32)
        aligned_gt = np.hstack([gt_xy, theta])
        mask = np.zeros(n, dtype=bool)
        mask[:len(gt_df)] = True
        return aligned_gt, mask

    # Interpolate GT onto EKF time grid
    # Only use overlapping time range
    t_start = max(ekf_timestamps[0],  gt_t[0])
    t_end   = min(ekf_timestamps[-1], gt_t[-1])
    mask    = (ekf_timestamps >= t_start) & (ekf_timestamps <= t_end)

    ekf_t_aligned = ekf_timestamps[mask]
    n_aligned     = mask.sum()

    print(f"  Timestamp alignment: {n_aligned}/{len(ekf_timestamps)} EKF rows overlap with GT")

    gt_x     = np.interp(ekf_t_aligned, gt_t, gt_df['gt_x'].values).astype(np.float32)
    gt_y     = np.interp(ekf_t_aligned, gt_t, gt_df['gt_y'].values).astype(np.float32)

    if 'gt_orient_z' in gt_df.columns and 'gt_orient_w' in gt_df.columns:
        gt_oz    = np.interp(ekf_t_aligned, gt_t, gt_df['gt_orient_z'].values)
        gt_ow    = np.interp(ekf_t_aligned, gt_t, gt_df['gt_orient_w'].values)
        gt_theta = (2.0 * np.arctan2(gt_oz, gt_ow)).astype(np.float32)
    else:
        gt_theta = np.zeros(n_aligned, dtype=np.float32)

    aligned_gt = np.column_stack([gt_x, gt_y, gt_theta])
    return aligned_gt, mask


def create_sequences(inputs: np.ndarray, targets: np.ndarray, window: int):
    X, y = [], []
    for i in range(len(inputs) - window):
        X.append(inputs[i:i + window])
        y.append(targets[i + window])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def compute_rmse(pred, true):
    err  = np.sqrt(((pred[:, 0] - true[:, 0])**2 +
                    (pred[:, 1] - true[:, 1])**2))
    return float(np.sqrt(np.mean(err**2))), float(np.mean(err)), float(np.std(err))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def train_denoising_rnn():
    print("\n" + "="*55)
    print("  RNN FUSION V2 — EKF → Ground Truth correction")
    print(f"  Sessions: {', '.join(SESSIONS)}")
    print("="*55)

    # ── FIX (multi-session support): the helper functions below already took
    # a `session` argument, but this function used to call them with no
    # arguments at all — load_ekf_output() and load_ground_truth_aligned(
    # ekf_timestamps) — which would raise TypeError immediately (both
    # require `session`). That's because the single-session body was never
    # actually rewritten to loop over SESSIONS after the helpers were
    # updated. Fixed below: loop over every session, load + align each one
    # independently, and never let a sliding window span two sessions.
    per_session = {}   # session -> dict(ekf_data, gt_data, ekf_timestamps, ekf_df)
    n_features = 3

    for session in SESSIONS:
        print(f"\n  --- Loading session {session} ---")
        ekf_df = load_ekf_output(session)
        n = len(ekf_df)

        if 'ekf_theta' in ekf_df.columns:
            ekf_data = ekf_df[['ekf_x', 'ekf_y', 'ekf_theta']].values.astype(np.float32)
        else:
            print("  Warning: ekf_theta missing — padding with zeros")
            ekf_xy   = ekf_df[['ekf_x', 'ekf_y']].values.astype(np.float32)
            ekf_data = np.hstack([ekf_xy, np.zeros((n, 1), dtype=np.float32)])

        ekf_timestamps = ekf_df['timestamp'].values.astype(float) \
            if 'timestamp' in ekf_df.columns else np.arange(n) / 50.0

        result = load_ground_truth_aligned(ekf_timestamps, session)
        if result is not None:
            gt_data, mask = result
            ekf_data       = ekf_data[mask]
            ekf_timestamps = ekf_timestamps[mask]
        else:
            print(f"\n  ⚠️  WARNING: session {session} has no ground truth — "
                  "training against odometry instead. This session's data "
                  "will teach the model to reproduce odometry drift, not "
                  "correct it. Strongly consider excluding it.")
            odom_xy = ekf_df[['odom_x', 'odom_y']].values.astype(np.float32)
            odom_xy = odom_xy[:len(ekf_data)]
            gt_data = np.hstack([odom_xy,
                                  np.zeros((len(ekf_data), 1), dtype=np.float32)])

        print(f"  Session {session}: {len(ekf_data)} aligned rows")
        per_session[session] = dict(ekf_data=ekf_data, gt_data=gt_data,
                                     ekf_timestamps=ekf_timestamps, ekf_df=ekf_df)

    # ── FIX (reviewer-flagged, verified real): normalization stats were
    # previously fit on ALL data (train+val+test combined, all sessions),
    # meaning test-set statistics leaked into the mean/std used to
    # normalize the training data itself. Fixed below: split each
    # session's RAW (pre-normalization) data into train/val/test FIRST,
    # then fit mean/std on the concatenated TRAIN portion only, then
    # normalize every split (including val/test) using those train-only
    # stats — standard practice, and the only way the reported test RMSE
    # actually reflects generalization to unseen data.
    #
    # FIX #2 (found via real run — session 20260712_141740's train/val
    # split showed held-out TEST loss ~3x the VAL loss used for
    # checkpoint selection): a plain contiguous 70/15/15 chronological
    # slice puts test ALWAYS at the tail of every session. This project
    # already diagnosed and fixed almost this exact problem once before
    # (the single-session train/test domain-shift bug — a session's tail
    # segment having a different speed/idle profile than the rest, so a
    # model never trained on that regime tested badly on it). A
    # contiguous 3-way split just rebuilds a smaller version of the same
    # issue into val/test specifically. Fixed by dividing each session
    # into N_BLOCKS contiguous chronological blocks and assigning blocks
    # to train/val/test on a repeating pattern (not "all early blocks to
    # train, all late blocks to test") — every split now gets a mix of
    # early/middle/late behavior from every session, while each
    # individual block stays internally contiguous (so sequences within
    # a block are still real, non-overlapping-with-other-splits windows).
    N_BLOCKS = 20
    # Pattern repeats every 7 blocks: 5 train, 1 val, 1 test (~71/14/14%,
    # close to the target 70/15/15) — repeating every 7 blocks means val
    # and test blocks recur roughly 3 times across a 20-block session,
    # spread through its full chronological length rather than
    # clustered anywhere.
    def block_split_label(block_idx):
        r = block_idx % 7
        if r == 3:
            return 'val'
        elif r == 6:
            return 'test'
        return 'train'

    per_session_split = {}
    for session, v in per_session.items():
        n_total = len(v['ekf_data'])
        edges = np.linspace(0, n_total, N_BLOCKS + 1).astype(int)
        blocks = {'train': [], 'val': [], 'test': []}
        for b in range(N_BLOCKS):
            lo, hi = edges[b], edges[b + 1]
            if hi <= lo:
                continue
            label = block_split_label(b)
            blocks[label].append((v['ekf_data'][lo:hi], v['gt_data'][lo:hi]))
        per_session_split[session] = blocks

    all_train_ekf = np.concatenate(
        [chunk[0] for s in SESSIONS for chunk in per_session_split[s]['train']],
        axis=0)
    mean = all_train_ekf.mean(axis=0)
    std  = all_train_ekf.std(axis=0) + 1e-6
    np.save(os.path.join(LOG_DIR, 'rnn_norm_stats.npy'), np.array([mean, std]))
    print(f"\n  Normalization stats fit on TRAIN split ONLY across "
          f"{len(SESSIONS)} sessions ({len(all_train_ekf)} rows) — "
          f"val/test never influence these values")

    # FIX #2 (real-data run revealed pooled residual stats were badly
    # wrong — see module docstring for the full diagnosis): residual
    # mean/std are now computed PER SESSION, each fit on ONLY that
    # session's own train-split residuals, instead of one pooled value
    # across all 3 sessions. A session with unusually large EKF error
    # (verified: session 014615's residual max is 6.6m vs 144326's 0.9m)
    # can no longer distort the normalized target scale for other,
    # better-behaved sessions.
    res_stats_per_session = {}
    for s in SESSIONS:
        train_ekf_s = np.concatenate(
            [chunk[0] for chunk in per_session_split[s]['train']], axis=0)
        train_gt_s = np.concatenate(
            [chunk[1] for chunk in per_session_split[s]['train']], axis=0)
        residual_s = compute_residual(train_gt_s, train_ekf_s)
        res_stats_per_session[s] = dict(
            mean=residual_s.mean(axis=0),
            std=residual_s.std(axis=0) + 1e-6,
        )
        print(f"  Session {s} residual stats (TRAIN split only) — "
              f"mean={res_stats_per_session[s]['mean']}, "
              f"std={res_stats_per_session[s]['std']}")
    np.save(os.path.join(LOG_DIR, 'rnn_residual_stats_per_session.npy'),
            res_stats_per_session, allow_pickle=True)
    print(f"  Split method: {N_BLOCKS} interleaved chronological blocks per "
          f"session (train/val/test all sample early+middle+late behavior, "
          f"not just contiguous slices)")

    # ── Build sequences PER SESSION PER BLOCK, then combine into splits ─────
    # A sliding window must never cross a session boundary OR a block
    # boundary (window would leak across a train/val/test seam) — so
    # create_sequences() is called separately on EACH block, not on a
    # concatenation of blocks (blocks assigned to the same split aren't
    # chronologically adjacent to each other anymore, by design).
    X_train_list, y_train_list = [], []
    X_val_list,   y_val_list   = [], []
    X_test_list,  y_test_list  = [], []

    for session in SESSIONS:
        blocks = per_session_split[session]
        res_mean_s = res_stats_per_session[session]['mean']
        res_std_s  = res_stats_per_session[session]['std']
        counts = {'train': 0, 'val': 0, 'test': 0}
        for split_name, chunks in blocks.items():
            for ekf_raw, gt_raw in chunks:
                if len(ekf_raw) <= WINDOW_SIZE:
                    continue  # block too short to yield even one sequence
                ekf_norm      = (ekf_raw - mean) / std
                # FIX (per-session residual normalization): target is the
                # residual normalized with THIS session's own stats, not
                # a pooled global one — see FIX #2 above.
                residual_raw  = compute_residual(gt_raw, ekf_raw)
                residual_norm = (residual_raw - res_mean_s) / res_std_s
                X, y = create_sequences(ekf_norm, residual_norm, WINDOW_SIZE)
                counts[split_name] += len(X)
                if split_name == 'train':
                    X_train_list.append(X); y_train_list.append(y)
                elif split_name == 'val':
                    X_val_list.append(X);   y_val_list.append(y)
                else:
                    X_test_list.append(X);  y_test_list.append(y)

        print(f"  Session {session}: {counts['train']} train / "
              f"{counts['val']} val / {counts['test']} test sequences "
              f"(from {N_BLOCKS} interleaved blocks)")

    X_train = torch.tensor(np.concatenate(X_train_list, axis=0))
    y_train = torch.tensor(np.concatenate(y_train_list, axis=0))
    X_val   = torch.tensor(np.concatenate(X_val_list,   axis=0))
    y_val   = torch.tensor(np.concatenate(y_val_list,   axis=0))
    X_test  = torch.tensor(np.concatenate(X_test_list,  axis=0))
    y_test  = torch.tensor(np.concatenate(y_test_list,  axis=0))

    print(f"\n  Combined: Train {len(X_train)}  |  Val {len(X_val)}  |  "
          f"Test {len(X_test)}  (pooled across {len(SESSIONS)} sessions, "
          f"test held out entirely until final reporting)")

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = DenoisingRNN(input_size=n_features, output_size=n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    print(f"\n  Training: epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LR}")
    print(f"  {'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}")
    print(f"  {'-'*35}")

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state = None
    best_epoch = -1
    SMOOTH_WINDOW = 5   # moving-average window for robust best-epoch selection
    PATIENCE      = 10  # FIX (reviewer-flagged, real gap): training previously
                         # always ran the full EPOCHS regardless of how long
                         # val loss had been failing to improve — best-
                         # checkpoint SAVING already existed (so a bad final
                         # epoch was never used for inference), but that's a
                         # different thing from STOPPING early to save
                         # compute once it's clear the model isn't improving.
                         # Stops if smoothed val loss hasn't beaten its best
                         # in PATIENCE consecutive epochs.
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        perm       = torch.randperm(len(X_train))
        epoch_loss = 0.0
        n_batches  = 0

        for i in range(0, len(X_train), BATCH_SIZE):
            ib      = perm[i:i + BATCH_SIZE]
            bx, by  = X_train[ib], y_train[ib]
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            # FIX (reviewer-flagged, real gap): no gradient clipping existed
            # — standard, low-risk stability measure for RNN training
            # (RNNs are prone to occasional gradient spikes even on short
            # sequences like WINDOW_SIZE=10). max_norm=1.0 is a
            # conservative, commonly-used default.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        avg_train = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            # FIX: checkpoint selection now uses the genuinely held-out
            # VAL split, never X_test — see VAL_SPLIT constant docstring.
            val_loss = criterion(model(X_val), y_val).item()

        train_losses.append(avg_train)
        val_losses.append(val_loss)

        if epoch + 1 >= SMOOTH_WINDOW:
            smoothed_val = float(np.mean(val_losses[-SMOOTH_WINDOW:]))
            if smoothed_val < best_val_loss:
                best_val_loss = smoothed_val
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        if (epoch + 1) % 10 == 0:
            print(f"  {epoch+1:>6}  {avg_train:>12.6f}  {val_loss:>12.6f}")

        if epoch + 1 >= SMOOTH_WINDOW and epochs_no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1} — no smoothed-val-loss "
                  f"improvement in {PATIENCE} epochs (best was epoch {best_epoch})")
            break

    if best_state is None:
        best_state = model.state_dict()
        best_epoch = EPOCHS
        best_val_loss = val_losses[-1]

    print(f"\n  Best smoothed ({SMOOTH_WINDOW}-epoch avg) val loss: "
          f"{best_val_loss:.6f} at epoch {best_epoch} "
          f"(restoring these weights for inference, not the final epoch's)")
    model.load_state_dict(best_state)

    # FIX: this is the ONLY place X_test/y_test are touched during the
    # entire training process — a genuine, held-out generalization check,
    # not the same data used to pick the checkpoint above (see VAL_SPLIT
    # docstring for why the old 2-way split's "test" number was optimistic).
    model.eval()
    with torch.no_grad():
        final_test_loss = criterion(model(X_test), y_test).item()
    print(f"  Held-out TEST loss (never used for checkpoint selection): "
          f"{final_test_loss:.6f}")

    pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
    }).to_csv(os.path.join(LOG_DIR, 'rnn_training_curve.csv'), index=False)
    print(f"  Saved: rnn_training_curve.csv (per-epoch train/val loss)")

    torch.save(model.state_dict(),
               os.path.join(LOG_DIR, 'rnn_model_denoising.pth'))
    print(f"  Saved: rnn_model_denoising.pth (best-smoothed-val-loss "
          f"checkpoint, epoch {best_epoch}, trained on {len(SESSIONS)} sessions)")

    # ── Inference PER SESSION → corrected trajectory + per-session metrics ──
    # Run inference separately per session (not concatenated) since
    # concatenating raw trajectories from different drives would draw a
    # meaningless connecting line between the end of one session and the
    # start of another. Report both per-session and pooled overall RMSE.
    model.eval()
    all_err = []
    session_results = {}

    for session, v in per_session.items():
        res_mean_s = res_stats_per_session[session]['mean']
        res_std_s  = res_stats_per_session[session]['std']
        pred, window = run_rnn_inference(model, mean, std, res_mean_s, res_std_s,
                                          v['ekf_data'])

        n_pred     = len(pred)
        gt_aligned = v['gt_data'][window:window + n_pred]
        raw_rmse, raw_mae, raw_std = compute_rmse(pred, gt_aligned)

        # FIX: rnn_smoothed_output was previously just an unmodified copy
        # of rnn_output — see smooth_trajectory() docstring for the full
        # explanation of why this matters (erratic/looping plotted paths).
        if n_pred < 5:
            print(f"  Warning: session {session} too short ({n_pred} rows) "
                  f"for S-G smoothing — using raw (unsmoothed) output")
        smoothed = smooth_trajectory(pred)

        rmse, mae, std_err = compute_rmse(smoothed, gt_aligned)
        err = np.sqrt((smoothed[:, 0]-gt_aligned[:, 0])**2 +
                      (smoothed[:, 1]-gt_aligned[:, 1])**2)
        all_err.append(err)

        timestamps = v['ekf_timestamps'][WINDOW_SIZE:WINDOW_SIZE + n_pred]
        raw_df = pd.DataFrame({
            'timestamp': timestamps,
            'rnn_x':     pred[:, 0],
            'rnn_y':     pred[:, 1],
            'rnn_theta': pred[:, 2],
        })
        smoothed_df = pd.DataFrame({
            'timestamp': timestamps,
            'rnn_x':     smoothed[:, 0],
            'rnn_y':     smoothed[:, 1],
            'rnn_theta': smoothed[:, 2],
        })
        raw_df.to_csv(os.path.join(LOG_DIR, f'rnn_output_{session}.csv'), index=False)
        smoothed_df.to_csv(os.path.join(LOG_DIR, f'rnn_smoothed_output_{session}.csv'), index=False)

        session_results[session] = dict(rmse=rmse, mae=mae, std=std_err,
                                         pred=smoothed, gt_aligned=gt_aligned,
                                         ekf_df=v['ekf_df'])
        print(f"\n  ── Session {session} — RNN Accuracy vs Ground Truth ──")
        print(f"     RMSE (raw, unsmoothed) : {raw_rmse:.4f} m")
        print(f"     RMSE (S-G smoothed)    : {rmse:.4f} m  <- used downstream")
        print(f"     MAE  : {mae:.4f} m")
        print(f"     Std  : {std_err:.4f} m")
        print(f"     Saved: rnn_output_{session}.csv (raw) + "
              f"rnn_smoothed_output_{session}.csv (S-G smoothed, {n_pred} rows)")

    overall_err = np.concatenate(all_err)
    overall_rmse = float(np.sqrt(np.mean(overall_err**2)))
    overall_mae  = float(np.mean(overall_err))
    overall_std  = float(np.std(overall_err))
    print(f"\n  ── POOLED (all {len(SESSIONS)} sessions) — RNN Accuracy ──")
    print(f"     RMSE : {overall_rmse:.4f} m  (paper EKF-RNN: 0.042/0.067/0.102 m)")
    print(f"     MAE  : {overall_mae:.4f} m")
    print(f"     Std  : {overall_std:.4f} m  (paper RNN std: 0.0527 m)")

    metrics_rows = [{'session': s, 'rmse': r['rmse'], 'mae': r['mae'], 'std': r['std']}
                     for s, r in session_results.items()]
    metrics_rows.append({'session': 'POOLED', 'rmse': overall_rmse,
                          'mae': overall_mae, 'std': overall_std})
    pd.DataFrame(metrics_rows).to_csv(
        os.path.join(LOG_DIR, 'rnn_metrics.csv'), index=False)
    print(f"  Saved: rnn_metrics.csv (per-session + pooled)")

    # ── Plots: one shared loss curve, one trajectory plot PER SESSION ───────
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_losses, label='Train Loss', color='blue')
    ax.plot(val_losses,   label='Val Loss',   color='orange', linestyle='--')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
    ax.set_title('RNN Training & Validation Loss (paper Fig 11)')
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, 'rnn_denoising_loss_curve.png'), dpi=150)
    print(f"  Saved: rnn_denoising_loss_curve.png")

    for session, r in session_results.items():
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(r['ekf_df']['ekf_x'].values, r['ekf_df']['ekf_y'].values,
                label='EKF output', color='orange', alpha=0.6, linewidth=1)
        ax.plot(r['pred'][:, 0], r['pred'][:, 1],
                label='RNN corrected', color='blue', linewidth=2)
        ax.plot(r['gt_aligned'][:, 0], r['gt_aligned'][:, 1],
                label='Ground Truth', color='green', linewidth=1, linestyle='--')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_title(f'EKF → RNN Correction — session {session}')
        ax.legend(); ax.grid(True); ax.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(LOG_DIR, f'rnn_trajectory_comparison_{session}.png'),
                    dpi=150)
        plt.close(fig)
    print(f"  Saved: rnn_trajectory_comparison_<session>.png for each session")

    print("\n✅ RNN fusion v2 complete!")
    print("   Next: lidar_fusion_v2.py (will need updating to read per-session")
    print("   rnn_smoothed_output_<session>.csv files, same as this script)")


if __name__ == '__main__':
    train_denoising_rnn()
