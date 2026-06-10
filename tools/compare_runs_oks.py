#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""Cross-run OKS comparison: common instances only.

Loads multiple prediction bundles (each a directory with ``manifest.json``
and ``frames.json`` produced by ``tools/benchmark_e2e.py``) and reports the
average OKS per run **restricted to GT instances that every selected run
matched** (OKS >= threshold).

Comparing accuracy on the common subset removes the confounding effect of
recall differences: a run that detects fewer people cannot inflate its OKS
average by skipping hard instances.

Usage – individual run directories
-----------------------------------
    python tools/compare_runs_oks.py \\
        benchmark/predictions/20260528_coco_e2e/ViTPose-base \\
        benchmark/predictions/20260528_coco_e2e/HRNet-w48 \\
        benchmark/predictions/20260528_coco_e2e/RTMPose-l

Usage – discover all runs in a group directory
-----------------------------------------------
    python tools/compare_runs_oks.py \\
        --group benchmark/predictions/20260528_coco_e2e

Notes
-----
* All prediction bundles must come from the same dataset (same annotation
  file) so that GT instance ordering is identical across runs.
* The stored matches in ``frames.json`` were computed at OKS >= 0.5 by
  default.  Passing ``--oks-thr`` lower than that stored threshold has no
  effect (those matches were discarded at prediction time).
"""

import argparse
import json
import os
import os.path as osp
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ── Bundle I/O ────────────────────────────────────────────────────────────────

def _load_bundle(run_dir: str) -> Tuple[dict, List[dict]]:
    """Load manifest and frames from a prediction bundle directory."""
    manifest_path = osp.join(run_dir, 'manifest.json')
    frames_path = osp.join(run_dir, 'frames.json')
    if not osp.isfile(manifest_path):
        raise FileNotFoundError(f'manifest.json not found in: {run_dir}')
    if not osp.isfile(frames_path):
        raise FileNotFoundError(f'frames.json not found in: {run_dir}')
    with open(manifest_path, encoding='utf-8') as f:
        manifest = json.load(f)
    with open(frames_path, encoding='utf-8') as f:
        frames = json.load(f)
    return manifest, frames


def _run_label(manifest: dict, run_dir: str) -> str:
    """Human-readable label derived from manifest metadata."""
    model_name = manifest.get('model_name') or ''
    model_variant = manifest.get('model_variant') or ''
    if model_name and model_variant:
        return f'{model_name}-{model_variant}'
    if model_name:
        return model_name
    return osp.basename(osp.abspath(run_dir))


def _stored_match_thr(manifest: dict) -> float:
    """Return the OKS threshold used when building the stored matches."""
    return float(manifest.get('badcase_defaults', {}).get('thr', 0.5))


# ── Core computation ─────────────────────────────────────────────────────────

def _build_matched_set(
    frames: List[dict],
    oks_thr: float,
) -> Dict[int, Dict[int, float]]:
    """Build ``{img_id: {gt_idx: oks}}`` for matches at or above *oks_thr*.

    Each entry in ``metrics.matches`` already satisfied the prediction-time
    threshold (default 0.5). We optionally filter further here.

    Args:
        frames: Loaded frame records from frames.json.
        oks_thr: Minimum OKS to keep a match.

    Returns:
        Nested dict mapping ``img_id → gt_idx → oks``.
    """
    matched: Dict[int, Dict[int, float]] = {}
    for frame in frames:
        img_id = frame['img_id']
        for m in frame.get('metrics', {}).get('matches', []):
            if m['oks'] >= oks_thr:
                matched.setdefault(img_id, {})[m['gt_idx']] = float(m['oks'])
    return matched


def compare_runs(
    run_dirs: List[str],
    oks_thr: float = 0.5,
    verbose: bool = True,
) -> dict:
    """Compute per-run OKS statistics over commonly matched GT instances.

    Algorithm
    ---------
    1. For each run, build a ``{img_id: {gt_idx: oks}}`` map.
    2. Restrict to images present in *all* runs.
    3. For each such image find the intersection of matched ``gt_idx`` sets.
    4. Collect the OKS values for those common instances per run.
    5. Report mean, median, and std of OKS per run.

    Args:
        run_dirs: List of prediction bundle directories.
        oks_thr: OKS threshold; must be >= the threshold used at prediction
            time (typically 0.5).
        verbose: Print loading progress.

    Returns:
        Dict with keys ``n_common_instances``, ``n_common_images``,
        ``oks_thr``, ``datasets``, and ``per_run`` (list sorted by
        ``avg_oks`` descending).

    Raises:
        ValueError: Fewer than 2 run directories provided.
    """
    if len(run_dirs) < 2:
        raise ValueError('At least 2 run directories are required.')

    # ── Load and validate ────────────────────────────────────────────────────
    runs = []
    for d in run_dirs:
        manifest, frames = _load_bundle(d)
        stored_thr = _stored_match_thr(manifest)
        if oks_thr < stored_thr - 1e-6:
            print(
                f'  Warning: --oks-thr {oks_thr} is below the stored match '
                f'threshold {stored_thr} for run {_run_label(manifest, d)!r}. '
                f'Matches below {stored_thr} were discarded at prediction time '
                f'and cannot be recovered. Using {stored_thr} instead.',
                file=sys.stderr,
            )
            effective_thr = stored_thr
        else:
            effective_thr = oks_thr

        label = _run_label(manifest, d)
        matched = _build_matched_set(frames, effective_thr)
        img_ids: Set[int] = {f['img_id'] for f in frames}

        n_matched_total = sum(len(v) for v in matched.values())
        runs.append({
            'dir': d,
            'label': label,
            'manifest': manifest,
            'matched': matched,
            'img_ids': img_ids,
            'num_frames': len(frames),
            'n_matched_total': n_matched_total,
        })
        if verbose:
            print(f'  {label!r}: {len(frames)} frames, '
                  f'{n_matched_total} matched instances '
                  f'(OKS >= {effective_thr:.2f})')

    # ── Common images ────────────────────────────────────────────────────────
    common_img_ids: Set[int] = runs[0]['img_ids']
    for r in runs[1:]:
        common_img_ids = common_img_ids & r['img_ids']

    if verbose:
        print(f'\n  Common images across all {len(runs)} runs: '
              f'{len(common_img_ids)}')

    # ── Intersection of matched GT instances ─────────────────────────────────
    # Identify every (img_id, gt_idx) that was matched in ALL runs.
    # Collect the per-run OKS for those instances.
    #
    # The GT instance order within each image is deterministic (same annotation
    # file, same loading code), so (img_id, gt_idx) reliably refers to the
    # same physical person across runs on the same dataset.
    common_oks: Dict[Tuple[int, int], List[float]] = {}

    for img_id in common_img_ids:
        # Per-run gt_idx sets for this image
        gt_sets = [set(r['matched'].get(img_id, {}).keys()) for r in runs]
        common_gt = gt_sets[0]
        for s in gt_sets[1:]:
            common_gt &= s

        for gt_idx in common_gt:
            oks_per_run = [r['matched'][img_id][gt_idx] for r in runs]
            common_oks[(img_id, gt_idx)] = oks_per_run

    n_common = len(common_oks)
    if verbose:
        print(f'  Common GT instances matched in all runs: {n_common}')

    # ── Per-run statistics ───────────────────────────────────────────────────
    per_run_stats = []
    for i, r in enumerate(runs):
        if n_common > 0:
            oks_values = np.array([v[i] for v in common_oks.values()],
                                  dtype=np.float64)
            avg_oks = float(oks_values.mean())
            median_oks = float(np.median(oks_values))
            std_oks = float(oks_values.std())
        else:
            avg_oks = median_oks = std_oks = 0.0

        per_run_stats.append({
            'label': r['label'],
            'dir': osp.abspath(r['dir']),
            'avg_oks': avg_oks,
            'median_oks': median_oks,
            'std_oks': std_oks,
            'n_matched_total': r['n_matched_total'],
            'num_frames': r['num_frames'],
            'test_dataset': r['manifest'].get('test_dataset', ''),
        })

    per_run_stats.sort(key=lambda x: x['avg_oks'], reverse=True)

    datasets = sorted({r['manifest'].get('test_dataset', '') for r in runs}
                      - {''})
    return {
        'oks_thr': oks_thr,
        'n_common_images': len(common_img_ids),
        'n_common_instances': n_common,
        'datasets': datasets,
        'per_run': per_run_stats,
    }


# ── Output ────────────────────────────────────────────────────────────────────

def _print_results(results: dict) -> None:
    sep = '=' * 76
    print(f'\n{sep}')
    print('  Cross-Run OKS Comparison  (common instances only)')
    print(sep)
    print(f'  OKS threshold      : {results["oks_thr"]:.2f}')
    print(f'  Dataset(s)         : {", ".join(results["datasets"]) or "unknown"}')
    print(f'  Common images      : {results["n_common_images"]}')
    print(f'  Common instances   : {results["n_common_instances"]}  '
          f'(matched in every run at OKS >= {results["oks_thr"]:.2f})')
    print()

    col_label = 42
    hdr = (f'  {"Run":<{col_label}}  '
           f'{"Avg OKS":>8}  {"Median":>8}  {"Std":>6}  '
           f'{"Total matched":>13}')
    print(hdr)
    print(f'  {"-"*col_label}  {"-"*8}  {"-"*8}  {"-"*6}  {"-"*13}')

    for r in results['per_run']:
        label = r['label']
        if len(label) > col_label:
            label = label[:col_label - 1] + '…'
        print(
            f'  {label:<{col_label}}  '
            f'{r["avg_oks"]:>8.4f}  '
            f'{r["median_oks"]:>8.4f}  '
            f'{r["std_oks"]:>6.4f}  '
            f'{r["n_matched_total"]:>13}'
        )

    print(f'{sep}\n')


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Compare prediction runs by OKS on commonly detected instances',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        'run_dirs',
        nargs='*',
        help='Prediction bundle directories to compare (each must contain '
             'manifest.json and frames.json)',
    )
    p.add_argument(
        '--group',
        default=None,
        metavar='DIR',
        help='Parent directory; auto-discovers all run sub-directories',
    )
    p.add_argument(
        '--oks-thr',
        type=float,
        default=0.5,
        metavar='THR',
        help='Minimum OKS to treat a prediction as a valid match '
             '(default: 0.5). Cannot be lower than the threshold used at '
             'prediction time (typically 0.5).',
    )
    p.add_argument(
        '--out',
        default=None,
        metavar='FILE',
        help='Save comparison results as JSON to FILE',
    )
    p.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress per-run loading messages',
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    run_dirs: List[str] = list(args.run_dirs)

    if args.group:
        group_dir = osp.abspath(args.group)
        if not osp.isdir(group_dir):
            print(f'Error: --group directory not found: {group_dir}',
                  file=sys.stderr)
            sys.exit(1)
        discovered = sorted([
            osp.join(group_dir, entry)
            for entry in os.listdir(group_dir)
            if osp.isfile(osp.join(group_dir, entry, 'manifest.json'))
        ])
        if not discovered:
            print(f'No prediction bundles found in {group_dir}',
                  file=sys.stderr)
            sys.exit(1)
        run_dirs.extend(discovered)
        print(f'Discovered {len(discovered)} run(s) in {group_dir}')

    if len(run_dirs) < 2:
        print(
            'Error: at least 2 run directories are required.\n'
            'Pass them as positional arguments or use --group <parent_dir>.',
            file=sys.stderr,
        )
        sys.exit(1)

    print(f'\nLoading {len(run_dirs)} prediction bundle(s) ...')
    results = compare_runs(
        run_dirs,
        oks_thr=args.oks_thr,
        verbose=not args.quiet,
    )
    _print_results(results)

    if args.out:
        out_dir = osp.dirname(osp.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f'Results saved to {args.out}')


if __name__ == '__main__':
    main()
