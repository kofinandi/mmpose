# Copyright (c) OpenMMLab. All rights reserved.
"""Extension of tools/test.py that tracks evaluation results in a central
JSON file, organized by model group and variant."""

import argparse
import json
import os
import os.path as osp
from datetime import datetime

import mmengine
from mmengine.config import Config, DictAction
from mmengine.hooks import Hook
from mmengine.runner import Runner

DEFAULT_RESULTS_FILE = 'eval_results.json'


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMPose test (and eval) model with result tracking')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--model-name',
        required=True,
        help='model group name (e.g. ViTPose, HRNet, ShuffleNet)')
    parser.add_argument(
        '--model-variant',
        required=True,
        help='model variant within the group (e.g. small, base, large, huge)')
    parser.add_argument(
        '--results-file',
        default=DEFAULT_RESULTS_FILE,
        help=f'path to the central JSON results file '
             f'(default: {DEFAULT_RESULTS_FILE})')
    parser.add_argument(
        '--work-dir', help='the directory to save evaluation results')
    parser.add_argument('--out', help='additionally save metric results to '
                        'a standalone file')
    parser.add_argument(
        '--dump',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--show-dir',
        help='directory where the visualization images will be saved.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to display the prediction results in a window.')
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='visualize per interval samples.')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='display time of every window. (second)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument(
        '--badcase',
        action='store_true',
        help='whether analyze badcase in test')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    cfg.launcher = args.launcher
    cfg.load_from = args.checkpoint

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if (args.show and not args.badcase) or (args.show_dir is not None):
        assert 'visualization' in cfg.default_hooks, \
            'PoseVisualizationHook is not set in the ' \
            '`default_hooks` field of config. Please set ' \
            '`visualization=dict(type="PoseVisualizationHook")`'

        cfg.default_hooks.visualization.enable = True
        cfg.default_hooks.visualization.show = False \
            if args.badcase else args.show
        if args.show:
            cfg.default_hooks.visualization.wait_time = args.wait_time
        cfg.default_hooks.visualization.out_dir = args.show_dir
        cfg.default_hooks.visualization.interval = args.interval

    if args.badcase:
        assert 'badcase' in cfg.default_hooks, \
            'BadcaseAnalyzeHook is not set in the ' \
            '`default_hooks` field of config. Please set ' \
            '`badcase=dict(type="BadcaseAnalyzeHook")`'

        cfg.default_hooks.badcase.enable = True
        cfg.default_hooks.badcase.show = args.show
        if args.show:
            cfg.default_hooks.badcase.wait_time = args.wait_time
        cfg.default_hooks.badcase.interval = args.interval

        metric_type = cfg.default_hooks.badcase.get('metric_type', 'loss')
        if metric_type not in ['loss', 'accuracy']:
            raise ValueError('Only support badcase metric type'
                             "in ['loss', 'accuracy']")

        if metric_type == 'loss':
            if not cfg.default_hooks.badcase.get('metric'):
                cfg.default_hooks.badcase.metric = cfg.model.head.loss
        else:
            if not cfg.default_hooks.badcase.get('metric'):
                cfg.default_hooks.badcase.metric = cfg.test_evaluator

    if args.dump is not None:
        assert args.dump.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        dump_metric = dict(type='DumpResults', out_file_path=args.dump)
        if isinstance(cfg.test_evaluator, (list, tuple)):
            cfg.test_evaluator = [*cfg.test_evaluator, dump_metric]
        else:
            cfg.test_evaluator = [cfg.test_evaluator, dump_metric]

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


def load_results(results_file: str) -> dict:
    """Load the central results JSON file, or return an empty dict."""
    if osp.isfile(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    return {}


def save_results(results_file: str, data: dict) -> None:
    """Persist the results dict to disk, creating parent dirs if needed."""
    os.makedirs(osp.dirname(osp.abspath(results_file)), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(data, f, indent=2)


def append_entry(results_file: str, model_name: str, model_variant: str,
                 metrics: dict, config_path: str, checkpoint_path: str) -> None:
    """Append one evaluation entry under results[model_name][model_variant]."""
    data = load_results(results_file)

    entry = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'config': config_path,
        'checkpoint': checkpoint_path,
        'metrics': metrics,
    }

    data.setdefault(model_name, {}).setdefault(model_variant, []).append(entry)
    save_results(results_file, data)

    print(f'\n[test_tracked] Results saved to {results_file} '
          f'({model_name} / {model_variant})')


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg = merge_args(cfg, args)

    runner = Runner.from_cfg(cfg)

    # Capture metrics via a hook so we can both save to --out and track them.
    captured = {}

    class TrackingHook(Hook):

        def after_test_epoch(self, _, metrics=None):
            if metrics is None:
                return
            captured['metrics'] = metrics
            if args.out:
                mmengine.dump(metrics, args.out)

    runner.register_hook(TrackingHook(), 'LOWEST')

    runner.test()

    if captured.get('metrics'):
        append_entry(
            results_file=args.results_file,
            model_name=args.model_name,
            model_variant=args.model_variant,
            metrics=captured['metrics'],
            config_path=osp.abspath(args.config),
            checkpoint_path=osp.abspath(args.checkpoint),
        )
    else:
        print('[test_tracked] Warning: no metrics were produced; '
              'results file was not updated.')


if __name__ == '__main__':
    main()
