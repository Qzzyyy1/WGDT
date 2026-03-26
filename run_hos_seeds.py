import argparse
import csv
import json
import os
import subprocess
import sys
import threading
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

import numpy as np


ROOT = Path(__file__).resolve().parent
PAIR_PRESETS = OrderedDict([
    ('pavia', ('PaviaU_7gt', 'PaviaC_OS')),
    ('houston', ('Houston13_7gt', 'Houston18_OS')),
    ('hyrank', ('HyRank_source', 'HyRank_target')),
    ('yancheng', ('Yancheng_ZY', 'Yancheng_GF')),
])
PAIR_SEEDS = OrderedDict([
    ('pavia', [7, 43, 45, 46, 66, 67, 77, 81, 88, 98]),
    ('houston', [1, 23, 30, 35, 52, 64, 68, 72, 73, 91]),
    ('hyrank', [6, 13, 20, 40, 49, 58, 60, 67, 81, 91]),
    ('yancheng', [15, 18, 19, 24, 37, 50, 51, 62, 77, 80]),
])


def parse_args():
    parser = argparse.ArgumentParser(
        description='Batch-run WGDT and summarize HOS / class accuracy across random seeds.'
    )
    parser.add_argument('--pairs', nargs='+', choices=list(PAIR_PRESETS.keys()), default=list(PAIR_PRESETS.keys()))
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--gpus', nargs='+', type=int, help='Physical GPU ids, for example: --gpus 0 1 2 3')
    parser.add_argument('--log_name', type=str, default='WGDT')
    parser.add_argument('--python', type=str, default=sys.executable)
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--pre_train_epochs', type=int)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--patch', type=int)
    parser.add_argument('--pre_train', choices=['True', 'False'])
    parser.add_argument('--draw', choices=['True', 'False'])

    args = parser.parse_args()
    if args.gpus is None:
        args.gpus = [args.device]
    if not args.gpus:
        raise ValueError('--gpus must contain at least one GPU id')
    return args


def get_result_path(log_name, source_dataset, target_dataset, seed):
    return ROOT / 'logs' / log_name / '{} {}-{} seed={}.json'.format(
        log_name, source_dataset, target_dataset, seed
    )


def get_runner_log_path(log_name, source_dataset, target_dataset, seed):
    log_dir = ROOT / 'logs' / log_name / 'runner_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / '{}-{}-seed={}.log'.format(source_dataset, target_dataset, seed)


def build_command(args, source_dataset, target_dataset, seed, visible_device):
    command = [
        args.python,
        'main.py',
        '--source_dataset', source_dataset,
        '--target_dataset', target_dataset,
        '--seed', str(seed),
        '--device', str(visible_device),
        '--log_name', args.log_name,
    ]

    optional_args = [
        ('--epochs', args.epochs),
        ('--pre_train_epochs', args.pre_train_epochs),
        ('--batch', args.batch),
        ('--patch', args.patch),
        ('--pre_train', args.pre_train),
        ('--draw', args.draw),
    ]
    for key, value in optional_args:
        if value is not None:
            command.extend([key, str(value)])

    return command


def make_tasks(args):
    tasks = []
    for pair_name in args.pairs:
        source_dataset, target_dataset = PAIR_PRESETS[pair_name]
        for seed in PAIR_SEEDS[pair_name]:
            tasks.append({
                'pair_name': pair_name,
                'source_dataset': source_dataset,
                'target_dataset': target_dataset,
                'seed': seed,
            })
    return tasks


def run_single_experiment(args, task, gpu_id, console_lock):
    source_dataset = task['source_dataset']
    target_dataset = task['target_dataset']
    seed = task['seed']
    result_path = get_result_path(args.log_name, source_dataset, target_dataset, seed)

    if args.skip_existing and result_path.exists():
        with console_lock:
            print('[SKIP][GPU {}] {} -> {}, seed={}'.format(gpu_id, source_dataset, target_dataset, seed))
        return result_path

    command = build_command(args, source_dataset, target_dataset, seed, visible_device=0)
    runner_log_path = get_runner_log_path(args.log_name, source_dataset, target_dataset, seed)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    with console_lock:
        print('[START][GPU {}] {} -> {}, seed={}'.format(gpu_id, source_dataset, target_dataset, seed))

    with runner_log_path.open('w', encoding='utf-8') as log_file:
        log_file.write('gpu={}\n'.format(gpu_id))
        log_file.write('CUDA_VISIBLE_DEVICES={}\n'.format(gpu_id))
        log_file.write('command={}\n\n'.format(' '.join(command)))
        log_file.flush()

        subprocess.run(
            command,
            cwd=str(ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )

    if not result_path.exists():
        raise FileNotFoundError('Result file not found: {}'.format(result_path))

    with console_lock:
        print('[DONE][GPU {}] {} -> {}, seed={}'.format(gpu_id, source_dataset, target_dataset, seed))
    return result_path


def run_tasks_parallel(args, tasks):
    task_queue = Queue()
    for task in tasks:
        task_queue.put(task)

    console_lock = threading.Lock()
    stop_event = threading.Event()
    errors = []

    def worker(gpu_id):
        while not stop_event.is_set():
            try:
                task = task_queue.get_nowait()
            except Empty:
                return

            try:
                run_single_experiment(args, task, gpu_id, console_lock)
            except Exception as exc:
                errors.append((task, gpu_id, exc))
                stop_event.set()
            finally:
                task_queue.task_done()

    threads = []
    for gpu_id in args.gpus:
        thread = threading.Thread(target=worker, args=(gpu_id,), daemon=True)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    if errors:
        task, gpu_id, exc = errors[0]
        runner_log_path = get_runner_log_path(
            args.log_name, task['source_dataset'], task['target_dataset'], task['seed']
        )
        raise RuntimeError(
            'Task failed on GPU {}: {} -> {}, seed={}. See {}'.format(
                gpu_id, task['source_dataset'], task['target_dataset'], task['seed'], runner_log_path
            )
        ) from exc


def load_metrics(result_path):
    with result_path.open('r', encoding='utf-8') as file:
        result = json.load(file)

    if 'hos' not in result:
        raise KeyError('Missing field "hos" in {}'.format(result_path))
    if 'classes_acc' not in result:
        raise KeyError('Missing field "classes_acc" in {}'.format(result_path))

    return {
        'hos': float(result['hos']),
        'classes_acc': [float(value) for value in result['classes_acc']],
    }


def summarize_metrics(metrics_list):
    hos_values = np.asarray([item['hos'] for item in metrics_list], dtype=np.float64)
    classes_acc_values = np.asarray([item['classes_acc'] for item in metrics_list], dtype=np.float64)

    if classes_acc_values.ndim != 2:
        raise ValueError('classes_acc shape is invalid: {}'.format(classes_acc_values.shape))

    hos_mean = float(hos_values.mean())
    hos_std = float(hos_values.std())
    classes_acc_mean = classes_acc_values.mean(axis=0)

    return {
        'num_runs': int(hos_values.size),
        'hos_values': hos_values.tolist(),
        'hos_mean': hos_mean,
        'hos_std': hos_std,
        'hos_mean_pm_std': '{:.6f} ± {:.6f}'.format(hos_mean, hos_std),
        'classes_acc_mean': classes_acc_mean.tolist(),
    }


def build_markdown_tables(summary):
    hos_lines = [
        '| Pair | Source | Target | Runs | HOS (mean ± std) |',
        '| --- | --- | --- | ---: | --- |',
    ]
    for pair_name, pair_summary in summary['datasets'].items():
        hos_lines.append(
            '| {} | {} | {} | {} | {} |'.format(
                pair_name,
                pair_summary['source_dataset'],
                pair_summary['target_dataset'],
                pair_summary['num_runs'],
                pair_summary['hos_mean_pm_std'],
            )
        )

    max_classes = max(len(item['classes_acc_mean']) for item in summary['datasets'].values())
    class_header = ['Pair'] + ['Class {}'.format(index + 1) for index in range(max_classes)]
    class_sep = ['---'] + ['---:' for _ in range(max_classes)]
    class_lines = [
        '| ' + ' | '.join(class_header) + ' |',
        '| ' + ' | '.join(class_sep) + ' |',
    ]
    for pair_name, pair_summary in summary['datasets'].items():
        values = ['{:.6f}'.format(value) for value in pair_summary['classes_acc_mean']]
        values.extend([''] * (max_classes - len(values)))
        class_lines.append('| {} | {} |'.format(pair_name, ' | '.join(values)))

    return '\n'.join([
        '# HOS Summary',
        '',
        *hos_lines,
        '',
        '# Class Accuracy Mean',
        '',
        *class_lines,
        '',
    ])


def save_summary(args, summary):
    output_dir = ROOT / 'logs' / args.log_name
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_tag = '-'.join(args.pairs)
    file_stem = 'hos_summary_{}_preset_seeds'.format(pair_tag)
    json_path = output_dir / '{}.json'.format(file_stem)
    csv_path = output_dir / '{}.csv'.format(file_stem)
    md_path = output_dir / '{}.md'.format(file_stem)

    with json_path.open('w', encoding='utf-8') as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    max_classes = max(len(item['classes_acc_mean']) for item in summary['datasets'].values())
    header = [
        'pair', 'source_dataset', 'target_dataset', 'num_runs',
        'hos_mean_pm_std', 'hos_mean', 'hos_std'
    ] + ['class_acc_{}'.format(index + 1) for index in range(max_classes)]

    with csv_path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for pair_name, pair_summary in summary['datasets'].items():
            class_means = list(pair_summary['classes_acc_mean'])
            padding = [''] * (max_classes - len(class_means))
            writer.writerow([
                pair_name,
                pair_summary['source_dataset'],
                pair_summary['target_dataset'],
                pair_summary['num_runs'],
                pair_summary['hos_mean_pm_std'],
                pair_summary['hos_mean'],
                pair_summary['hos_std'],
            ] + class_means + padding)

    markdown_table = build_markdown_tables(summary)
    with md_path.open('w', encoding='utf-8') as file:
        file.write(markdown_table)

    return json_path, csv_path, md_path


def main():
    args = parse_args()
    tasks = make_tasks(args)

    print('Total tasks: {}'.format(len(tasks)))
    print('Using GPUs: {}'.format(args.gpus))
    print('One task per GPU at a time')

    run_tasks_parallel(args, tasks)

    summary = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'log_name': args.log_name,
        'python': args.python,
        'gpus': args.gpus,
        'pair_seeds': {pair_name: PAIR_SEEDS[pair_name] for pair_name in args.pairs},
        'datasets': OrderedDict(),
    }

    for pair_name in args.pairs:
        source_dataset, target_dataset = PAIR_PRESETS[pair_name]
        metrics_list = []

        for seed in PAIR_SEEDS[pair_name]:
            result_path = get_result_path(args.log_name, source_dataset, target_dataset, seed)
            metrics_list.append(load_metrics(result_path))

        pair_summary = summarize_metrics(metrics_list)
        pair_summary.update({
            'source_dataset': source_dataset,
            'target_dataset': target_dataset,
        })
        summary['datasets'][pair_name] = pair_summary

    json_path, csv_path, md_path = save_summary(args, summary)

    print('\nHOS summary:')
    for pair_name, pair_summary in summary['datasets'].items():
        print('- {}: {}'.format(pair_name, pair_summary['hos_mean_pm_std']))
    print('\nJSON saved to: {}'.format(json_path))
    print('CSV saved to: {}'.format(csv_path))
    print('Markdown saved to: {}'.format(md_path))


if __name__ == '__main__':
    main()
