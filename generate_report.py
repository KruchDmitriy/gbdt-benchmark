# coding=utf-8
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

from log_parser import read_results, parse_log, ALGORITHMS
from experiments import EXPERIMENT_TYPE


def calculate_statistics(tracks, niter):
    niter -= 1

    best_track = None
    best_quality = np.inf
    best_iter = -1

    median = []
    low = []
    high = []
    total = []

    for track in tracks:
        cur_quality = track.get_best_score()
        time_per_iter = track.get_time_per_iter()
        if time_per_iter.shape[0] < niter:
            continue

        median.append(np.median(time_per_iter))
        low.append(np.quantile(time_per_iter, 0.25))
        high.append(np.quantile(time_per_iter, 0.75))
        time_series = track.time_series
        total.append(time_series[niter] - time_series[0])

        if best_quality > cur_quality:
            best_quality = cur_quality
            best_iter = np.argmin(track.get_series()[1])
            best_track = track

    if best_track is None:
        return {}

    print(best_track)

    median = sum(median) / len(median)
    low = sum(low) / len(low)
    high = sum(high) / len(high)
    dev = max(median - low, high - median)
    total = np.median(total)

    return {
        "Best": best_quality,
        "Iter": best_iter,
        "MedianTime": median,
        "Deviation": dev,
        "TotalTime": total
    }


def get_experiment_stats(results, gpu, niter):
    stats = {}

    if os.path.isdir(results):
        tracks = json_from_logs(results)
    else:
        tracks = read_results(results)

    for experiment_name in tracks.iterkeys():
        stats[experiment_name] = {}

        experiment_tracks = tracks[experiment_name]
        experiment_tracks = dict(filter(lambda track: gpu == ('GPU' in track[0]), experiment_tracks.items()))

        for algorithm_name in experiment_tracks.iterkeys():
            stats[experiment_name][algorithm_name] = {}
            table_tracks = split_tracks(experiment_tracks[algorithm_name])

            for params, cur_tracks in table_tracks.iteritems():
                stat = calculate_statistics(cur_tracks, niter)
                if stat == {}:
                    continue
                stats[experiment_name][algorithm_name][params] = stat

    stats = dict(filter(lambda experiment_stat: len(experiment_stat[1]) > 0, stats.items()))

    return stats


def get_table_header(experiment_stats):
    parameter_set = None

    for algorithm_name in experiment_stats.iterkeys():
        alg_parameter_set = set(experiment_stats[algorithm_name].keys())
        if parameter_set is None:
            parameter_set = alg_parameter_set
        else:
            parameter_set &= alg_parameter_set

    return sorted(list(parameter_set))


def get_median_str(stat):
    median = np.round(stat["MedianTime"], 3)
    dev = np.round(stat["Deviation"], 3)

    median_str = str(median)
    if abs(dev) > 0:
        median_str += u' +/- ' + str(dev)

    return median_str


def print_all_in_one_table(stats, gpu, params, output):
    median_table = []
    total_table = []

    index = ["catboost", "xgboost", "lightgbm"]

    for algorithm_name in index:
        median_row = []
        total_row = []

        if gpu:
            algorithm_name += "-GPU"
        else:
            algorithm_name += "-CPU"

        for experiment_name in stats.iterkeys():
            experiment_stats = stats[experiment_name]

            if params not in experiment_stats[algorithm_name]:
                median_row.append(0.)
                total_row.append(0.)
                continue

            cur_stat = experiment_stats[algorithm_name][params]
            median_row.append(get_median_str(cur_stat))
            total_row.append(np.round(cur_stat["TotalTime"], 3))

        median_table.append(median_row)
        total_table.append(total_row)

    median_table = pd.DataFrame(median_table, index=index, columns=stats.keys())
    total_table = pd.DataFrame(total_table, index=index, columns=stats.keys())

    with open(output, 'w') as f:
        f.write('Median time per iter, sec')
        f.write('\n')
        f.write(median_table.to_string())
        f.write('\n')
        f.write('Total time, sec')
        f.write('\n')
        f.write(total_table.to_string())
        f.write('\n')


def print_experiment_table(stats, output):
    for experiment_name in stats.iterkeys():
        experiment_stats = stats[experiment_name]

        header = get_table_header(experiment_stats)

        median_table = []
        total_table = []

        for algorithm_name in experiment_stats.iterkeys():
            algorithm_stats = experiment_stats[algorithm_name]
            median_row = []
            total_row = []

            for parameter in header:
                cur_stat = algorithm_stats[parameter]

                total = np.round(cur_stat["TotalTime"], 3)

                median_row.append(get_median_str(cur_stat))
                total_row.append(total)

            median_table.append(median_row)
            total_table.append(total_row)

        index = experiment_stats.keys()
        median_table = pd.DataFrame(median_table, index=index, columns=header)
        total_table = pd.DataFrame(total_table, index=index, columns=header)

        with open(experiment_name + output, 'w') as f:
            f.write('Median time per iter, sec')
            f.write('\n')
            f.write(median_table.to_string())
            f.write('\n')
            f.write('Total time, sec')
            f.write('\n')
            f.write(total_table.to_string())
            f.write('\n')


def split_tracks(tracks):
    depths = []
    samples = []

    for track in tracks:
        depths.append(track.params.max_depth)

        if "subsample" not in track.params_dict.keys():
            samples.append(1.0)
            continue

        samples.append(track.params.subsample)

    depths = set(depths)
    samples = set(samples)

    table_tracks = {(depth, subsample): [] for depth in depths for subsample in samples}

    for track in tracks:
        subsample = track.params.subsample if "subsample" in track.params_dict.keys() else 1.
        table_tracks[(track.params.max_depth, subsample)].append(track)

    return table_tracks


def json_from_logs(dir_name):
    results = {}
    for experiment_name in os.listdir(dir_name):
        if experiment_name not in EXPERIMENT_TYPE.keys():
            continue

        experiment_dir_name = os.path.join(dir_name, experiment_name)
        task_type = EXPERIMENT_TYPE[experiment_name][0]
        print(experiment_name + ' ' + task_type)
        results[experiment_name] = {}
        for algorithm_name in os.listdir(experiment_dir_name):
            print(algorithm_name)
            if algorithm_name not in ALGORITHMS:
                continue

            results[experiment_name][algorithm_name] = []
            cur_dir = os.path.join(experiment_dir_name, algorithm_name)
            for log_name in os.listdir(cur_dir):
                path = os.path.join(cur_dir, log_name)
                iterations_str = re.findall(r'iterations\[(\d+)\]', log_name)
                if not os.path.isfile(path) or len(iterations_str) != 1:
                    continue
                params_str = log_name.rstrip('.log')
                iterations = int(iterations_str[0])
                try:
                    track = parse_log(algorithm_name, experiment_name, task_type, params_str, path, iterations)
                except Exception as e:
                    print('Log for ' + path + ' is broken: ' + repr(e))
                    continue
                results[experiment_name][algorithm_name].append(track)
    return results


def print_n_experiment_duration(results, gpu, n, output):
    niter = 5000

    if os.path.isdir(results):
        tracks = json_from_logs(results)
    else:
        tracks = read_results(results)

    table = []
    index = []

    for experiment_name in tracks.iterkeys():
        experiment_tracks = tracks[experiment_name]
        experiment_tracks = dict(filter(lambda track_: gpu == ('GPU' in track_[0]), experiment_tracks.items()))

        row = []

        for algorithm_name in sorted(experiment_tracks.iterkeys()):
            value = 0.
            if len(experiment_tracks[algorithm_name]) < 2:
                continue

            for track in experiment_tracks[algorithm_name]:
                value += (np.median(track.time_per_iter) * niter) / 60.  # minutes

            row.append(value / 60. / float(len(experiment_tracks[algorithm_name])) * n)  # hours

        if len(row) != 0:
            index.append(experiment_name)
            table.append(row)

    header = ['catboost', 'lightgbm', 'xgboost']
    table = pd.DataFrame(table, index=index, columns=header)

    with open(output, 'w') as f:
        f.write('Optimization time, hours')
        f.write('\n')
        f.write(table.to_string())
        f.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--result', default='./results.json')
    parser.add_argument('-o', '--output')
    parser.add_argument('-t', '--type',
                        choices=['common-table', 'by-depth-table', 'json', 'opt-time'],
                        default='common-table')
    parser.add_argument('-f', '--filter', choices=['only-gpu', 'only-cpu'], default='only-gpu')
    parser.add_argument('-p', '--params', default=(8.0, 1.0))
    parser.add_argument('--niter', type=int, default=999)
    args = parser.parse_args()

    on_gpu = args.filter == 'only-gpu'
    stats = get_experiment_stats(args.result, on_gpu, niter=args.niter)

    output = args.output
    if args.output is None:
        output = args.type + '.txt'

    if args.type == 'common-table':
        print_all_in_one_table(stats, on_gpu, params=args.params, output=output)
        return

    if args.type == 'by-depth-table':
        print_experiment_table(stats, output)
        return

    if args.type == 'json':
        while len(stats.keys()) == 1:
            stats = stats.values()[0]

        with open(output, 'w') as f:
            json.dump(stats, f)

        return

    if args.type == 'opt-time':
        print_n_experiment_duration(results=args.result, gpu=on_gpu, n=50, output=output)
        return


if __name__ == "__main__":
    main()
