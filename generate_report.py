import argparse
import json

import numpy as np

from log_parser import read_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--results-dir', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--curve-output', required=True)
    args = parser.parse_args()

    tracks = read_results(args.results_dir)

    assert len(tracks.keys()) == 1

    tracks = tracks[tracks.keys()[0]]

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

        median.append(np.median(time_per_iter))
        low.append(np.quantile(time_per_iter, 0.25))
        high.append(np.quantile(time_per_iter, 0.75))
        total.append(track.duration)

        if best_quality > cur_quality:
            best_quality = cur_quality
            best_iter = np.argmin(track.get_series()[1])
            best_track = track

    print(best_track)

    median = sum(median) / len(median)
    low = sum(low) / len(low)
    high = sum(high) / len(high)
    total = sum(total) / len(total)

    with open(args.output, 'w') as f:
        json.dump({
            "Best": best_quality,
            "Iter": best_iter,
            "MedianTime": median,
            "LowTime": low,
            "HighTime": high,
            "TotalTime": total
        }, f)

    best_track_scores = list(best_track.get_series()[1])
    with open(args.curve_output, 'w') as f:
        json.dump(best_track_scores, f, indent=4)


if __name__ == "__main__":
    main()
