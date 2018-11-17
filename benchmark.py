import argparse
from experiments import EXPERIMENTS
from learners import *
from generate_report import get_experiment_stats, print_all_in_one_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--datasets', default='datasets')
    parser.add_argument('--iterations', default=1000, type=int)
    parser.add_argument('--results', default='results')
    parser.add_argument('--table', default='common-table.txt')
    args = parser.parse_args()

    experiments_names = [
        'abalone',
        'airline',
        'epsilon',
        'higgs',
        'letters',
        'msrank',
        'msrank-classification',
        'synthetic',
        'synthetic-5k-features'
    ]

    learners = [
        XGBoostLearner,
        LightGBMLearner,
        CatBoostLearner
    ]

    iterations = args.iterations

    datasets_dir = args.datasets
    results_dir = args.results

    params_grid = {
        "iterations": [iterations],
        'max_depth': [6],
        'learning_rate': [0.03, 0.07, 0.15]
    }

    for experiment_name in experiments_names:
        print(experiment_name)
        experiment = EXPERIMENTS[experiment_name]
        experiment.run(args.use_gpu, learners, params_grid, datasets_dir, results_dir)

    stats = get_experiment_stats(args.results_dir, args.use_gpu, niter=iterations)
    print_all_in_one_table(stats, args.use_gpu, params=args.params, output=args.table)


if __name__ == "__main__":
    main()
