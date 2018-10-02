import argparse
import json

from sklearn.model_selection import train_test_split, ParameterGrid

import ml_dataset_loader.datasets as data_loader
from learners import *


class Data:
    def __init__(self, X, y, name, task, metric, train_size=0.9):
        assert train_size > 0. and train_size < 1.

        test_size = 1. - train_size
        self.name = name
        self.task = task
        self.metric = metric

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=test_size, random_state=0)

        if task == 'Classification':
            self.y_train = self.y_train.astype(int)
            self.y_test = self.y_test.astype(int)


class Experiment:
    def __init__(self, data_func, name, task, metric):
        self.data_func = data_func
        self.name = name
        self.task = task
        self.metric = metric

    def run(self, use_gpu, learners, params_grid, eval_on_train, out_dir):
        X, y = self.data_func()
        data = Data(X, y, self.name, self.task, self.metric)

        device_type = 'GPU' if use_gpu else 'CPU'

        for LearnerType in learners:
            learner = LearnerType(data, use_gpu, eval_on_train)
            algorithm_name = learner.name() + '-' + device_type
            print('Started to train ' + algorithm_name)

            for params in ParameterGrid(params_grid):
                print(params)

                log_dirname = os.path.join(out_dir, experiment.name, algorithm_name)
                if eval_on_train:
                    elapsed = learner.run(params, log_dirname)
                else:
                    elapsed = learner.run(params, log_dirname, data)

                print('Timing: ' + str(elapsed) + ' sec')


def _get_all_values_from_subset(items, subset):
    filtered_keys = filter(lambda x: x in subset, items.keys())
    return [items[key] for key in filtered_keys]


if __name__ == '__main__':
    datasets = {
        "year-msd": Experiment(data_loader.get_year, "YearPredictionMSD", "Regression", "RMSE"),
        "syntetic": Experiment(data_loader.get_synthetic_regression, "Synthetic", "Regression", "RMSE"),
        "cover-type": Experiment(data_loader.get_cover_type, "Cover Type", "Multiclass", "Accuracy"),
        "epsilon": Experiment(data_loader.get_epsilon, "Epsilon", "Classification", "Accuracy"),
        "higgs": Experiment(data_loader.get_higgs, "Higgs", "Classification", "Accuracy"),
        "bosch": Experiment(data_loader.get_bosch, "Bosch", "Classification", "Accuracy"),
        "airline": Experiment(data_loader.get_airline, "Airline", "Classification", "Accuracy"),
        "higgs-sampled": Experiment(data_loader.get_higgs_sampled, "Higgs", "Classification", "Accuracy"),
        "epsilon-sampled": Experiment(data_loader.get_epsilon_sampled, "Epsilon", "Classification", "Accuracy"),
        "synthetic-classification": Experiment(data_loader.get_synthetic_classification, "Synthetic2", "Classification", "Accuracy"),
        "msrank": Experiment(data_loader.get_msrank, "MSRank", "Regression", "RMSE"),
        "msrank-classification": Experiment(data_loader.get_msrank, "MSRank", "Multiclass", "Accuracy")
    }

    learners = {
        "xgb": XGBoostLearner,
        "lgb": LightGBMLearner,
        "cat": CatBoostLearner
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--learners', nargs='*', choices=learners.keys(), required=True)
    parser.add_argument('--datasets', nargs='*', choices=datasets.keys(), required=True)
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--iterations', type=int, required=True)
    parser.add_argument('--params-grid', default=None, help='path to json file, each key corresponds to learner parameter,\
            e.g. max_depth, and list of values to run in experiment')
    parser.add_argument('--eval-on-train', default=True, help='eval test metrics during training')
    parser.add_argument('-o', '--out-dir', default='results')
    args = parser.parse_args()

    experiment_learners = _get_all_values_from_subset(learners, args.learners)
    experiments = _get_all_values_from_subset(datasets, args.datasets)

    params_grid = {
    	"iterations": [args.iterations]
    }

    if args.params_grid:
        with open(args.params_grid) as f:
            grid = json.load(f)
        params_grid.update(grid)


    for experiment in experiments:
    	print(experiment.name)
    	experiment.run(args.use_gpu, experiment_learners, params_grid, args.eval_on_train, args.out_dir)

