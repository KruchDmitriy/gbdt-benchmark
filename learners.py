import catboost as cat
import lightgbm as lgb
import numpy as np
import os
import sys
import time
import xgboost as xgb

from copy import deepcopy
from datetime import datetime
from sklearn.metrics import mean_squared_error, accuracy_score


# Global parameter
RANDOM_SEED = 0


class FileWithTime:
    def __init__(self, file_descriptor):
        self.file_descriptor = file_descriptor

    def write(self, message):
        if message == '\n':
            self.file_descriptor.write('\n')
            return

        time = datetime.now()
        new_message = "Time: [%d.%06d]\t%s" % (time.second, time.microsecond, message)
        self.file_descriptor.write(new_message)

    def flush(self):
        self.file_descriptor.flush()

    def close(self):
        self.file_descriptor.close()


class Logger:
    def __init__(self, filename):
        self.filename = filename
        self.stdout = sys.stdout

    def __enter__(self):
        self.file = FileWithTime(open(self.filename, 'w'))
        sys.stdout = self.file

    def __exit__(self, exception_type, exception_value, traceback):
        if exception_type is not None:
            print(str(exception_value) + '\n' + str(traceback))

        sys.stdout = self.stdout
        self.file.close()


def _params_to_str(params):
    return ''.join(map(lambda (key, value): '{}[{}]'.format(key, str(value)), params.items()))


class Learner:
    def __init__(self, data, use_gpu, eval_on_train):
        self.default_params = self._configure(data, use_gpu, eval_on_train)
        self.trees_step = 10

    def _fit(self, tunable_params):
        params = deepcopy(self.default_params)
        params.update(tunable_params)

        print('Parameters:')
        print(params)
        return params

    def __eval_iter(self, pred):
        if self.metric == "RMSE":
            return np.sqrt(mean_squared_error(data.y_test, pred))
        elif self.metric == "Accuracy":
            if self.task == "Classification":
                pred = pred > 0.5
            elif data.task == "Multiclass":
                if pred.ndim > 1:
                    pred = np.argmax(pred, axis=1)
            return accuracy_score(y_test, pred)
        else:
            raise ValueError("Unknown metric: " + data.metric)

    def __eval(self, data):
        scores = []

        for n_tree in range(num_iterations, step=self.trees_step):
            prediction = self.predict(n_tree)
            score = self.__eval_iter(data, prediction)
            scores.append(score)

        return scores

    def predict(self, n_tree):
        raise Exception('Not implemented')

    def set_train_dir(self, params, path):
        pass

    def check_log(self, log_file):
        print('Checking log')
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = f.readlines()
                return len(lines) > 0 and 'Elapsed: ' in lines[-1]
        return False

    def run(self, params, log_dirname, eval_data=None):
        if not os.path.exists(log_dirname):
            os.makedirs(log_dirname)

        path = os.path.join(log_dirname, _params_to_str(params))
        filename = os.path.join(path + '.log')

        if self.check_log(filename):
            print('Skipping experiment, reason: log already exists and is consistent')
            return 0

        self.set_train_dir(params, path)

        with Logger(filename):
            start = time.time()
            self._fit(params)
            elapsed = time.time() - start
            print('Elapsed: ' + str(elapsed))

            if eval_data:
                scores = self.__eval(eval_data)
                for idx, score in enumerate(scores):
                    print('[{1:5}] {2:}'.format(idx * self.trees_step, score))

        return elapsed


class XGBoostLearner(Learner):
    def __init__(self, data, use_gpu, eval_on_train):
        Learner.__init__(self, data, use_gpu, eval_on_train)

    def name(self):
        return 'xgboost'

    def _configure(self, data, use_gpu, eval_on_train):
        params = {
            'n_gpus': 1,
            'silent': 0,
            'seed': RANDOM_SEED
        }

        if use_gpu:
            params['tree_method'] = 'gpu_hist'
        else:
            params['tree_method'] = 'hist'

        if data.task == "Regression":
            params["objective"] = "reg:linear"
            if use_gpu:
                params["objective"] = "gpu:" + params["objective"]
        elif data.task == "Multiclass":
            params["objective"] = "multi:softmax"
            params["num_class"] = np.max(data.y_test) + 1
        elif data.task == "Classification":
            params["objective"] = "binary:logistic"
            if use_gpu:
                params["objective"] = "gpu:" + params["objective"]
        else:
            raise ValueError("Unknown task: " + data.task)

        if eval_on_train:
            if data.metric == 'Accuracy':
                params['eval_metric'] = 'error'

        self.dtrain = xgb.DMatrix(data.X_train, data.y_train)
        self.dtest = xgb.DMatrix(data.X_test, data.y_test)

        return params

    def _fit(self, tunable_params):
        params = Learner._fit(self, tunable_params)
        self.learner = xgb.train(params, self.dtrain, tunable_params['iterations'], evals=[(self.dtest, 'eval')])

    def predict(self, n_tree):
        return self.learner.predict(self.dtest, ntree_limit=n_tree)


class LightGBMLearner(Learner):
    def __init__(self, data, use_gpu, eval_on_train):
        Learner.__init__(self, data, use_gpu, eval_on_train)

    def name(self):
        return 'lightgbm'

    def _configure(self, data, use_gpu, eval_on_train):
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'verbose': 0,
            'random_state': RANDOM_SEED,
            'bagging_freq': 1
        }

        if use_gpu:
            params["device"] = "gpu"

        if data.task == "Regression":
            params["objective"] = "regression"
        elif data.task == "Multiclass":
            params["objective"] = "multiclass"
            params["num_class"] = np.max(data.y_test) + 1
        elif data.task == "Classification":
            params["objective"] = "binary"
        else:
            raise ValueError("Unknown task: " + data.task)

        if eval_on_train:
            if data.task == 'Classification':
                params['metric'] = 'binary_error'
            elif data.task == 'Multiclass':
                params['metric'] = 'multi_error'
            elif data.task == 'Regression':
                params['metric'] = 'rmse'

        self.lgb_train = lgb.Dataset(data.X_train, data.y_train)
        self.lgb_eval = lgb.Dataset(data.X_test, data.y_test, reference=self.lgb_train)

        return params

    def _fit(self, tunable_params):
        if 'max_depth' in tunable_params:
            tunable_params['num_leaves'] = 2 ** tunable_params['max_depth']
            del tunable_params['max_depth']

        num_iterations = tunable_params['iterations']
        del tunable_params['iterations']

        params = Learner._fit(self, tunable_params)
        self.learner = lgb.train(
            params,
            self.lgb_train,
            num_boost_round=num_iterations,
            valid_sets=self.lgb_eval
        )

    def predict(self, n_tree):
        return self.learner.predict(self.lgb_eval, num_iteration=n_tree)


class CatBoostLearner(Learner):
    def __init__(self, data, use_gpu, eval_on_train):
        Learner.__init__(self, data, use_gpu, eval_on_train)

    def name(self):
        return 'catboost'

    def _configure(self, data, use_gpu, eval_on_train):
        params = {
            'devices' : [0],
            'logging_level': 'Info',
            'use_best_model': False,
            'bootstrap_type': 'Bernoulli'
        }

        if use_gpu:
            params['task_type'] = 'GPU'

        if data.task == 'Regression':
            params['loss_function'] = 'RMSE'
        elif data.task == 'Classification':
            params['loss_function'] = 'Logloss'
        elif data.task == 'Multiclass':
            params['loss_function'] = 'MultiClass'

        if eval_on_train:
            if data.metric == 'Accuracy':
                params['custom_metric'] = 'Accuracy'

        self.cat_train = cat.Pool(data.X_train, data.y_train)
        self.cat_test = cat.Pool(data.X_test, data.y_test)

        return params

    def _fit(self, tunable_params):
        params = Learner._fit(self, tunable_params)
        self.model = cat.CatBoost(params)
        self.model.fit(self.cat_train, eval_set=self.cat_test, verbose_eval=True)

    def set_train_dir(self, params, path):
        params["train_dir"] = path

    def predict(self, n_tree):
        if data.task == "Multiclass":
            prediction = self.model.predict_proba(self.cat_test, ntree_end=n_tree)
        else:
            prediction = self.model.predict(self.cat_test, ntree_end=n_tree)

        return prediction
