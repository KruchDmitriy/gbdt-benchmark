# GDBT speed benchmarks

This benchmark is a classical algorithm that iterates over pre-defined grid of hyper-parameters for every library, measures time to perform boosting iteration and dumps metric`s value for test split. [XGBoost benchmark](https://xgboost.ai/2018/07/04/gpu-xgboost-update.html) code served as a starting point for this library.

# Run benchmark

First of all you need to define a hyperparameters grid in json file, (you may find grid example in the root directory named example_params_grid.json)

Here every line define a hyperparameter name and a list of values to iterate in experiments.

    {
        "max_depth": [6, 10, 12],
        "learning_rate": [0.02, 0.05, 0.08],
        "reg_lambda": [1, 10],
        "subsample": [0.5, 1.0]
    }

This is example of how you can run an experiments on GPU for all three libraries for dataset airline using grid ''example_params_grid.json'' with 5000 iterations.

    python run.py --learners cat lgb xgb --datasets airline --params-grid example_params_grid.json --iterations 5000 --use-gpu

# Supported datasets
Higgs, Epsilon, MSRank, CoverType, Airlines, Year prediction MSD, Bosch, Synthetic regression/classification  
Names for script run.py: 
    
    airline,bosch,cover-type,epsilon,epsilon-sampled,higgs,higgs-sampled,msrank,msrank-classification,syntetic,synthetic-classification,year-msd

# Draw graphics:
It is supported to draw four types of graphics: 
1. Learning curves for top N runs between all experiments (parameter name -- *best*).
2. Box plot of time per iteration for each library (*time-per-iter*).
3. Draw time to achieve percent from best quality (*quality-vs-time*).
4. Learning curves for concrete experiments (*custom*).

Third method calculates the best achieved quality between all experiments. Then fix a grid of relative percents of this value. Then for each level of quality the method filter all experiments from grid search that reach it and compute median (circle point), minimum and maximum time when algorithm achieved that score.

#### Examples
Draw learning curves for 5 best experiments on dataset MSRank (multiclass mode):

    python plot.py --type best --top 5 --from-iter 500 -i ./results/MSRank-MultiClass/ -o msrank_mc_plots

Draw time curves with figure size 10x8 for experiments on Higgs dataset starting from 85% of best achieved quality to 100% in 30 points: 

    python plot.py --type quality-vs-time -f 10 8 -o higgs_plots -i ./results/Higgs/ --low-percent 0.85 --num-bins 30

Draw time per iteration box plots for every library:

    python plot.py --type time-per-iter -f 10 8 -o higgs_plots -i ./results/Higgs

# Experiments

| Dataset           | CatBoost | LightGBM | XGBoost | 
| ----------------- | -------- | -------- | ------- |
| Higgs             |          |          |         |
| Epsilon           |          |          |         |