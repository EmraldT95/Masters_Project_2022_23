# Comparison of Random, BOHB and DEHB search strategies on PMLB Datasets
The goal of this project was to compare the performance of Random search against Bayesian Optimisation with Hyperband (BOHB) search method named on a given set of datasets and a fixed hyperparameter search space. Additionally, we also check the performance of Differential Evolution with Hyperband (DEHB) search under the same conditions.

## Pre-requisites
The code works on `Python >= 3.9`. All the required packages can be installed from requirements files.

````
pip install -r requirements.txt
````

## Usage:

All the `pmlb` datasets used for in the codebase can be found in `src/estimators.py` file. To run the algorithm for all the classification datasets, execute the below code in the terminal:

````bash
python run.py
````

The available flags for `run.py` are:
| Flag                        | Values                                | Default      | Description                                                  |
| --------------------------- | ------------------------------------- | ------------ | ------------------------------------------------------------ |
| `-wl`, `--wallclock_limit`  | any *int*                             | 1800         | The Wallclock limit to run search algorithm per dataset.     |
| `-tk`, `--task`             | `'reg', 'cls'`                        | `'cls'`      | Whether to perform Classification or Regression.             |
| `-st`, `--search_type`      | `'Random', 'BOHB', 'DEHB', 'All'`     | `'All'`      | Type of search strategy to run.                              |
| `-p`, `--path`              | any *str*                             | `'./output'` | Output folder to get the files from.                         |
| `--test`                    | *boolean on/off*                      |              | To test the incumbents for a run.                            |

Below is how you test for a sample regression task:

````bash
python run.py --task reg --path ./path_to_your_output_folder --test
````

## Plotting:

To create plots, we have to run the `plots.py` file. Currently we have two plots available - the Pareto front and the Average train/test score. To generate the pareto front plot for a sample run, execute the below code in the terminal.

````bash
python plots.py --path ./path_to_your_output_folder
````

The available flags for `run.py` are:
| Flag                        | Values                                | Default      | Description                                                  |
| --------------------------- | ------------------------------------- | ------------ | ------------------------------------------------------------ |
| `-pt`, `--plot`             | `'Pareto', 'Avg', 'All'`              | `'All'`      | Type of plot to generate.                                    |
| `-p`, `--path`              | any *str*                             | `'./output'` | Output folder to get the files from.                         |
| `--test`                    | *boolean on/off*                      |              | To include test results of a run (Used in `Avg` plot).       |
