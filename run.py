import os
import argparse
import json
import copy
import logging
import time
import itertools
import gc

import pdb
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score, r2_score
from ConfigSpace.configuration_space import Configuration

from src.estimators import Classifiers, Regressors, CAT_DATASETS, REG_DATASETS
from src.data import Dataset
from src.search_algorithms import smac_search, dehb_search

current_dir = os.getcwd()

_ESTIMATORS = {
    "Classification": Classifiers,
    "Regression": Regressors
}

def get_estimator_instance(config: dict, task: str, data: np.ndarray, seed: int) -> tuple:
    """
        Creates an instance of the base estimator based on the configuration passed.

        Parameters:
        ----------
        config: dict
            A dictionary containing the estimator type and its hyperparameters.
        task: str
            Either regression or classification. This is needed to use the correct values
            for the hyperparameters depending on the estimator.
        data: NDArray
            The training dataset as an ndarray. This is needed mainly for SVM to set the
            `dual` hyperparameter.
        seed: int
            Used for deterministic output for the estimators that are stochastic.

        Returns:
        -------
        sklearn.estimator object: An instance of the estimator based on the config passed.
    """
    base_estimator = config.pop("base_estimator")
    
    # The configuration space for some of the estimators have custom parameters because of a 
    # parameter naming clash with other estimators. Dealing with that here.
    if base_estimator == "MLP":
        num_layers = config.pop("n_layer")
        num_neurons = config.pop("n_neurons")
        config.update(hidden_layer_sizes=[num_neurons] * num_layers)
        if task == "Classification":
            config.update(random_state=seed)

    if base_estimator == "SVM":
        # Prefer dual=False when n_samples > n_features
        dual_val = data.shape[0] < data.shape[1]
        config.update(max_iter=config.pop("max_iter_svm"))
        if task == "Classification":
            class_weight = config.pop("class_weight")
            class_weight = None if class_weight == "None" else class_weight
            config.update(dual=dual_val, class_weight=class_weight, random_state=seed)
        else:
            config.update(random_state=seed)
    
    if base_estimator == "XGB":
        if task == "Classification":
            config.update(random_state=seed)

    if base_estimator == "BRR":
        tol_brr = config.pop("tol_brr")
        config.update(tol=tol_brr)

    # return an instance of the estimator
    return _ESTIMATORS[task](base_estimator, config)

def trainer(cfg: Configuration, search_type: str, task_type: str, train: pd.DataFrame, seed: int, **kwargs):
    """
        Creates an instance of the Estimator and fits the given data on it.

        Parameters:
        ----------
        cfg: Configuration
            The configuration used by the tae_runner to optimize the target function.
        search_type: str
            The search strategy - Random, BOHB, DEHB, All.
        task_type: str
            Either regression or classification.
        train: DataFrame
            The training dataset to use to optimize the target function.
        seed: int
            The RandomState seed to get deterministic results.
        kwargs: dict
            Contains other arguments such as the budget of the current run, set by SMAC itself.

        Returns:
        -------
        tuple (SMAC) | dict (DEHB) : The cost along with some additional information.
    """
    assert train is not None
    # Get the classifier instance
    config_dict = cfg.get_dictionary()
    train_X = train.X.values
    train_y = train.y.values.ravel()
    # Passing a copy of the config dict since we are overriding some values due to name clash
    estimator = get_estimator_instance(copy.deepcopy(config_dict), task_type, train_X, seed)
    estimator_type = config_dict["base_estimator"]
    print(f"------Estimator : {estimator_type}--------")
    print(config_dict)

    start = time.time()

    # If the size of the dataset is greater than 100k records, or
    # if the estimator is of type MLP, use hold-one-out for validation. 
    # Else use KFold cross-validation.
    try:
        if train_X.shape[0] > 100000 or estimator_type == "MLP":
            train_X_t, train_X_v, train_y_t, train_y_v = train_test_split(train_X, train_y, train_size=0.8, random_state=seed)
            estimator.fit(train_X_t, train_y_t)
            if task == "Classification":
                val_score = f1_score(train_y_v, estimator.predict(train_X_v), average='micro')
            else:
                val_score = r2_score(train_y_v, estimator.predict(train_X_v))
        else:
            kf = KFold(n_splits=5)
            scores = []
            for train_indices, val_indices in kf.split(train_X):
                estimator.fit(train_X[train_indices], train_y[train_indices])
                if task == "Classification":
                    score = f1_score(train_y[val_indices], estimator.predict(train_X[val_indices]), average='micro')
                else:
                    score = r2_score(train_y[val_indices], estimator.predict(train_X[val_indices]))

                scores.append(score)

            val_score = np.mean(scores)
    except Exception as e:
        print("------ Estimator training failed ------\n", e)
        val_score = 0

    cost = time.time() - start

    print(f"Estimator validation score = {val_score}")
    del estimator

    if search_type == "DEHB":
        budget = kwargs['budget']
        result = {
            "fitness": 1-val_score, # DE/DEHB minimizes
            "cost": cost,
            "info": {
                "score": val_score,
                "budget": budget,
                "search_type": search_type
            }
        }
    else:
        result = (1 - val_score, {"val_f1_score": val_score, "search_type": search_type})

    return result

def tester(output_path: str, train: pd.DataFrame, test: pd.DataFrame, task_type: str, search_type: str, scores: dict, seed: int):
    """
        Tests the given config of a test dataset

        Parameters
        ----------
            output_path: str
                The path where the run files reside.
            train: DataFrame
                The training dataset.
            test: DataFrame
                The test dataset.
            task_type: str
                Either regression or classification.
            search_type: str
                The search strategy - Random, BOHB, DEHB, All.
            scores: dict
                A dict to store the f1/r2 scores of the incumbents and the config itself .
            seed: int
                The RandomState seed to get deterministic results.

        Returns
        -------
        dict : The scores dict.
    """
    assert train is not None
    assert test is not None

    # Function that does the training and testing of the incumbent config
    def _get_test_score(config):
        estimator_type = config["base_estimator"]
        estimator = get_estimator_instance(config, task_type, train_X, seed)
        # Train (since we are not saving the models itself)
        estimator.fit(train_X, train_y)
        # Test
        predictions = estimator.predict(test_X)
        # Get the f1/r2 scores in the test dataset
        if task_type == "Classification":
            score = f1_score(test_y, predictions, average='micro')
        else:
            score = r2_score(test_y, predictions)
        
        return (score, estimator_type)

    train_X = train.X.values
    train_y = train.y.values.ravel()
    test_X = test.X.values
    test_y = test.y.values.ravel()

    dir_list = os.listdir(output_path)
    for incumbent_json in dir_list:
        f = open(os.path.join(current_dir, f"{output_path}/{incumbent_json}"))
        config_dict = json.loads(f.read())
        _type = config_dict.pop("search_type")
        if _type == "BOHB":
            BOHB_best_config = config_dict
        elif _type == "Random":
            RS_best_config = config_dict
        else:
            DEHB_best_config = config_dict

    # Get the test scores
    if search_type == "Random" or search_type == "All":
        _score, estimator_type = _get_test_score(RS_best_config)
        scores["RS_Score"].append(_score)
        scores["RS_Classifier"].append(estimator_type)
    else:
        scores["RS_Score"].append(0)
        scores["RS_Classifier"].append("-")

    if search_type == "BOHB" or search_type == "All":
        _score, estimator_type = _get_test_score(BOHB_best_config)
        scores["BOHB_Score"].append(_score)
        scores["BOHB_Classifier"].append(estimator_type)
    else:
        scores["BOHB_Score"].append(0)
        scores["BOHB_Classifier"].append("-")

    if search_type == "DEHB" or search_type == "All":
        _score, estimator_type = _get_test_score(DEHB_best_config)
        scores["DEHB_Score"].append(_score)
        scores["DEHB_Classifier"].append(estimator_type)
    else:
        scores["DEHB_Score"].append(0)
        scores["DEHB_Classifier"].append("-")

    return scores

def call_gc():
    """
        Calls garbage collector manually to free memory for next run
    """
    collected = gc.collect()
    print("Garbage collector: collected", "%d objects.\n" % collected)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action="store_true", dest="test")
    parser.add_argument('-wl','--wallclock_limit', type=int, dest="wallclock_limit", default=1800,
                        help="Wallclock limit to run search algorithm per dataset")
    parser.add_argument('-tk', '--task', type=str, dest="task", choices=[
                        'cls', 'reg'], default='cls', help="Whether to perform Classification or Regression on the dataset")
    parser.add_argument('-st', '--search_type', type=str, dest="type", choices=[
                        'Random', 'BOHB', 'DEHB', 'All'], default='All', help="Type of search strategy to run")
    parser.add_argument('-p', '--path', type=str, dest="path", default='./output', help="Output folder to get the files from")

    args = parser.parse_args()
    seed = 42
    scores: dict[str, float] = {
        "RS_Score": [],
        "RS_Classifier": [],
        "BOHB_Score": [],
        "BOHB_Classifier": [],
        "DEHB_Score": [],
        "DEHB_Classifier": []
    }

    logger = logging.getLogger("HPO")
    logging.basicConfig(level=logging.INFO)

    task = "Classification" if args.task == "cls" else "Regression"
    cs = _ESTIMATORS[task].space()
    cs.seed(seed)
    wallclock_limit = args.wallclock_limit

    # SMAC scenario dictionary
    scenario = {
        "run_obj": "quality",  # we optimize quality (alternative to runtime),
        # "ta_run_limit": 100,
        "wallclock_limit": wallclock_limit,  # Maximum amount of wallclock-time used for optimization
        "cs": cs,  # configuration space,
        "deterministic": True,
        "memory_limit": 5120,
        "abort_on_first_run_crash": False,
        # "algo_runs_timelimit": 300, # Maximum amount of CPU-time used for optimization
        "cutoff": 3, # Maximum runtime, after which the target algorithm is cancelled
        "limit_resources": True
    }

    # Based on the task get the config space and datasets
    if task == "Classification":
        datasets = CAT_DATASETS
    else:
        datasets = REG_DATASETS

    # Loop through the datasets and find the best config
    for dataset_name in datasets:
        print(dataset_name)
        dataset = Dataset.from_pmlb(dataset_name, task)
        # Run dataset.info() to see some basic information about the dataset.
        print("="*40)
        dataset.info()
        print("="*40)
        dataset.pre_processing()
        # Get the data split
        train, test = dataset.split([0.8, 0.2])

        # Getting feature importances, sorting it, and filtering out features till the 96th percentile
        if task == "Classification":
            feature_importances: pd.DataFrame = dataset.get_feature_imp(train).sort_values('importance', ascending=False)
            feature_importances["accumulated"] = list(itertools.accumulate(feature_importances.values.ravel(), lambda x, y: x+y))
            feature_importances = feature_importances.query("accumulated < 0.96")
            # Only use the above features for training and testing
            train.X = train.X[feature_importances.index]
            test.X = test.X[feature_importances.index]
        print("Training: features shape = ", train.X.to_numpy().shape, ", target values shape = ", test.X.to_numpy().shape)

        # Setting the output directory for the runs and incumbent configs
        scenario["output_dir"] = os.path.join(current_dir, f"{args.path}/{dataset_name}")

        if args.test:
            output_path = os.path.join(current_dir, f"{scenario['output_dir']}/incumbents")
            print(f"-------------- Testing -------------------")
            scores = tester(output_path, train, test, task, args.type, scores, seed)
        else:
            # Creating folder if it doesnt already exist
            if not os.path.exists(f"{scenario['output_dir']}/incumbents"):
                os.makedirs( f"{scenario['output_dir']}/incumbents")

            if args.type == "Random" or args.type == "All":
                print("="*40, "\nRandom Search\n", "="*40)
                RS_best_config = smac_search(cs, "Random", scenario, train, task, seed, current_dir, trainer)
                call_gc()
            if args.type == "BOHB" or args.type == "All":
                print("="*40, "\nBOHB Search\n", "="*40)
                BOHB_best_config = smac_search(cs, "BOHB", scenario, train, task, seed, current_dir, trainer)
                call_gc()
            if args.type == "DEHB" or args.type == "All":
                print("="*40, "\nDEHB Search\n", "="*40)
                DEHB_best_config = dehb_search(cs, scenario, train, task, seed, current_dir, trainer)
                call_gc()

    # Print test results for all datasets
    if args.test:
        results = pd.DataFrame(columns=list(scores.keys()), index=datasets, data=scores)
        # This will help with plotting. Saving in a folder only if all search types are checked.
        if args.type == "All":
            save_dir = os.path.dirname(os.path.join(current_dir, f"{scenario['output_dir']}"))
            with open(os.path.join(current_dir, f"{save_dir}/test_results.json"), 'w') as f:
                json.dump(results.to_json(), f)
                print("Test results saved!\n", "="*40)
        print(results)
