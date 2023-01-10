import os
import argparse
import json
import copy
import logging
import time
import itertools
from functools import partial
import gc

import pdb
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score
from smac.facade.smac_mf_facade import SMAC4MF
from smac.facade.roar_facade import ROAR
from smac.scenario.scenario import Scenario
from dehb import DEHB
from ConfigSpace import ConfigurationSpace
from ConfigSpace.read_and_write import json as cs_json

from src.estimators import Classifiers, CAT_DATASETS
from src.data import Dataset, Split

current_dir = os.getcwd()

_ESTIMATORS = {
    "Classification": Classifiers
}

def get_estimator_instance(config: dict, task: str, data: np.ndarray) -> tuple:
    base_estimator = config.pop("base_estimator")
    
    # The configuration space for some of the estimators have custom parameters because of a 
    # parameter naming clash with other estimators. Dealing with that here.
    if base_estimator == "MLP":
        num_layers = config.pop("n_layer")
        num_neurons = config.pop("n_neurons")
        config.update(hidden_layer_sizes=[num_neurons] * num_layers)

    if base_estimator == "SVM":
        # Prefer dual=False when n_samples > n_features
        dual_val = data.shape[0] < data.shape[1]
        class_weight = config.pop("class_weight")
        class_weight = None if class_weight == "None" else class_weight
        config.update(dual=dual_val, class_weight=class_weight, max_iter=config.pop("max_iter_svm"))

    # return an instance of the estimator
    return _ESTIMATORS[task](base_estimator, config)

def trainer(cfg, search_type, task_type="Classification", train: pd.DataFrame = None, **kwargs):
    """
        Creates an instance of the Estimator and fits the given data on it.

        Parameters
        ----------
        cfg: Configuration
        seed: RandomState
        budget: float
            Max iterations

        Returns
        -------
        tuple : (1 - score, additional_info)
    """
    assert train is not None
    # Get the classifier instance
    config_dict = cfg.get_dictionary()
    train_X = train.X.values
    train_y = train.y.values.ravel()
    # Passing a copy of the config dict since we are overriding some values due to name clash
    estimator = get_estimator_instance(copy.deepcopy(config_dict), task_type, train_X)
    estimator_type = config_dict["base_estimator"]
    print(f"------Estimator : {estimator_type}--------")
    print(config_dict)

    start = time.time()

    # If the size of the dataset is greater than 100k records, 
    # use hold-one-out for validation. Else use KFold cross-validation.
    try:
        if train_X.shape[0] > 100000:
            train_X_t, train_X_v, train_y_t, train_y_v = train_test_split(train_X, train_y, train_size=0.8, random_state=42)
            estimator.fit(train_X_t, train_y_t)
            val_score = f1_score(train_y_v, estimator.predict(train_X_v), average='micro')
        else:
            kf = KFold(n_splits=5)
            scores = []
            for train_indices, val_indices in kf.split(train_X):
                estimator.fit(train_X[train_indices], train_y[train_indices])
                score = f1_score(train_y[val_indices], estimator.predict(train_X[val_indices]), average='micro')
                scores.append(score)

            val_score = np.mean(scores)
    except Exception as e:
        print("------ Estimator training failed ------\n", e)
        val_score = 0

    cost = time.time() - start

    print(f"Estimator validation f1 score = {val_score}")
    del estimator

    if search_type == "DEHB":
        budget = kwargs['budget']
        result = {
            "fitness": -val_score,  # DE/DEHB minimizes
            "cost": cost,
            "info": {
                "f1_score": val_score,
                "budget": budget
            }
        }
    else:
        result = (1 - val_score, {"val_f1_score": val_score})

    return result

def tester(baseline_config, bohb_config, dehb_config, train, test, task_type, scores):
    """
        Tests the given config of a test dataset
    """
    assert train is not None
    assert test is not None

    train_X = train.X.values
    train_y = train.y.values.ravel()
    test_X = test.X.values
    test_y = test.y.values.ravel()

    # Get the estimator and the config used
    baseline_config.pop("search_type")
    bohb_config.pop("search_type")
    dehb_config.pop("search_type")
    baseline_estimator_type = baseline_config["base_estimator"]
    baseline_estimator = get_estimator_instance(baseline_config, task_type, train_X)
    bohb_estimator_type = bohb_config["base_estimator"]
    bohb_estimator = get_estimator_instance(bohb_config, task_type, train_X)
    dehb_estimator_type = dehb_config["base_estimator"]
    dehb_estimator = get_estimator_instance(dehb_config, task_type, train_X)


    # Train (since we are not saving the models itself)
    baseline_estimator.fit(train_X, train_y)
    bohb_estimator.fit(train_X, train_y)
    dehb_estimator.fit(train_X, train_y)

    # Test
    baseline_predictions = baseline_estimator.predict(test_X)
    bohb_predictions = bohb_estimator.predict(test_X)
    dehb_predictions = dehb_estimator.predict(test_X)

    # Get the f1 scores of both in the test dataset
    baseline_score = f1_score(test_y, baseline_predictions, average='micro')
    bohb_score = f1_score(test_y, bohb_predictions, average='micro')
    dehb_score = f1_score(test_y, dehb_predictions, average='micro')
    # print(f"BOHB Incumbent test f1 score = {bohb_score}")
    scores["RS_Score"].append(baseline_score)
    scores["BOHB_Score"].append(bohb_score)
    scores["DEHB_Score"].append(dehb_score)
    scores["RS_Classifier"].append(baseline_estimator_type)
    scores["BOHB_Classifier"].append(bohb_estimator_type)
    scores["DEHB_Classifier"].append(dehb_estimator_type)

    return scores

def flush_memory():
    """
        Calls garbage collector manually to free memory for next run
    """
    collected = gc.collect()
    print("Garbage collector: collected", "%d objects.\n" % collected)

def dehb_search(cs: ConfigurationSpace, scenario:dict, data: Split, task: str):
    """
        Runs DEHB for finding the best configuration that optimizes the 'trainer' function.
    """
    # 20% wallclock-limit as minimum budget
    min_budget, max_budget = (0.2 * scenario["wallclock_limit"], scenario["wallclock_limit"])
    os.environ["MALLOC_TRIM_THRESHOLD_ "] = "10"
    dehb = DEHB(
        f=partial(trainer, search_type="DEHB", task_type=task, train=data),
        cs=cs,
        min_budget=min_budget,
        max_budget=max_budget,
        eta=3,
        output_path=f"{scenario['output_dir']}/dehb_run",
        n_workers=1                # set to >1 to utilize parallel workers
    )
    _, _, _ = dehb.run(total_cost=scenario["wallclock_limit"], verbose=True, save_intermediate=False, max_budget=max_budget)

    incumbent = dehb.vector_to_configspace(dehb.inc_config)

    # store the optimal configuration to disk
    opt_config = incumbent.get_dictionary()
    opt_config.update({"search_type": "DEHB"})
    with open(os.path.join(current_dir, f"{scenario['output_dir']}/incumbents/DEHB_opt_cfg.json"), 'w') as f:
        json.dump(opt_config, f)

def smac_search(cs: ConfigurationSpace, search_type: str, scenario:dict, data: Split, task: str, seed: int):
    """
        Runs SMAC3 for finding the best configuration that optimizes the 'trainer' function.

        Parameters
        ----------
        cs: ConfigurationSpace
            The Hyperparameter configuration space to search in
        search_type: str
            Determines which SMAC3 facade to use and the search strategy
        scenario: dict
            The scenario dictionary that specifies the resource limitations for the run
        data: Split
            The training data that is to be used during training
        task: str
            The type of task, i.e, Classification or Regression
        seed: int

        Returns
        -------
        The incumbent hyperparameter configuration
    """
    # SMAC3 Scenario object
    smac_scenario = Scenario(scenario)

    if search_type == "Random":
        # Random search in SMAC
        smac = ROAR(
            scenario=smac_scenario,
            rng=seed,
            run_id=seed,
            tae_runner=partial(trainer, search_type=search_type, task_type=task, train=data),
            initial_design_kwargs={"configs": [cs.get_default_configuration()]}
        )
    else:
        # The Hyperband config
        intensifier_kwargs = {
            "initial_budget": 0.2 * scenario['wallclock_limit'],
            "max_budget": scenario['wallclock_limit'],
            "eta": 2
        }

        # This is the closest implementation of BOHB in SMAC3 (check the docs)
        smac = SMAC4MF(
            scenario=smac_scenario,
            rng=seed,
            run_id=seed,
            tae_runner=partial(trainer, search_type=search_type, task_type=task, train=data),
            intensifier_kwargs=intensifier_kwargs,
            initial_design_kwargs={'n_configs_x_params': 1, 'max_config_fracs': .2})

    _, def_value, _, def_addtn_vals = smac.get_tae_runner().run(config=cs.get_default_configuration(), seed=seed)

    # Start optimization
    try:  # try finally used to catch any interrupt
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    _, inc_value, _, addtn_vals = smac.get_tae_runner().run(config=incumbent, seed=seed)

    # store the optimal configuration to disk
    opt_config = incumbent.get_dictionary()
    opt_config.update({"search_type": search_type})
    with open(os.path.join(current_dir, f"{scenario['output_dir']}/incumbents/{search_type}_opt_cfg.json"), 'w') as f:
        json.dump(opt_config, f)

    del smac
    return opt_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action="store_true", dest="test")
    parser.add_argument('-t', '--task', type=str, dest="task", choices=[
                        'cls', 'reg'], default='cls', help="Whether to perform Classification or Regression on the dataset")

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

    # Based on the task get the config space and datasets
    if args.task == "cls":
        task = "Classification"
        datasets = CAT_DATASETS
        cs = _ESTIMATORS[task].space()
        cs.seed(seed)
        wallclock_limit = 900

        # Create the pcs file
        # with open('cls_configspace.json', 'w') as f:
        #     f.write(cs_json.write(cs, indent=2))

        # SMAC scenario dictionary
        scenario = {
            "run_obj": "quality",  # we optimize quality (alternative to runtime)
            # "ta_run_limit": 100,
            "wallclock_limit": wallclock_limit,  # Maximum amount of wallclock-time used for optimization
            "cs": cs,  # configuration space,
            # "pcs_fn": "cls_configspace.json",
            # "deterministic": False,
            "memory_limit": 5120,
            "abort_on_first_run_crash": False,
            "algo_runs_timelimit": 300, # Maximum amount of CPU-time used for optimization
            "cutoff": 120, # Maximum runtime, after which the target algorithm is cancelled
            "rand_prob": 0.3,
            "limit_resources": True
        }

        for dataset_name in datasets:
            dataset = Dataset.from_pmlb(dataset_name, task)
            # Run dataset.info() to see some basic information about the dataset.
            print("="*40)
            dataset.info()
            print("="*40)
            dataset.pre_processing()
            # Get the data split
            train, test = dataset.split([0.8, 0.2])

            # Getting feature importances, sorting it, and filtering out features till the 96th percentile
            feature_importances: pd.DataFrame = dataset.get_feature_imp(train).sort_values('importance', ascending=False)
            feature_importances["accumulated"] = list(itertools.accumulate(feature_importances.values.ravel(), lambda x, y: x+y))
            feature_importances = feature_importances.query("accumulated < 0.96")
            # Only use the above features for training and testing
            train.X = train.X[feature_importances.index]
            test.X = test.X[feature_importances.index]
            print("Training: features shape = ", train.X.to_numpy().shape, ", target values shape = ", test.X.to_numpy().shape)

            # Setting the output directory for the runs and incumbent configs
            scenario["output_dir"] = os.path.join(current_dir, f"output/{dataset_name}")
            if not os.path.exists(f"{scenario['output_dir']}/incumbents"):
                os.makedirs( f"{scenario['output_dir']}/incumbents")

            if args.test:
                output_path = os.path.join(current_dir, f"{scenario['output_dir']}/incumbents")
                dir_list = os.listdir(output_path)
                for incumbent_json in dir_list:
                    f = open(os.path.join(current_dir, f"{output_path}/{incumbent_json}"))
                    config_dict = json.loads(f.read())
                    if config_dict["search_type"] == "BOHB":
                        BOHB_best_config = config_dict
                    elif config_dict["search_type"] == "Random":
                        RS_best_config = config_dict
                    else:
                        DEHB_best_config = config_dict
                print(f"-------------- Testing -------------------")
                scores = tester(RS_best_config.copy(), BOHB_best_config.copy(), DEHB_best_config.copy(), train, test, task, scores)
            else:
                print("="*40, "\nRandom Search\n", "="*40)
                # RS_best_config = smac_search(cs, "Random", scenario, train, task, seed)
                flush_memory()
                print("="*40, "\nBOHB Search\n", "="*40)
                BOHB_best_config = smac_search(cs, "BOHB", scenario, train, task, seed)
                flush_memory()
                print("="*40, "\nDEHB Search\n", "="*40)
                # DEHB_best_config = dehb_search(cs, scenario, train, task)
                flush_memory()
        results = pd.DataFrame(columns=list(scores.keys()), index=datasets, data=scores)
        print(results)
    else:
        task = "Regression"
