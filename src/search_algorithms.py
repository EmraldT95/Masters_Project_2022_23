import os, json
from functools import partial
from typing import Callable

from smac.facade.smac_mf_facade import SMAC4MF
from smac.facade.roar_facade import ROAR
from smac.scenario.scenario import Scenario
from dehb import DEHB
from ConfigSpace import ConfigurationSpace

from src.data import Split

def dehb_search(cs: ConfigurationSpace, scenario:dict, data: Split, task: str, seed: int, root_path: str, target_func: Callable):
    """
        Runs DEHB for finding the best configuration that optimizes the 'trainer' function.
    """
    # 20% wallclock-limit as minimum budget
    min_budget, max_budget = (0.2 * scenario["wallclock_limit"], scenario["wallclock_limit"])
    dimensions = len(cs.get_hyperparameters())
    # os.environ["MALLOC_TRIM_THRESHOLD_ "] = "10"
    dehb = DEHB(
        f=partial(target_func, search_type="DEHB", task_type=task, train=data, seed=seed),
        cs=cs,
        min_budget=min_budget,
        max_budget=max_budget,
        eta=3,
        strategy="randtobest1_bin",
        output_path=f"{scenario['output_dir']}/dehb_run"
    )
    _, _, _ = dehb.run(total_cost=scenario["wallclock_limit"], verbose=True, save_intermediate=False)

    incumbent = dehb.vector_to_configspace(dehb.inc_config)

    # store the optimal configuration to disk
    opt_config = incumbent.get_dictionary()
    opt_config.update({"search_type": "DEHB"})
    with open(os.path.join(root_path, f"{scenario['output_dir']}/incumbents/DEHB_opt_cfg.json"), 'w') as f:
        json.dump(opt_config, f)
    # This will help with plotting
    with open(os.path.join(root_path, f"{scenario['output_dir']}/dehb_run/DEHB_opt_cfg.json"), 'w') as f:
        json.dump(opt_config, f)

def smac_search(cs: ConfigurationSpace, search_type: str, scenario:dict, data: Split, task: str, seed: int, root_path: str, target_func: Callable):
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
        dcit: The incumbent hyperparameter configuration
    """

    smac_scenario = Scenario(scenario) # SMAC3 Scenario object
    tae_runner = partial(target_func, search_type=search_type, task_type=task, train=data, seed=seed)
    wallclock_lim = scenario['wallclock_limit']

    if search_type == "Random":
        # Random search in SMAC
        smac = ROAR(
            scenario=smac_scenario,
            rng=seed,
            run_id=seed,
            tae_runner=tae_runner,
            initial_design_kwargs={"configs": [cs.get_default_configuration()]}
        )
    else:
        # The Hyperband config
        intensifier_kwargs = {
            "initial_budget": 0.2 * wallclock_lim,
            "max_budget": wallclock_lim,
            "eta": 3
        }

        if wallclock_lim >= 3600:
            n_config_x_params = 10
        elif wallclock_lim >= 1800:
            n_config_x_params = 6
        else:
            n_config_x_params = 4

        # This is the closest implementation of BOHB in SMAC3 (check the docs)
        smac = SMAC4MF(
            scenario=smac_scenario,
            rng=seed,
            run_id=seed,
            tae_runner=tae_runner,
            intensifier_kwargs=intensifier_kwargs,
            initial_design_kwargs={'n_configs_x_params': n_config_x_params, 'max_config_fracs': .2})

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
    with open(os.path.join(root_path, f"{scenario['output_dir']}/incumbents/{search_type}_opt_cfg.json"), 'w') as f:
        json.dump(opt_config, f)
    # This will help with plotting
    with open(os.path.join(root_path, f"{smac_scenario.output_dir_for_this_run}/{search_type}_opt_cfg.json"), 'w') as f:
        json.dump(opt_config, f)

    del smac
    return opt_config
