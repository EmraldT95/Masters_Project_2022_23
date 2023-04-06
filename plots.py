import os
import pickle
import matplotlib.pyplot as plt
import argparse
import json
from datetime import datetime
import numpy as np

COLORS = {"Random": "red", "BOHB": "blue", "DEHB": "gold"}

def find_search_type(run_files: list[str]):
    """
        Finds the search algorithm type that is associated with the data
        based on the incumbent file.

        Parameters
        ----------
        run_files: list[str]
            The list of files in the run folder
    """
    for inc_file in run_files:
        if "opt_cfg" in inc_file:
            if "BOHB" in inc_file:
                return "BOHB"
            elif "Random" in inc_file:
                return "Random"
            else:
                return "DEHB"

def plot_cost_vs_time(runs: list, run_path: str, dataset_name: str):
    # Go through each run directory
    for run_dir in runs:
        run_files_path = os.path.join(run_path, run_dir)
        run_files = os.listdir(run_files_path)
        run_start_time = 0
        search_type = find_search_type(run_files)

        if run_dir != "incumbents":
            costs = []
            time = []

            # BOHB and Random Search directories
            if "run_42" in run_dir:
                with open(os.path.join(run_files_path, "runhistory.json"), 'r') as f:
                    history = json.load(f)
                    data = history['data']
                    run_start_time = datetime.fromtimestamp(data[0][1][3])
                    
                    # Loop through the data and find the Pareto points
                    for i, entry in enumerate(data):
                        cost = entry[1][0]
                        start_time = datetime.fromtimestamp(entry[1][3])
                        delta = start_time - run_start_time
                        # Check if the cost reduced from the last entry in costs
                        if cost <= 1.0:
                            if len(costs) == 0 or cost < costs[-1]:
                                costs.append(cost)
                                time.append(delta.total_seconds())
            # DEHB
            else:
                pkl_files = [run_file for run_file in run_files if "history" in run_file]
                with open(os.path.join(run_files_path, pkl_files[-1]), 'rb') as f:
                    data = pickle.load(f)
                
                for i, entry in enumerate(data):
                    cost = entry[1]
                    run_start_time = run_start_time + entry[2]
                    # For Regression, we use the r2_score. The cost is 1 - r2_score
                    # and hence can have a value greater than 1, which is a bad score.
                    # Hence is ignored from the plot.
                    if cost <= 1.0:
                        # Appending first enty as is
                        if len(costs) == 0 or cost < costs[-1]:
                            costs.append(cost)
                            time.append(run_start_time)
                        
            plt.scatter(time, costs, s=15, color=COLORS[search_type])
            plt.step(time, costs, where="post", label=search_type, color=COLORS[search_type])

    plt.legend(bbox_to_anchor=(1, 1))
    plt.title(f'Optimization Cost vs. Time for the dataset - {dataset_name}')
    plt.grid(True)
    plt.ylabel('Cost')
    plt.xlabel('Time(s)')
    # plt.savefig(f'{dataset_name}.png')
    plt.show()

def average_plot(output_path: str, datasets: list, with_test: bool):
    BOHB_score = np.array([])
    DEHB_score = np.array([])
    Random_score = np.array([])
    test_Random_score = np.array([])
    test_BOHB_score = np.array([])
    test_DEHB_score = np.array([])

    # Go through each run directory
    for dataset in datasets:
        run_path = os.path.join(output_path, dataset)
        runs = os.listdir(run_path)
        for run_dir in runs:
            run_files_path = os.path.join(run_path, run_dir)
            run_files = os.listdir(run_files_path)
            search_type = find_search_type(run_files)

            if run_dir != "incumbents":
                # BOHB and Random Search directories
                if "run_42" in run_dir:
                    with open(os.path.join(run_files_path, "traj.json"), 'r') as f:
                        for line in f:
                            pass
                        last_line = json.JSONDecoder().decode(line)
                        # Find what type of search it was based on the incumbent file
                        if search_type == "BOHB":
                            BOHB_score = np.append(BOHB_score, [1 - last_line['cost']])
                        else:
                            Random_score = np.append(Random_score, [1 - last_line['cost']])
                else:
                    incumbent_file = [inc_file for inc_file in run_files if "incumbent" in inc_file]
                    with open(os.path.join(run_files_path, incumbent_file[0]), 'r') as f:
                        inc = json.load(f)
                        DEHB_score = np.append(DEHB_score, 1 - inc['score'])
            else:
                with open(os.path.join(output_path, "test_results.json"), 'r') as f:
                    test_results = json.JSONDecoder().decode(json.load(f))
                    test_Random_score = np.array(list(test_results["RS_Score"].values()))
                    test_BOHB_score = np.array(list(test_results["BOHB_Score"].values()))
                    test_DEHB_score = np.array(list(test_results["DEHB_Score"].values()))

    avg_scores = np.array([Random_score.mean().round(3), BOHB_score.mean().round(3), DEHB_score.mean().round(3)])
    test_avg_scores = np.array([test_Random_score.mean().round(3), test_BOHB_score.mean().round(3), test_DEHB_score.mean().round(3)])
    labels = ['Random(ROAR)', 'BOHB', 'DEHB']

    fig, ax = plt.subplots()
    ax.set_ylabel('f1/r2 Score')

    if with_test:
        x = np.arange(len(labels))
        width = 0.35
        train_scores = ax.bar(x - width/2, avg_scores, width, label="Train")
        ax.bar_label(train_scores, padding=3)
        test_scores = ax.bar(x + width/2, test_avg_scores, width, label="Test")
        ax.bar_label(test_scores, padding=3)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1)
        ax.set_title('Average train/test incumbent scores \nfor all search types across all datasets.')
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
        plt.tight_layout()
    else:
        scores = ax.bar(labels, avg_scores, color=COLORS[search_type])
        ax.bar_label(scores, padding=3)
        ax.set_title('Average train incumbent scores for all search types across all datasets.')
    
    plt.grid(True)
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('--test', action="store_true", dest="test")
    parser.add_argument('-p', '--path', type=str, dest="path", default='./output', help="Output folder to get the files from.")
    parser.add_argument('-pt', '--plot', type=str, dest="plot", choices=[
                        'CostvTime', 'Avg', 'All'], default='All', help="The plots you want to generate - Pareto, Avg or All (default).")
    parser.add_argument('--test', action="store_true", dest="test")

    args = parser.parse_args()
    arg_path = args.path
    datasets = os.listdir(arg_path)
    datasets = [dataset for dataset in datasets if dataset != "test_results.json"]

    if args.plot == "CostvTime" or args.plot == "All":
        for dataset in datasets:
            run_path = os.path.join(arg_path, dataset)
            runs = os.listdir(run_path)
            plot_cost_vs_time(runs, run_path, dataset)

    if args.plot == "Avg" or args.plot == "All":
        average_plot(arg_path, datasets, args.test)
