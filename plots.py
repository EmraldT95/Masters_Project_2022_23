import os
import pickle
import matplotlib.pyplot as plt
import argparse
import json
from datetime import datetime
import numpy as np

COLORS = ["red", "blue", "gold"]

def plot_pareto_front(runs: list, run_path: str, dataset_name: str):

    search_type = "DEHB"
    plot_color = COLORS[2]
    # Go through each run directory
    for run_dir in runs:
        run_files_path = os.path.join(run_path, run_dir)
        run_files = os.listdir(run_files_path)
        BOHB_check = [inc_file for inc_file in run_files if "BOHB" in inc_file]
        Random_check = [inc_file for inc_file in run_files if "Random" in inc_file]
        run_start_time = 0
        overall_time = 0

        # Find what type of search it was based on the incumbent file
        if len(BOHB_check) > 0:
            search_type = "BOHB"
            plot_color = COLORS[1]
        elif len(Random_check) > 0:
            search_type = "Random"
            plot_color = COLORS[0]

        if run_dir != "incumbents":
            # BOHB and Random Search directories
            if "run_42" in run_dir:
                with open(os.path.join(run_files_path, "runhistory.json"), 'r') as f:
                    history = json.load(f)
                    data = history['data']
                    run_start_time = datetime.fromtimestamp(data[0][1][3])
                    pareto_front = np.array([[data[0][1][0], 0.0]]) # cost, start_time
                    non_pareto = np.array([])
                    # print(data)
                    
                    # Loop through the data and find the Pareto points
                    for i, entry in enumerate(data[1:]):
                        cost = entry[1][0]
                        start_time = datetime.fromtimestamp(entry[1][3])
                        delta = start_time - run_start_time
                        overall_time = entry[0][3] if entry[0][3] > overall_time else overall_time
                        # Check if the cost reduced from the last entry in the pareto front
                        if cost <= 1.0:
                            if cost < pareto_front[-1][0]:
                                pareto_front = np.append(pareto_front, np.array([[cost, delta.total_seconds()]]), axis=0)
                            # else:
                            #     non_pareto = np.append(non_pareto, np.array([[cost, delta.total_seconds()]])).reshape(-1,2)

            else:
                pkl_files = [run_file for run_file in run_files if "history" in run_file]
                pkl_files = sorted(pkl_files)
                with open(os.path.join(run_files_path, pkl_files[-1]), 'rb') as f:
                    data = pickle.load(f)

                run_start_time = run_start_time + data[0][2]
                pareto_front = np.array([[data[0][1], 0.0]]) # cost, start_time
                non_pareto = np.array([])
                
                # Loop through the data and find the Pareto points
                for i, entry in enumerate(data[1:]):
                    cost = entry[1]
                    # Check if the cost reduced from the last entry in the pareto front
                    if cost <= 1.0:
                        if cost < pareto_front[-1][0]:
                            pareto_front = np.append(pareto_front, np.array([[cost, run_start_time]]), axis=0)
                        # else:
                        #     non_pareto = np.append(non_pareto, np.array([[cost, run_start_time]])).reshape(-1,2)
                    run_start_time = run_start_time + entry[2]
                

            # plt.scatter(non_pareto[:,1], non_pareto[:,0], alpha=0.5, s=5)
            plt.scatter(pareto_front[:,1], pareto_front[:,0], s=15, color=plot_color)
            plt.step(pareto_front[:,1], pareto_front[:,0], where="post", label=search_type, color=plot_color)

    plt.legend(bbox_to_anchor=(1, 1))
    # plt.tight_layout()
    plt.xticks(np.arange(0, overall_time + 1, 300))
    plt.title(f'Pareto front for dataset - {dataset_name}')
    plt.grid(True)
    plt.ylabel('Cost')
    plt.xlabel('Time(s)')
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
            BOHB_check = [inc_file for inc_file in run_files if "BOHB" in inc_file]

            if run_dir != "incumbents":
                # BOHB and Random Search directories
                if "run_42" in run_dir:
                    with open(os.path.join(run_files_path, "traj.json"), 'r') as f:
                        for line in f:
                            pass
                        last_line = json.JSONDecoder().decode(line)
                        # Find what type of search it was based on the incumbent file
                        if len(BOHB_check) > 0:
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

    avg_scores = np.array([Random_score.mean(), BOHB_score.mean(), DEHB_score.mean()])
    test_avg_scores = np.array([test_Random_score.mean(), test_BOHB_score.mean(), test_DEHB_score.mean()])
    labels = ['Random', 'BOHB', 'DEHB']

    fig, ax = plt.subplots()
    ax.set_ylabel('f1/r2 Score')

    if with_test:
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width/2, avg_scores, width, label="Train")
        ax.bar(x + width/2, test_avg_scores, width, label="Test")
        ax.set_xticks(x, labels)
        ax.set_title('Average train/test incumbent scores \nfor all search types across all datasets.')
        ax.legend()
        plt.legend(bbox_to_anchor=(1, 1))
        plt.tight_layout()
    else:
        ax.bar(labels, avg_scores, color=COLORS)
        ax.set_title('Average train incumbent scores for all search types across all datasets.')
    
    plt.grid(True)
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('--test', action="store_true", dest="test")
    parser.add_argument('-p', '--path', type=str, dest="path", default='./output', help="Output folder to get the files from.")
    parser.add_argument('-pt', '--plot', type=str, dest="plot", choices=[
                        'Pareto', 'Avg', 'All'], default='All', help="The plots you want to generate.")
    parser.add_argument('--test', action="store_true", dest="test")

    args = parser.parse_args()
    arg_path = args.path
    datasets = os.listdir(arg_path)
    datasets = [dataset for dataset in datasets if dataset != "test_results.json"]

    if args.plot == "Pareto" or args.plot == "All":
        for dataset in datasets[4:9]:
            run_path = os.path.join(arg_path, dataset)
            runs = os.listdir(run_path)
            plot_pareto_front(runs, run_path, dataset)

    if args.plot == "Avg" or args.plot == "All":
        average_plot(arg_path, datasets, args.test)
