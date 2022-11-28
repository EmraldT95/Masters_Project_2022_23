import argparse
from dataclasses import dataclass
from typing import Iterable, Any, Dict

import yaml
import pandas as pd
import numpy as np
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

CAT_DATASETS = [
    'adult',
    'agaricus_lepiota',
    'analcatdata_asbestos'
]

@dataclass
class Split:
    X: np.ndarray
    y: np.ndarray

@dataclass
class Dataset:
    name: str
    task: str
    dataset: pd.DataFrame
    metadata: Dict
    features: pd.DataFrame
    target: pd.DataFrame
    imp_features: np.ndarray

    def split(
        self,
        splits: Iterable[float],
        seed: int = 1,
    ) -> tuple[Split, ...]:
        """Create splits of the data

        Parameters
        ----------
        splits : Iterable[float]
            The percentages of splits to generate

        seed : int | None = None
            The seed to use for the splits

        Returns
        -------
        tuple[Split, ...]
            The collected splits
        """
        splits = list(splits)
        assert abs(1 - sum(splits)) <= 1e-6, "Splits must sum to 1"

        sample_sizes = tuple(int(s * len(self.features)) for s in splits)

        collected_splits = []

        next_xs = self.features.to_numpy()
        next_ys = self.target.to_numpy()

        for size in sample_sizes[:-1]:
            xs, next_xs, ys, next_ys = train_test_split(
                next_xs, next_ys, train_size=size, random_state=seed
            )
            collected_splits.append(Split(X=xs, y=ys))
        collected_splits.append(Split(X=next_xs, y=next_ys))

        return tuple(collected_splits)

    def pre_processing(self) -> None:
        """
        Doing some preprocessing to the dataset to make it easier to model
        using different ML algorithms
        """
        features = self.features.columns.to_list()
        del_names = []
        ctg_features = []
        for name in features:
            # Delete columns that have the same value for all instances
            unq_features, _ = np.unique(self.features[name], return_counts=True)
            if len(unq_features) == 1:
                del_names.append(name)

            # Finding categorical features to one-hot encode. Oridinal features left as is
            # since the order might matter in some cases
            if name not in del_names and self.metadata.get(name)["type"] in ["categorical", "object"]:
                ctg_features.append(name)

        print(f"PRE_PROCESSING: No. of deleted features with the same value for all entries - {len(del_names)}")
        self.features = self.features.drop(del_names, axis=1)

        # One-hot encode categorical features.
        print(f"PRE_PROCESSING: No. of one-hot encoded features - {len(ctg_features)}")
        for feature_name in ctg_features:
            feature = self.features[feature_name]
            self.features = self.features.drop(feature_name, axis=1)
            ohe_features = pd.get_dummies(feature, feature_name)
            self.features = pd.concat([self.features, ohe_features], axis=1)

        # Move unique instances to the test set

        # Remove features that have more than 60% distinct values, if categorical

    def get_feature_imp(self):
        """
        Find feature importance using RandomForest classifier
        """
        train, test = self.split([0.2, 0.8])
        rf = RandomForestClassifier(max_depth=10, random_state=42, n_estimators = 300).fit(train.X, train.y)
        feature_importances = pd.DataFrame(
                                    rf.feature_importances_,
                                    index=self.features.columns.to_list(),
                                    columns=['importance']
                                ).sort_values('importance', ascending=False)
        print(feature_importances)

    def info(self) -> None:
        """
        Prints some basic information about the target values
        """
        print(f'Dataset: "{self.name}"')
        print("Total no. of Instances: ", len(self.target.values))
        print("Total no. of Features: ", len(self.features.columns.to_list()))
        # This information is needed only in the case of classification task
        if self.task == "Classification":
            unq_targets, unq_targets_count = np.unique(self.target, return_counts=True)
            print("Unique Target values: \n",
                pd.DataFrame(data=unq_targets_count, index=unq_targets, columns=["Count"]))
            print(unq_targets)

        # Adding the actual feature type information to the Dataframe
        pd_desc = self.dataset.describe().transpose()
        col_type = []
        col_desc = []
        for idx in pd_desc.index:
            col_type.append(self.metadata.get(idx)["type"])
            col_desc.append(self.metadata.get(idx)["desc"])
        pd_desc["type"] = col_type
        pd_desc["description"] = col_desc
        print(pd_desc, '\n')

    @staticmethod
    def from_pmlb(name: str, task: str, cache_dir='../datasets'):
        """
        Fetches the PMLB dataset from the repository. A helper function is provided
        by the `pmlb` package itself to fetch the data. A

        Args:
            name (str): Name of the dataset to fetch
            task (str): Classification or Regression 
            cache_dir (str): The location where the pulled datasets are stored locally.
        """
        dataset = fetch_data(name, local_cache_dir=cache_dir, dropna=True)
        features = dataset.drop('target', axis=1)
        target = dataset['target']

        # Get the metadata file so that we can determine the correct feature/target types
        # for each of the datasets. The dtype given in the pandas dataframe is unfortunately
        # not enough to determine the actual intended type of each of the columns 
        try:
            url = f'https://raw.githubusercontent.com/EpistasisLab/pmlb/master/datasets/{name}/metadata.yaml'
            response= requests.get(url)
            # Save the YAML file
            metadata_file = open(f'../datasets/{name}/metadata.yaml','w')
            metadata_file.write(response.text)
            # Getting the names and types of the features and target variable
            metadata_yaml = yaml.safe_load(response.text)
            metadata = {}
            metadata["target"] = {
                "type": metadata_yaml["target"]["type"],
                "desc": metadata_yaml["target"]["description"]
            }
            for col in metadata_yaml["features"]:
                metadata[col["name"]] = {
                    "type": col["type"],
                    "desc": col["description"]
                }
        except:
            raise ValueError("Failed to get metadata. The resulting dataset might be erroneous.")
        
        return Dataset(
            name=name,
            task=task,
            dataset=dataset,
            features=features,
            target=target,
            metadata=metadata,
            imp_features=np.array([])
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, dest="task", choices=[
                        'cls', 'reg'], default='cls', help="Whether to perform Classification or Regression on the dataset")

    args = parser.parse_args()
    # rand_datasets = random.choices(dataset_names, k=7)
    # print('7 arbitrary datasets from PMLB:\n', '\n '.join(rand_datasets))

    # # Determine the actual task
    if args.task == "cls":
        task = "Classification"
        # dataset_list = classification_dataset_names[:3]
    else:
        task = "Regression"
        # dataset_list = regression_dataset_names[:3]

    # Fetch the datasets and get some basic information about them
    dataset_list = CAT_DATASETS
    
    import requests
    for dataset in dataset_list:
        curr_dataset = Dataset.from_pmlb(dataset, task)
        curr_dataset.info()
        curr_dataset.pre_processing()
        curr_dataset.get_feature_imp()
        # # curr_dataset.pre_processing()
        # print(curr_dataset.dataset.attrs())
        print("===================================\n")
