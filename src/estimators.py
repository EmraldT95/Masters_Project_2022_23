import numpy as np

from sklearn.base import BaseEstimator
import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from src.model_config_space import KNN, MLP, NB, SVM, XGB

CAT_DATASETS = [
    'adult',
    # 'agaricus_lepiota',
    'cleveland',
    # 'biomed',
    # 'breast_cancer',
    # 'chess',
    # 'credit_g',
    # 'glass2',
    'fars',
    'connect_4'
]

_CLASSIFIERS = {
    "KNN": KNN,
    "MLP": MLP,
    "NB": NB,
    "SVM": SVM,
    "XGB": XGB
}

class Estimator(BaseEstimator):
    def __init__(self, name="", estimator=None):
        self._name = name
        self._estimator = estimator
        super().__init__()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Trains the classifier with the given dataset

        Args:
            X: The features of the training dataset
            y: The labels of the training dataset
        """

        self._estimator.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the output given by the trained classifier for a given
        set of input features. The predicted value is then used to calculate
        f1_score by comparing it to the actual output.

        Args:
            X: The features of the test dataset
        """

        return self._estimator.predict(X)

    def get_params(self, deep=True):
        """Gets the parameters used in the estimators
        """

        # params = self._estimator.get_params()
        return super().get_params(deep)

class Classifiers(Estimator):
    """This class initializes each of the individual base classifiers. This helps deal
    with multi-label problem more easily as it transforms the dataset provided locally,
    and then deal with it as multiple single-label problems. Essentially, it does the
    pre-processing of the data to fit its needs.

    In case of MLP, we don't really wrap around one of the problem transformer bases as
    it is known to work well with raw data and learn different levels of representation
    of the data given.

    Args:
        name: The type of base classifier to use
        hyperparameters: The hyperparameters to be used for the classifier
    """

    def __init__(self, name, hyperparameters):
        self._name = name
        self._hyperparameters = hyperparameters
        assert self._name in _CLASSIFIERS.keys()

        # if hyperparameters are passed, else use default hyperparameters
        if self._hyperparameters:
            self._base_classifier = _CLASSIFIERS[self._name](**self._hyperparameters)
        else:
            self._base_classifier = _CLASSIFIERS[self._name]()
        super().__init__(self._name, self._base_classifier)
    
    @staticmethod
    def space() -> CS:
        cs = ConfigurationSpace()
        # Adding the list of classifiers as a categorical hyperparameters
        classifier_types = [key for key in list(_CLASSIFIERS.keys())]
        classifier = CategoricalHyperparameter("base_estimator", classifier_types)
        cs.add_hyperparameter(classifier)

        # Adding the config space of all the respective classifiers
        for key in classifier_types:
            cs.add_configuration_space(
                prefix="",
                delimiter="",
                configuration_space=_CLASSIFIERS.get(key).space(),
                parent_hyperparameter={"parent": cs["base_estimator"], "value": key},
            )
        return cs

    @property
    def estimator(self):
        return self._estimator

    @property
    def base_classifier(self):
        return self._base_classifier
