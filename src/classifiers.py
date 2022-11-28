import numpy as np

from src.model_config_space import KNN, MLP, NB, SVM, RF

_CLASSIFIERS = {
    "KNN": KNN,
    "MLP": MLP,
    "NB": NB,
    "SVM": SVM,
    "RF": RF
}


class Classifiers:
    """This class initializes each of the individual base classifiers and also wraps
    it around a problem transformation base class from skmultilearn. This helps deal
    with multi-label problem more easily as it transforms the dataset provided locally,
    and then deal with it as multiple single-label problems. Essentially, it does the
    pre-processing of the data to fit its needs.

    In case of MLP, we don't really wrap around one of the problem transformer bases as
    it is known to work well with raw data and learn different levels of representation
    of the data given.

    Args:
        name: The type of base classifier to use
        hyperparameters: The hyperparameters to be used for the classifier
        prob_trans: The problem transformation base to apply
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
        self._estimator = self._base_classifier

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

    def get_params(self):
        """Gets the parameters used in the base classifier
        """

        params = self._base_classifier.get_params()
        return params

    @property
    def estimator(self):
        return self._estimator

    @property
    def base_classifier(self):
        return self._base_classifier
