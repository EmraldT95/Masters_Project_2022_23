import numpy as np

from sklearn.base import BaseEstimator
import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from src.model_config_space import KNN_Cls, KNN_Reg, MLP_Cls, MLP_Reg, NB, SVM_Cls, SVM_Reg, XGB_Cls, XGB_Reg, Bayesian_Ridge_Reg

CAT_DATASETS = [
    'adult',
    'agaricus_lepiota',
    'analcatdata_authorship',
    'cleveland',
    'car_evaluation',
    'biomed',
    'breast_cancer',
    'chess',
    'credit_g',
    'glass2',
    'coil2000',
    'dermatology',
    'GAMETES_Epistasis_2_Way_20atts_0.1H_EDM_1_1',
    'german',
    'iris',
    'magic',
    'letter',
    'connect_4',
    'wine_quality_white',
    'yeast'
]

REG_DATASETS = [
    '197_cpu_act',
    '201_pol',
    '215_2dplanes',
    '218_house_8L',
    '225_puma8NH',
    '230_machine_cpu',
    '294_satellite_image',
    '503_wind',
    '529_pollen',
    '562_cpu_small',
    '574_house_16H',
    '589_fri_c2_1000_25',
    '620_fri_c1_1000_25',
    '649_fri_c0_500_5',
    '654_fri_c0_500_10',
    '690_visualizing_galaxy',
    '1027_ESL',
    '1193_BNG_lowbwt',
    '1199_BNG_echoMonths',
    'banana',
]

_CLASSIFIERS = {
    "KNN": KNN_Cls,
    "MLP": MLP_Cls,
    "NB": NB,
    "SVM": SVM_Cls,
    "XGB": XGB_Cls
}

_REGRESSORS = {
    "KNN": KNN_Reg,
    "MLP": MLP_Reg,
    "BRR": Bayesian_Ridge_Reg,
    "SVM": SVM_Reg,
    "XGB": XGB_Reg,
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
    """
        This class initializes each of the individual base classifiers.

        Args:
            name: The type of base classifier to use
            hyperparameters: The hyperparameters to be used for the classifier
    """

    def __init__(self, name, hyperparameters):
        assert name in _CLASSIFIERS.keys(), "Classifier not in configspace"
        self._base_classifier = _CLASSIFIERS[name](**hyperparameters)
        super().__init__(name, self._base_classifier)
    
    @staticmethod
    def space() -> CS:
        """
            Returns the config space containing all the classifiers needed along
            with their parameters as children
        """
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
        return self._base_classifier

class Regressors(Estimator):
    """This class initializes each of the individual base regressors. 

    Args:
        name: The type of base classifier to use
        hyperparameters: The hyperparameters to be used for the classifier
    """

    def __init__(self, name, hyperparameters):
        assert name in _REGRESSORS.keys(), "Classifier not in configspace"
        self._base_classifier = _REGRESSORS[name](**hyperparameters)
        super().__init__(name, self._base_classifier)
    
    @staticmethod
    def space() -> CS:
        cs = ConfigurationSpace()
        # Adding the list of classifiers as a categorical hyperparameters
        classifier_types = [key for key in list(_REGRESSORS.keys())]
        classifier = CategoricalHyperparameter("base_estimator", classifier_types)
        cs.add_hyperparameter(classifier)

        # Adding the config space of all the respective classifiers
        for key in classifier_types:
            cs.add_configuration_space(
                prefix="",
                delimiter="",
                configuration_space=_REGRESSORS.get(key).space(),
                parent_hyperparameter={"parent": cs["base_estimator"], "value": key},
            )
        return cs

    @property
    def estimator(self):
        return self._base_classifier
