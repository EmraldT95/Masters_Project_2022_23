from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import BayesianRidge
from xgboost import XGBClassifier, XGBRegressor
import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter, OrdinalHyperparameter
from ConfigSpace.conditions import NotEqualsCondition, EqualsCondition, InCondition

from typing import Any

class KNN_Cls(KNeighborsClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def space() -> CS:
        n_neighbors = UniformIntegerHyperparameter("n_neighbors", lower=2, upper=15, log=False)
        weights = CategoricalHyperparameter("weights", ['uniform', 'distance'], default_value='uniform')
        # weights = CategoricalHyperparameter("weights", ['uniform', 'distance'], default_value='uniform', weights=[0.3, 0.7])
        algorithm = CategoricalHyperparameter("algorithm", ['auto', 'ball_tree', 'kd_tree', 'brute'], default_value='auto')
        leaf_size = UniformIntegerHyperparameter("leaf_size", lower=10, upper=70, default_value=30, log=False)
        p = UniformIntegerHyperparameter("p", lower=1, upper=2, default_value=2)  # Manhattan or Euclidean

        cs = ConfigurationSpace()
        cs.add_hyperparameters([weights, n_neighbors, algorithm, leaf_size, p])
        # the leaf size is considered only if the algorithm is Ball Tree or KD Tree
        # ('Auto' can choose these or brute force, hence the not equals condition)
        cond_1 = NotEqualsCondition(leaf_size, algorithm, "brute")
        cs.add_conditions([cond_1])
        return cs

class XGB_Cls(XGBClassifier):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(predictor='cpu_predictor', **kwargs)

    @staticmethod
    def space() -> CS:
        n_estimators = UniformIntegerHyperparameter("n_estimators", lower=2, upper=20, default_value=10)
        max_depth = UniformIntegerHyperparameter("max_depth", lower=5, upper=20, default_value=5)
        subsample = UniformFloatHyperparameter("subsample", lower=0.1, upper=0.8, default_value=0.2)
        gamma = UniformFloatHyperparameter("gamma", lower=0.1, upper=1, default_value=0.1)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([n_estimators, subsample, max_depth, gamma])
        return cs

class MLP_Cls(MLPClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def space() -> CS:
        n_layer = UniformIntegerHyperparameter("n_layer", 1, 3, default_value=2)
        n_neurons = UniformIntegerHyperparameter("n_neurons", 10, 100, log=True, default_value=10)
        activation = CategoricalHyperparameter("activation", ['logistic', 'tanh', 'relu'], default_value='tanh')
        solver = CategoricalHyperparameter("solver", ['sgd', 'adam'], default_value='adam')
        alpha = UniformFloatHyperparameter("alpha", lower=0.05, upper=2, default_value=0.1, log=True)  # L2 regularization strength
        learning_rate = CategoricalHyperparameter("learning_rate", ['constant', 'invscaling', 'adaptive'], default_value='constant')
        learning_rate_init = UniformFloatHyperparameter("learning_rate_init", lower=1e-5, upper=1e-1, default_value=1e-3, log=True)
        max_iter = UniformIntegerHyperparameter("max_iter", 2000, 15000, default_value=5000)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([n_layer, n_neurons, activation, solver, alpha, learning_rate, learning_rate_init, max_iter])
        cond_1 = InCondition(learning_rate_init, solver, ["adam", "sgd"])
        cond_2 = EqualsCondition(learning_rate, solver, "sgd")
        cs.add_conditions([cond_1, cond_2])
        return cs


class NB(GaussianNB):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def space() -> CS:
        var_smoothing = UniformFloatHyperparameter("var_smoothing", lower=1e-11, upper=1e-07, default_value=1e-09)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([var_smoothing])
        return cs


class SVM_Cls(LinearSVC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def space() -> CS:
        multi_class = CategoricalHyperparameter("multi_class", ['ovr', 'crammer_singer'], default_value='ovr')
        tolerance = UniformFloatHyperparameter("tol", lower=1e-4, upper=1e-3, default_value=1e-4)
        intercept_scaling = UniformFloatHyperparameter("intercept_scaling", lower=0.5, upper=2, default_value=1.0)
        class_weight = CategoricalHyperparameter("class_weight", ['balanced', 'None'], default_value='None')
        max_iter_svm = UniformIntegerHyperparameter("max_iter_svm", 2000, 15000, default_value=5000)
        C = UniformFloatHyperparameter("C", lower=0.5, upper=2, default_value=1.0)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([multi_class, tolerance, intercept_scaling, class_weight, C, max_iter_svm])
        return cs

class KNN_Reg(KNeighborsRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def space() -> CS:
        n_neighbors = UniformIntegerHyperparameter("n_neighbors", lower=2, upper=15, log=False)
        weights = CategoricalHyperparameter("weights", ['uniform', 'distance'], default_value='uniform')
        # weights = CategoricalHyperparameter("weights", ['uniform', 'distance'], default_value='uniform', weights=[0.3, 0.7])
        algorithm = CategoricalHyperparameter("algorithm", ['auto', 'ball_tree', 'kd_tree', 'brute'], default_value='auto')
        leaf_size = UniformIntegerHyperparameter("leaf_size", lower=10, upper=70, default_value=30, log=False)
        p = UniformIntegerHyperparameter("p", lower=1, upper=2, default_value=2)  # Manhattan or Euclidean

        cs = ConfigurationSpace()
        cs.add_hyperparameters([weights, n_neighbors, algorithm, leaf_size, p])
        # the leaf size is considered only if the algorithm is Ball Tree or KD Tree
        # ('Auto' can choose these or brute force, hence the not equals condition)
        cond_1 = NotEqualsCondition(leaf_size, algorithm, "brute")
        cs.add_conditions([cond_1])
        return cs

class SVM_Reg(LinearSVR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def space() -> CS:
        # kernel = CategoricalHyperparameter('kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], default_value='rbf')
        # gamma = CategoricalHyperparameter("gamma_svr", ['scale', 'auto'], default_value='auto')
        tolerance = UniformFloatHyperparameter("tol", lower=1e-4, upper=1e-3, default_value=1e-4)
        intercept_scaling = UniformFloatHyperparameter("intercept_scaling", lower=0.5, upper=2, default_value=1.0)
        max_iter_svm = UniformIntegerHyperparameter("max_iter_svm", 2000, 15000, default_value=5000)
        C = UniformFloatHyperparameter("C", lower=0.5, upper=2, default_value=1.0)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([tolerance, intercept_scaling, C, max_iter_svm])
        return cs

class Bayesian_Ridge_Reg(BayesianRidge):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def space() -> CS:
        n_iter = UniformIntegerHyperparameter("n_iter", 300, 1000, default_value=300)
        tolerance = UniformFloatHyperparameter("tol_brr", lower=1e-4, upper=1e-3, default_value=1e-4)
        alpha_1 = UniformFloatHyperparameter("alpha_1", lower=1e-6, upper=1e-3, default_value=1e-6)
        lambda_1 = UniformFloatHyperparameter("lambda_1", lower=1e-6, upper=1e-3, default_value=1e-6)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([n_iter, tolerance, alpha_1, lambda_1])
        return cs

class XGB_Reg(XGBRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def space() -> CS:
        n_estimators = UniformIntegerHyperparameter("n_estimators", lower=2, upper=20, default_value=10)
        max_depth = UniformIntegerHyperparameter("max_depth", lower=5, upper=20, default_value=5)
        subsample = UniformFloatHyperparameter("subsample", lower=0.1, upper=0.8, default_value=0.8)
        gamma = UniformFloatHyperparameter("gamma", lower=0.1, upper=1, default_value=0.1)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([n_estimators, subsample, max_depth, gamma])
        return cs

class MLP_Reg(MLPRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def space() -> CS:
        n_layer = UniformIntegerHyperparameter("n_layer", 1, 3, default_value=2)
        n_neurons = UniformIntegerHyperparameter("n_neurons", 10, 100, log=True, default_value=10)
        activation = CategoricalHyperparameter("activation", ['logistic', 'tanh', 'relu'], default_value='tanh')
        solver = CategoricalHyperparameter("solver", ['sgd', 'adam'], default_value='adam')
        alpha = UniformFloatHyperparameter("alpha", lower=0.05, upper=2, default_value=0.1, log=True)  # L2 regularization strength
        learning_rate = CategoricalHyperparameter("learning_rate", ['constant', 'invscaling', 'adaptive'], default_value='constant')
        learning_rate_init = UniformFloatHyperparameter("learning_rate_init", lower=1e-5, upper=1e-1, default_value=1e-3, log=True)
        max_iter = UniformIntegerHyperparameter("max_iter", 2000, 15000, default_value=5000)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([n_layer, n_neurons, activation, solver, alpha, learning_rate, learning_rate_init, max_iter])
        cond_1 = InCondition(learning_rate_init, solver, ["adam", "sgd"])
        cond_2 = EqualsCondition(learning_rate, solver, "sgd")
        cs.add_conditions([cond_1, cond_2])
        return cs
