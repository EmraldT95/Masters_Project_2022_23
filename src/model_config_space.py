from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter, OrdinalHyperparameter
from ConfigSpace.conditions import NotEqualsCondition, AndConjunction, EqualsCondition
from ConfigSpace.conditions import InCondition, GreaterThanCondition


class KNN(KNeighborsClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def space() -> CS:
        n_neighbors = UniformIntegerHyperparameter("n_neighbors", lower=2, upper=15, log=False)
        weights = CategoricalHyperparameter("weights", ['uniform', 'distance'], default_value='uniform', weights=[0.3, 0.7])
        algorithm = CategoricalHyperparameter("algorithm", ['auto', 'ball_tree', 'kd_tree', 'brute'], default_value='auto')
        leaf_size = UniformIntegerHyperparameter("leaf_size", lower=10, upper=70, default_value=30, log=False)
        p = UniformIntegerHyperparameter("p", lower=1, upper=2, default_value=2)  # Manhattan or Euclidean
        cs = ConfigurationSpace()
        cs.add_hyperparameters([n_neighbors, weights, algorithm, leaf_size, p])

        # the leaf size is considered only if the algorithm is Ball Tree or KD Tree
        # ('Auto' can choose these or brute force, hence the not equals condition)
        cond_1 = NotEqualsCondition(leaf_size, algorithm, "brute")
        cs.add_conditions([cond_1])

        return cs


class RF(RandomForestClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def space() -> CS:
        criterion = CategoricalHyperparameter("criterion", ["gini", "entropy", "log_loss"], default_value="gini")
        n_estimators = UniformIntegerHyperparameter("n_estimators", lower=50, upper=150, default_value=100)
        max_features = CategoricalHyperparameter("max_features", ["sqrt", "log2", 'None'], default_value="sqrt")
        bootstrap = CategoricalHyperparameter("bootstrap", [True, False], default_value=True)
        class_weight_rf = CategoricalHyperparameter("class_weight_rf", ['balanced', 'None'], default_value='None')
        cs = ConfigurationSpace()
        cs.add_hyperparameters([criterion, n_estimators, max_features, bootstrap, class_weight_rf])
        return cs

class MLP(MLPClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def space() -> CS:
        n_layer = UniformIntegerHyperparameter("n_layer", 1, 4, default_value=2)
        n_neurons = UniformIntegerHyperparameter("n_neurons", 4, 256, log=True, default_value=10)
        activation = CategoricalHyperparameter("activation", ['logistic', 'tanh', 'relu'], default_value='tanh')
        solver = CategoricalHyperparameter("solver", ['sgd', 'adam'], default_value='adam')
        alpha = UniformFloatHyperparameter("alpha", lower=0.1, upper=2, default_value=0.1, log=True)  # L2 regularization strength
        learning_rate = CategoricalHyperparameter("learning_rate", ['constant', 'invscaling', 'adaptive'], default_value='constant')
        learning_rate_init = UniformFloatHyperparameter("learning_rate_init", lower=1e-5, upper=1e-1, default_value=1e-3, log=True)
        warm_start = CategoricalHyperparameter("warm_start", [True, False], default_value=False)
        momentum = UniformFloatHyperparameter("momentum", lower=0., upper=0.99, default_value=0.9)
        early_stopping = CategoricalHyperparameter("early_stopping", [True, False], default_value=False)
        beta_1 = UniformFloatHyperparameter("beta_1", lower=0., upper=.99, default_value=0.9)
        beta_2 = UniformFloatHyperparameter("beta_2", lower=0., upper=.999, default_value=0.999)
        epsilon = UniformFloatHyperparameter("epsilon", lower=1e-8, upper=1e-4, default_value=1e-8)
        n_iter_no_change = UniformIntegerHyperparameter("n_iter_no_change", lower=10, upper=20, default_value=10)
        max_iter_mlp = OrdinalHyperparameter("max_iter_mlp", sequence=[2000, 5000, 10000, 15000, 20000, 25000])

        cs = ConfigurationSpace()
        cs.add_hyperparameters([n_layer, n_neurons, activation, solver, alpha, learning_rate, learning_rate_init,
                                warm_start, momentum, early_stopping, beta_1, beta_2, epsilon, n_iter_no_change, max_iter_mlp])

        ############################
        # Conditional hyperparamters
        ############################

        cond_1 = InCondition(learning_rate_init, solver, ["adam", "sgd"])
        cond_3 = InCondition(early_stopping, solver, ["adam", "sgd"])
        cond_4 = InCondition(n_iter_no_change, solver, ["adam", "sgd"])
        cond_6 = EqualsCondition(learning_rate, solver, "sgd")
        cond_7 = EqualsCondition(momentum, solver, "sgd")
        cond_9 = EqualsCondition(beta_1, solver, "adam")
        cond_10 = EqualsCondition(beta_2, solver, "adam")
        cond_11 = EqualsCondition(epsilon, solver, "adam")

        cs.add_conditions([cond_1, cond_3, cond_4, cond_6, cond_7, cond_9, cond_10, cond_11])

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


class SVM(LinearSVC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def space() -> CS:
        multi_class = CategoricalHyperparameter("multi_class", ['ovr', 'crammer_singer'], default_value='ovr')
        tolerance = UniformFloatHyperparameter("tolerance", lower=1e-4, upper=1e-3, default_value=1e-4)
        intercept_scaling = UniformFloatHyperparameter("intercept_scaling", lower=0.5, upper=2, default_value=1.0)
        class_weight = CategoricalHyperparameter("class_weight", ['balanced', 'None'], default_value='None')
        max_iter_svm = OrdinalHyperparameter("max_iter_svm", sequence=[2000, 5000, 10000, 15000, 20000, 25000])
        C = UniformFloatHyperparameter("C", lower=0.5, upper=2, default_value=1.0)
        cs = ConfigurationSpace()
        cs.add_hyperparameters([multi_class, tolerance, intercept_scaling, class_weight, C, max_iter_svm])
        return cs
