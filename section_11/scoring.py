import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


class DataModel(object):
    data = []
    train = ()
    valid = ()
    test = ()

    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    y_train = pd.DataFrame()
    y_test = pd.DataFrame()

    _TRAINING_SIZE = 0.7
    _VALIDATION_SIZE = 0.15
    _TEST_SIZE = 0.15
    _X = pd.DataFrame()
    _y = pd.DataFrame()

    def __init__(self, X, y, train_size, valid_size, test_size):
        if train_size + valid_size + test_size != 1:
            raise ValueError(
                f"Combined split sizes must equal 1. Current split size={train_size + valid_size + test_size}")

        self._TRAINING_SIZE = train_size
        self._VALIDATION_SIZE = valid_size
        self._TEST_SIZE = test_size
        self._X = X
        self._y = y

    def train_test_validation_split(self, data):
        len_df = len(data)
        train_split = round(self._TRAINING_SIZE * len_df)
        valid_split = round(train_split + self._VALIDATION_SIZE * len_df)

        self.train = self._X[:train_split], self._y[:train_split]
        self.valid = self._X[train_split:valid_split], self._y[train_split:valid_split]
        self.test = self._X[valid_split:], self._y[valid_split:]

        return (self.train, self.valid, self.test)

    def split_train_test(self, test_size=None):
        if not test_size:
            test_size = self._TEST_SIZE

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self._X, self._y, test_size=test_size)


class Classifier(DataModel):
    def __init__(self, X, y, train_size=0.7, valid_size=0.15, test_size=0.15):
        super().__init__(X, y, train_size, valid_size, test_size)

    def evaluate_preds(self, y_true, y_preds):
        """
        Performs evaluation comparison on y_true labels vs. y_pred labels on a classification.
        """
        accuracy = accuracy_score(y_true, y_preds)
        precision = precision_score(y_true, y_preds)
        recall = recall_score(y_true, y_preds)
        f1 = f1_score(y_true, y_preds)
        return {"accuracy": round(accuracy, 2),
                "precision": round(precision, 2),
                "recall": round(recall, 2),
                "f1": round(f1, 2)}

    def cross_validate(self, clf, scoring):
        cv_score = cross_val_score(clf, self._X, self._y, cv=5,
                                   scoring=scoring)

        return np.mean(cv_score)

    def fit_and_score(self, models=None):
        """Fits and evaluates given machine learning models.

        Args:
            models (dict): different Scikit-Learn machine learning models
        """

        # set some default models to use
        if not models:
            models = {"Logistic Regression": LogisticRegression(),
                      "KNeighbors": KNeighborsClassifier(),
                      "Random Forest": RandomForestClassifier()}

        # Make a dictionary to keep model scores
        model_scores = {}

        # Loop through models
        for name, model in models.items():
            if self.X_train.empty or self.X_test.empty or self.y_train.empty or self.y_test.empty:
                self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test()

            # Fit the model to the data
            model.fit(self.X_train, self.y_train)
            # Evaluate the model and append its score to model_scores
            model_scores[name] = model.score(self.X_test, self.y_test)
        return model_scores

    def tune_hyperparameters(self, n=20) -> dict:
        print("Go grab some coffee...this is gonna take a couple minutes")

        tuned = {"Logistic Regression": self._tune_LogisticsRegression(n),
                 "KNeighbors": self._tune_KNeighborsClassifier(n),
                 "Random Forest": self._tune_RandomizedSearchCV(n),
                 "Grid Search": self._tune_GridSearchCV(n)}

        return {"Baseline": self.fit_and_score(), "Tuned": tuned}

    def _tune_GridSearchCV(self, n) -> dict:
        # Different hyperparameters for our LogisticRegression model
        log_reg_grid = {"C": np.logspace(-4, 4, n),
                        "solver": ["liblinear"]}

        # setup grid hyperparameter search for LogisticRegression
        gs_log_reg = GridSearchCV(LogisticRegression(),
                                  param_grid=log_reg_grid,
                                  cv=5)

        # Fitgrid hypeparamter search model
        gs_log_reg.fit(self.X_train, self.y_train)

        return {"best_params": gs_log_reg.best_params_,
                "score": gs_log_reg.score(self.X_test, self.y_test)}

    def _tune_RandomizedSearchCV(self, n) -> dict:
        # Create a hyperparameter grid for RandomForestClassifier
        rf_grid = {"n_estimators": np.arange(10, 1000, n),
                   "max_depth": [None, 3, 5, 10],
                   "min_samples_split": np.arange(2, 20, 2),
                   "min_samples_leaf": np.arange(1, 20, 3)}

        # Setup random hyperparameter search for RandomForestClassifier
        rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                                   param_distributions=rf_grid,
                                   cv=5,
                                   n_iter=20)

        # Fit random hyperparameter search model for RandomForestClassifier
        rs_rf.fit(self.X_train, self.y_train)

        return {"best_params": rs_rf.best_params_,
                "score": rs_rf.score(self.X_test, self.y_test)}

    def _tune_LogisticsRegression(self, n) -> dict:
        # Create a hyperparameter grid for LogisticRegression()
        log_reg_grid = {"C": np.logspace(-4, 4, n),
                        "solver": ["liblinear"]}

        # Setup random hyperparamter search for LogisticRegression
        rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                        param_distributions=log_reg_grid,
                                        cv=5,
                                        n_iter=20)

        # Fit random hyperparameter search model for LogisticRegression
        rs_log_reg.fit(self.X_train, self.y_train)

        return {"best_params": rs_log_reg.best_params_,
                "score": rs_log_reg.score(self.X_test, self.y_test)}

    def _tune_KNeighborsClassifier(self, n) -> dict:
        test_scores = []

        # Setup KNN instance
        knn = KNeighborsClassifier()
        n_neighbor_range = range(1, n+1)

        # Loop through different n_neighbors
        for i in n_neighbor_range:
            knn.set_params(n_neighbors=i)

            # Fit the algorithm
            knn.fit(self.X_train, self.y_train)

            # Update the test scores list
            test_scores.append(knn.score(self.X_test, self.y_test))

        max_value = max(test_scores)

        return {"n_neighbors": test_scores.index(max_value), "score": max_value}


class Regressor(DataModel):
    '''
    To be implemented
    '''
    pass
