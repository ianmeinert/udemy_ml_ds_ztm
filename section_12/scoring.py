import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


@staticmethod
def fill_missing_data(data):
    for label, content in data.items():
        # Check for which numeric columns have null values
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                # Add a binary column which tells us if the data was missing or not
                data[label+"_is_missing"] = pd.isnull(content)
                # Fill missing numeric values with median
                data[label] = content.fillna(content.median())
            # Check for which categorial columns have null values
        elif not pd.api.types.is_numeric_dtype(content):
            # Add binary column to indicate whether sample had missing value
            data[label+"_is_missing"] = pd.isnull(content)
            # Turn categories into numbers and add +1 (for missing values == -1)
            data[label] = pd.Categorical(content).codes + 1


@staticmethod
def string_cols_to_category(data):
    # This will turn all of the string values into category values
    for label, content in data.items():
        if pd.api.types.is_string_dtype(content):
            data[label] = content.astype("category").cat.as_ordered()


class DataModel(object):
    train = ()
    valid = ()
    test = ()

    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    X_valid = pd.DataFrame()
    y_train = pd.DataFrame()
    y_test = pd.DataFrame()
    y_valid = pd.DataFrame()

    _TRAINING_SIZE = 0.7
    _VALIDATION_SIZE = 0.15
    _TEST_SIZE = 0.15
    _X = pd.DataFrame()
    _y = pd.DataFrame()

    def __init__(self, X, y, train_size=0.7, valid_size=0.15, test_size=0.15):
        if train_size + valid_size + test_size != 1:
            raise ValueError(
                f"Combined split sizes must equal 1. Current split size={train_size + valid_size + test_size}")

        self._TRAINING_SIZE = train_size
        self._VALIDATION_SIZE = valid_size
        self._TEST_SIZE = test_size
        self._X = X
        self._y = y

    def split_train_test_validation(self, data):
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

    def split_train_valid(self, column):
        # Split training and validation data into
        self.X_train, self.y_train = self._X.drop(
            column, axis=1), self._X[column]
        self.X_valid, self.y_valid = self._y.drop(
            column, axis=1), self._y[column]


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
            # Fit the model to the data
            model.fit(self.X_train, self.y_train)
            # Evaluate the model and append its score to model_scores
            model_scores[name] = model.score(self.X_test, self.y_test)
        return model_scores


class Regressor(DataModel):
    def __init__(self, X, y, train_size=0.7, valid_size=0.15, test_size=0.15):
        super().__init__(X, y, train_size, valid_size, test_size)

    def show_scores(self, model):
        train_preds = model.predict(self.X_train)
        val_preds = model.predict(self.X_valid)

        scores = {"Training MAE": mean_absolute_error(self.y_train, train_preds),
                  "Valid MAE": mean_absolute_error(self.y_valid, val_preds),
                  "Training RMSLE": self.rmsle(self.y_train, train_preds),
                  "Valid RMSLE": self.rmsle(self.y_valid, val_preds),
                  "Training R^2": r2_score(self.y_train, train_preds),
                  "Valid R^2": r2_score(self.y_valid, val_preds)}
        return scores

    def rmsle(self, y_true, y_preds):
        """
        Calculates root mean squared log error between predictions and true labels.
        """
        return np.sqrt(mean_squared_log_error(y_true, y_preds))
