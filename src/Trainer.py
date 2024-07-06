import json
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from interpret.glassbox import ExplainableBoostingClassifier
from xgboost import XGBClassifier


class Trainer:
    def __init__(self, params, dataset):
        self.params = params
        self.X_train, self.X_test, self.y_train, self.y_test = dataset.get_prepared_data()

        self.models = {
            "LogReg": LogisticRegression(),
            "EBM": ExplainableBoostingClassifier(),
            "XGB": XGBClassifier()
        }
        self.grid_params = self.params["hpo"]
        self.cv_folds = params["cv_folds"]
        self.results = []
        self.hpo_results = []

    def prepare_hpo(self, parameters):
        if "class_weight" in parameters:
            parameters["class_weight"] = [
                None if i == "None" else i for i in parameters["class_weight"]
            ]
        return parameters

    def train_model(self, model_name, model, X_train, y_train):
        if model_name in self.grid_params:
            print(f"Training {model_name}...")
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            processed_hpo_grid = self.prepare_hpo(self.grid_params[model_name])
            grid_search = GridSearchCV(model, processed_hpo_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
            grid_search.fit(X_train, y_train)
            self.hpo_results.append({
                "model": model_name,
                "best_params": grid_search.best_params_,
                "best_score": grid_search.best_score_
                })
            return grid_search.best_estimator_
        
    def evaluate_model(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, predictions)
        return accuracy, roc_auc

    def train(self):
        for model_name, model in self.models.items():
            trained_model = self.train_model(model_name, model, self.X_train, self.y_train)
            if trained_model:
                accuracy, roc_auc = self.evaluate_model(trained_model, self.X_test, self.y_test)
                predictions = trained_model.predict(self.X_test)
                recall = recall_score(self.y_test, predictions)
                precision = precision_score(self.y_test, predictions)
                self.results.append({
                    "model": model_name,
                    "accuracy": accuracy,
                    "roc_auc": roc_auc,
                    "recall": recall,
                    "precision": precision
                })

    def get_results(self):
        return self.results, self.hpo_results

