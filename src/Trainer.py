from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, make_scorer
from sklearn.linear_model import LogisticRegression
from interpret.glassbox import ExplainableBoostingClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import sys
import shap
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, params, dataset):
        self.params = params
        self.dataset = dataset

        model_dict = {
            "LogReg": LogisticRegression(),
            "EBM": ExplainableBoostingClassifier(),
            "XGB": XGBClassifier()
        }
        self.models = {model_name: model_dict[model_name] for model_name in params['models'] if model_name in model_dict}
        self.grid_params = self.params["hpo"]
        self.cv_folds = params["cv_folds"]
        self.results = {}
        self.hpo_results = []
        self.feature_importances = {}

    def prepare_hpo(self, parameters, step_name):
        processed_params = {}
        for key, value in parameters.items():
            new_key = f"{step_name}__{key}"
            if key == "class_weight":
                processed_params[new_key] = [None if i == "None" else i for i in value]
            elif key == "max_depth":
                processed_params[new_key] = [None if i == -1 else i for i in value]
        return processed_params

    def train_model(self, model_name, model):
        if model_name in self.grid_params:
            print(f"Training {model_name}...")
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

            pipeline = Pipeline([
                ('smote', SMOTE(sampling_strategy=0.1, random_state=42)),
                ('under', RandomUnderSampler(sampling_strategy=0.5, random_state=42)),
                ('classifier', model)
            ])


            grid_search = GridSearchCV(pipeline, self.grid_params[model_name], cv=cv, scoring="roc_auc", n_jobs=-1)
            grid_search.fit(self.dataset.X_train, self.dataset.y_train)

            cv_results = cross_val_score(grid_search.best_estimator_, self.dataset.X_train, self.dataset.y_train, cv=cv, scoring="roc_auc")
            auroc_std = cv_results.std()

            model_memory = sys.getsizeof(grid_search.best_estimator_)

            self.hpo_results.append({
                "model": model_name,
                "best_params": grid_search.best_params_,
                "best_score": grid_search.best_score_,
            })

            self.shap_importance(grid_search.best_estimator_['classifier'], model_name)
            

            return grid_search.best_estimator_, auroc_std, model_memory

    def shap_importance(self, model, model_name):
        explainer = shap.Explainer(model, self.dataset.X_train)
        shap_values = explainer(self.dataset.X_train)
        shap_importance = np.abs(shap_values.values).mean(axis=0)
        self.feature_importances[model_name] = shap_importance

    def evaluate_model(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        return accuracy, roc_auc, recall, precision

    def train(self):
        for model_name, model in self.models.items():
            trained_model, auroc_std, model_memory = self.train_model(model_name, model)
            if trained_model:
                accuracy, roc_auc, recall, precision = self.evaluate_model(trained_model, self.dataset.X_test, self.dataset.y_test)
                self.results[model_name] = {
                "accuracy": accuracy,
                "roc_auc": roc_auc,
                "recall": recall,
                "precision": precision,
                "auroc_std": auroc_std,
                "model_memory": model_memory
            }

    def plot_feature_importances(self):
        for model_name, importances in self.feature_importances.items():
            importances = np.array(importances)
            sorted_indices = np.argsort(importances)[::-1]  # sort
            sorted_importances = importances[sorted_indices]
            sorted_features = self.dataset.X_train.columns[sorted_indices]
            
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(sorted_importances)), sorted_importances, align='center')
            plt.yticks(range(len(sorted_importances)), sorted_features)
            plt.xlabel("Feature Importance")
            plt.title(f"Feature Importance for {model_name}")
            plt.gca().invert_yaxis()  
            plt.show()


    def get_results(self):
        return self.results, self.hpo_results, self.feature_importances
