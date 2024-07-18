import json
import numpy as np
from sklearn.model_selection import StratifiedKFold, ParameterGrid, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from interpret.glassbox import ExplainableBoostingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

class Trainer:
    def __init__(self, params, dataset):
        self.params = params
        self.X_train, self.y_train = dataset.get_prepared_data()[0], dataset.get_prepared_data()[2]
        self.X_test, self.y_test = dataset.get_prepared_data()[1], dataset.get_prepared_data()[3]

        model_dict = {
            "LogReg": LogisticRegression(),
            "EBM": ExplainableBoostingClassifier(),
            "XGB": XGBClassifier()
        }
        self.models = {model_name: model_dict[model_name] for model_name in params["models"] if model_name in model_dict}
        self.grid_params = self.params["hpo"]
        self.cv_folds = params["cv_folds"]
        self.results = []
        self.hpo_results = []

    def prepare_hpo(self, parameters):
        if "class_weight" in parameters:
            parameters["class_weight"] = [
                None if i == "None" else i for i in parameters["class_weight"]
            ]
        elif "max_depth" in parameters:
            parameters["max_depth"] = [
                None if i == -1 else i for i in parameters["max_depth"]
            ]
        return parameters

    def custom_cv(self, X, y, model, cv_folds, smote_params, under_params):
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in cv.split(X, y):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            smote = SMOTE(sampling_strategy=smote_params['sampling_strategy'], random_state=smote_params['random_state'])
            under = RandomUnderSampler(sampling_strategy=under_params['sampling_strategy'], random_state=under_params['random_state'])

            X_resampled, y_resampled = smote.fit_resample(X_train_fold, y_train_fold)
            X_resampled, y_resampled = under.fit_resample(X_resampled, y_resampled)

            model.fit(X_resampled, y_resampled)

            y_val_pred = model.predict(X_val_fold)
            roc_auc = roc_auc_score(y_val_fold, y_val_pred)
            scores.append(roc_auc)

        return np.mean(scores), np.std(scores)

    def train_model(self, model_name, model, X_train, y_train):
        if model_name in self.grid_params:
            print(f"Training {model_name}...")
            processed_hpo_grid = self.prepare_hpo(self.grid_params[model_name])
            best_model = None
            best_score = -np.inf
            best_params = None
            mean_test_score, std_test_score = None, None

            for params in ParameterGrid(processed_hpo_grid):
                try:
                    model.set_params(**params)
                    mean_score, std_score = self.custom_cv(X_train, y_train, model, self.cv_folds, 
                                                          smote_params={'sampling_strategy': 0.1, 'random_state': 42},
                                                          under_params={'sampling_strategy': 0.5, 'random_state': 42})

                    if mean_score > best_score:
                        best_score = mean_score
                        best_model = clone(model)
                        best_params = params
                        mean_test_score, std_test_score = mean_score, std_score
                except (ValueError, NotFittedError) as e:
                    print(f"Skip {params}, error: {e}")
                    continue  
                except Exception as e:
                    print(f"unexpected error: {e}")
                    continue

            if best_model is None:
                raise ValueError(f"no valid params for {model_name}")

            best_model.fit(X_train, y_train)

            self.hpo_results.append({
                "model": model_name,
                "best_params": best_params,
                "best_score": best_score
            })

            cv_metrics = {
                "roc_auc_mean": mean_test_score,
                "roc_auc_std": std_test_score
            }

        return best_model, cv_metrics

    def evaluate_model(self, model, X_test, y_test):
        try:
            predictions = model.predict(X_test)
        except NotFittedError:
            raise ValueError(f"model {model} not fitted")

        accuracy = accuracy_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        return accuracy, roc_auc, recall, precision

    def train(self):
        for model_name, model in self.models.items():
            try:
                trained_model, cv_metrics = self.train_model(model_name, model, self.X_train, self.y_train)
                accuracy, roc_auc, recall, precision = self.evaluate_model(trained_model, self.X_test, self.y_test)
                self.results.append({
                    "model": model_name,
                    "accuracy": accuracy,
                    "roc_auc": roc_auc,
                    "recall": recall,
                    "precision": precision,
                    "cv_metrics": cv_metrics
                })
            except ValueError as e:
                print(f"Error training {model_name}: {e}")
                continue
            except Exception as e:
                print(f"An unexpected error occurred during training of {model_name}: {e}")
                continue

    def get_results(self):
        return self.results, self.hpo_results
