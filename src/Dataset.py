from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss



class Dataset():
    def __init__(self, params, data, random_state):
        self.params = params
        self.data=data
        self.random_state = random_state
        self.imputer_dict = {
            "mean": SimpleImputer(strategy="mean"),
            "knn": KNNImputer(n_neighbors=10),
            }
        self.data_prepared = False 


    def split(self):
        X = self.data[self.params["numerical_features"] + self.params["categorical_features"]]
        y = self.data[self.params["target"]]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, train_size=self.params["train_size"], random_state=self.random_state)
    
    def drop(self):
        for column in self.data.columns:
            print(column, self.data[column].isnull().sum())
            print(len(self.data) * 0.5)
        
        missing_threshold = len(self.data) * 0.5
        columns_to_drop = []
        for column in self.data.columns:
            if self.data[column].isnull().sum() > missing_threshold:
                columns_to_drop.append(column)
        self.data.drop(columns_to_drop, axis=1, inplace=True)

        self.data.dropna(subset=["Sex"], inplace=True)
        #self.data.dropna(subset=["Eth"], inplace=True)

    def encode(self):
        self.X_train["Sex"] = self.X_train["Sex"].map({"F": 1, "M": 0})
        self.X_test["Sex"] = self.X_test["Sex"].map({"F": 1, "M": 0})

    def impute(self):
        print(self.X_train.isnull().sum())
        self.imputation = self.imputer_dict[self.params["imputation"]]

        self.X_train[self.params["numerical_features"]] = self.imputation.fit_transform(self.X_train[self.params["numerical_features"]])        
        self.X_test[self.params["numerical_features"]] = self.imputation.transform(self.X_test[self.params["numerical_features"]])
        print(self.X_train.isnull().sum())

    def scale(self):    
        scaler = StandardScaler()
        self.X_train[self.params["numerical_features"]] = scaler.fit_transform(self.X_train[self.params["numerical_features"]])
        self.X_test[self.params["numerical_features"]] = scaler.transform(self.X_test[self.params["numerical_features"]])

    def sampling(self):
        if self.params["sampling"] == "smote":
            self.apply_smote()
        elif self.params["sampling"] == "nearmiss":
            self.apply_near_miss()

    def apply_smote(self):
        smote = SMOTE(random_state=self.random_state)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

    def apply_near_miss(self):
            near_miss = NearMiss()
            self.X_train, self.y_train = near_miss.fit_resample(self.X_train, self.y_train)

    def prepare_data(self):
        self.drop()
        self.split()
        self.encode()
        self.impute()
        self.scale()
        #self.sampling()
        self.data_prepared = True
        print(self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape)
        print(self.X_train.head())

    def get_prepared_data(self):
        if not self.data_prepared:
            raise ValueError("Data has not been prepared.")
        return self.X_train, self.X_test, self.y_train, self.y_test

        

