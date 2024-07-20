from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler



class Dataset():
    def __init__(self, params, data, random_state, frac_ids=None):
        self.params = params
        self.data=data
        self.random_state = random_state
        self.imputer_dict = {
            "mean": SimpleImputer(strategy="mean"),
            "knn": KNNImputer(n_neighbors=5),
            }
        self.data_prepared = False
        self.data["id"] = range(len(self.data))
        if frac_ids:
            self.data = self.data[self.data["id"].isin(frac_ids)]

    def split(self):
        X = self.data[self.params["numerical_features"] + self.params["categorical_features"]]
        y = self.data[self.params["target"]]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, train_size=self.params["train_size"], random_state=self.random_state, stratify=y)
    
    def drop(self):
        for column in self.data.columns:
            pass
            #print(column, self.data[column].isnull().sum())
            #print(len(self.data) * 0.5)
        
        missing_threshold = len(self.data) * 0.5
        columns_to_drop = []
        
        for column in self.data.columns:
            if self.data[column].isnull().sum() > missing_threshold:
                columns_to_drop.append(column)
        self.data.drop(columns_to_drop, axis=1, inplace=True)
        #print(columns_to_drop)
        self.data.dropna(subset=["gender"], inplace=True)
        self.data.dropna(subset=["Eth"], inplace=True)
        self.params["numerical_features"] = [feature for feature in self.params["numerical_features"] if feature not in columns_to_drop]

    def encode(self):
        self.X_train["gender"] = self.X_train["gender"].map({"F": 1, "M": 0}).astype(float)
        self.X_test["gender"] = self.X_test["gender"].map({"F": 1, "M": 0}).astype(float)

        self.X_train = pd.get_dummies(self.X_train, columns=["Eth"], drop_first=True, dtype=float)
        self.X_test = pd.get_dummies(self.X_test, columns=["Eth"], drop_first=True, dtype=float)

    def impute(self):
        print("stating imputation...")
        print(self.X_train.isnull().sum())
        self.imputation = self.imputer_dict[self.params["imputation"]]

        self.X_train[self.params["numerical_features"]] = self.imputation.fit_transform(self.X_train[self.params["numerical_features"]])        
        self.X_test[self.params["numerical_features"]] = self.imputation.transform(self.X_test[self.params["numerical_features"]])
        print(self.X_train.isnull().sum())
        print("imputation done")

    def scale(self):    
        scaler = StandardScaler()
        self.X_train[self.params["numerical_features"]] = scaler.fit_transform(self.X_train[self.params["numerical_features"]])
        self.X_test[self.params["numerical_features"]] = scaler.transform(self.X_test[self.params["numerical_features"]])

    def sampling(self):
        over = SMOTE(sampling_strategy=0.1, random_state=self.random_state)
        under = RandomUnderSampler(sampling_strategy=0.5, random_state=self.random_state)

        print("Before Over Sampling:", self.X_train.shape, self.y_train.shape)
        X_over, y_over = over.fit_resample(self.X_train, self.y_train)
        print("After Over Sampling:", X_over.shape, y_over.shape)
    
        X_resampled, y_resampled = under.fit_resample(X_over, y_over)
        print("After Under Sampling:", X_resampled.shape, y_resampled.shape)
    
        self.X_train, self.y_train = X_resampled, y_resampled

    
    def prepare_data(self):
        print(f"Initial N of Features: {len(self.data.columns)-5}, N of Samples: {len(self.data)}")
        self.drop()
        self.split()
        self.encode()   
        self.impute()
        self.scale()
        #self.sampling()  
        print(f"After prepro N of Features: {len(self.data.columns)}, N of Samples: {len(self.data)}")
        self.data_prepared = True
        print(self.X_train.head())

    def make_fractional_ids(self, step=1000):
        self.drop()
        id_dict = {} # keys: fraction size {values: list of indices}


        # prop of positive 
        p_absolute = self.data[self.params["target"]].sum()
        num_samples = len(self.data)
        percentage_positive = p_absolute / num_samples
        
        prev_sampled_ids = []


        for i in range(1, num_samples // step):
            data = self.data.copy()
            data = data[~data["id"].isin(prev_sampled_ids)]

            #ensure prop of positives stays the same 
            pos_num_size = int(step * percentage_positive)
            neg_num_size = step - pos_num_size

            positives = data[data[self.params["target"]] == 1]
            negatives = data[data[self.params["target"]] == 0]
            
            pos_sample_indices = positives.sample(n=pos_num_size, random_state=self.random_state)["id"].tolist()
            neg_sample_indices = negatives.sample(n=neg_num_size, random_state=self.random_state)["id"].tolist()
            
            sample_ids = pos_sample_indices + neg_sample_indices
            prev_sampled_ids += sample_ids
            #print(prev_sampled_ids)
            id_dict[len(prev_sampled_ids)] = prev_sampled_ids.copy()

        return id_dict

    def get_prepared_data(self):
        if not self.data_prepared:
            raise ValueError("Data has not been prepared.")
        return self.X_train, self.X_test, self.y_train, self.y_test

        

