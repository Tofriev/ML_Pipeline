from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler

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
    
    def drop_missing(self):
        self.data.dropna(subset=["Sex"], inplace=True)
        self.data.dropna(subset=["Eth"], inplace=True)

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

    def prepare_data(self):
        self.drop_missing()
        self.split()
        self.encode()
        self.impute()
        self.scale()
        self.data_prepared = True
        print(self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape)
        print(self.X_train.head())

    def get_prepared_data(self):
        if not self.data_prepared:
            raise ValueError("Data has not been prepared.")
        return self.X_train, self.X_test, self.y_train, self.y_test

        

