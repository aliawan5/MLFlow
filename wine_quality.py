import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier

class Model:
    def __init__(self, model):
        self.model = model
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_model(self, file_path:str):
        try:
            self.data = pd.read_csv(file_path)
            print("Model loaded successfully")
        except FileNotFoundError:
            raise ("File not found")
        
    def preprocess_data(self):
        sc = StandardScaler()
        if self.data is not None:
            self.data.dropna(inplace=True)
            self.X = self.data.drop('quality', axis=1)
            if self.X is not None:
                self.X = sc.fit_transform(self.X)
            self.y = self.data['quality']
            print("Data preprocessed successfully")
        else:
            raise ("No data loaded")
        
    def split_data(self, test_size=0.2):
        if self.X is not None and self.y is not None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
            print("Data split successfully")
        else:
            raise ("No data loaded or preprocessed")
        
    def train_model(self):
        if self.X_train is not None and self.y_train is not None:
            self.model.fit(self.X_train, self.y_train)
            print("Model trained successfully")
        else:
            raise ("No data split for training")
        
    def evaluate_model(self):
        if self.X_test is not None and self.y_test is not None:
            y_pred = self.model.predict(self.X_test)

            self.mean_squared = mean_squared_error(self.y_test, y_pred)
            print(f"Mean Squared Error: {self.mean_squared}")

            self.mean_absolute = mean_absolute_error(self.y_test, y_pred)
            print(f"Mean Absolute Error: {self.mean_absolute}")

        else:
            raise ("No data split for evaluation")

    def mlflow(self):
        mlflow.set_tracking_uri("http://127.0.0.1:5000/")
        mlflow.set_experiment("Wine Quality Prediction")

        with mlflow.start_run():

            mlflow.log_params(self.model.get_params())

            mlflow.log_metric("mean_squared_error", self.mean_squared)
            mlflow.log_metric("mean_absolute_error", self.mean_absolute)

            mlflow.log_artifact("winequality-red.csv", artifact_path='data')

            mlflow.sklearn.log_model(self.model, 'model')
        
if __name__ == "__main__":
    model = RandomForestClassifier()
    m = Model(model)
    m.load_model("winequality-red.csv")
    m.preprocess_data()
    m.split_data()
    m.train_model()
    m.evaluate_model()
    m.mlflow()
    