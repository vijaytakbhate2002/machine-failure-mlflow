import pandas as pd
import mlflow
import config
from sklearn.base import ClassifierMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mlflow.sklearn import log_model
import argparse

# ingesting data
data = pd.read_csv(config.DATA)

# model training
def train_model(algo:ClassifierMixin, X_train, y_train, params):
    model = algo.set_params(params)
    model = model.fit(X_train, y_train)
    return model

# evaluation of model
def eval(model, X_train, X_test, y_train, y_test):
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    test_score = accuracy_score(y_test, y_test_pred)
    train_score = accuracy_score(y_train, y_train_pred)
    return {'mae':mae, 'mse':mse, 'r2':r2, 'test_score':test_score, 'train_score':train_score}

# mlflow run 
def run_mlflow(params):
    mlflow.set_experiment("Machine Failure")
    params = vars(params)
    with mlflow.start_run(run_name="Logistic Regrssion") as f:
        y = data[config.TARGET]
        X = data.drop(config.TARGET, axis='columns')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y)
        model = LogisticRegression(**params)
        model = model.fit(X_train, y_train)

        # parameter logging
        mlflow.log_params(params)

        # tag logging
        mlflow.set_tag("VERSION", '1.0.0')

        # matrics logging
        mlflow.log_metrics(eval(model, X_train, X_test, y_train, y_test))

        # model logging
        log_model(model, config.TRAINED_MODEL)

        # mlflow run end
        mlflow.end_run()

if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--penalty', '-p', type=str, default='l2')
    arg.add_argument('--C', '-c', type=float, default=0.2)
    arg.add_argument('--max_iter', '-mi', type=int, default=100)
    parsed_args = arg.parse_args()
    print(parsed_args)
    run_mlflow(parsed_args)







