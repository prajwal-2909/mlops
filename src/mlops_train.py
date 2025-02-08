
import os
import yaml
import pandas as pd
import numpy as np
import argparse
from pkgutil import get_data
from get_data import get_data, read_param
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib
import json
import mlflow
from urllib.parse import urlparse


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_and_evaluate(config_path):
    config = read_param(config_path)
    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    raw_data_path = config["load_data"]["clean_data"]
    split_data = config["split_data"]["test_size"]
    random_state = config["base"]["random_state"]
    df = pd.read_csv(raw_data_path, sep=",")
    model_dir = config["model_path"]

    n_estimators = config["estimators"]["RandomForest"]["params"]["n_estimators"]
    max_depth = config["estimators"]["RandomForest"]["params"]["max_depth"]

    target = config["base"]["target_col"]
    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    ###########################

    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    ############################
    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_runs:    
    
        lr = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state= random_state)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        print("RandomForest model (n_estimators=%f, max_depth=%f):" % (n_estimators, max_depth))

        score_files = config["reports"]["score"]
        params_file = config["reports"]["params"]

        with open(score_files, "w") as f:
            scores = {
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }
            json.dump(scores, f)
        
        with open(params_file, "w") as f:
            params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth
            }
            json.dump(params, f)

        tracking_uri_type = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_uri_type != "file":
            mlflow.sklearn.log_model(lr, "models", registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.sklearn.log_model(lr, "models")

        # model_path = config["model_path"]
        # joblib.dump(lr, model_path)
        model_dir = os.path.dirname(config["model_path"])
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "models.joblib")
        joblib.dump(lr, model_path)


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yml")
    parsed_args= args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
