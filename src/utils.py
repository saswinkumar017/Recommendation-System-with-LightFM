import os
import json
import pickle
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    "dataset": "ml-100k",
    "dataset_url": "http://files.grouplens.org/datasets/movielens/ml-100k.zip",
    "data_dir": os.path.join(PROJECT_ROOT, "dataset"),
    "models_dir": os.path.join(PROJECT_ROOT, "models"),
    "outputs_dir": os.path.join(PROJECT_ROOT, "outputs"),
    "graphs_dir": os.path.join(PROJECT_ROOT, "outputs", "graphs"),
    "recommendations_dir": os.path.join(PROJECT_ROOT, "outputs", "recommendations"),
    "trained_model_dir": os.path.join(PROJECT_ROOT, "outputs", "trained_model"),
    "model_params": {
        "no_components": 30,
        "learning_rate": 0.05,
        "max_sampled": 10,
        "random_state": 42,
    },
    "train_params": {
        "epochs": 30,
        "num_threads": 1,
    },
    "eval_params": {
        "k": 10,
        "test_size": 0.2,
    },
    "recommendation_params": {
        "top_k": 10,
    },
}


def ensure_dirs():
    dirs = [
        CONFIG["data_dir"],
        CONFIG["models_dir"],
        CONFIG["outputs_dir"],
        CONFIG["graphs_dir"],
        CONFIG["recommendations_dir"],
        CONFIG["trained_model_dir"],
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def save_json(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    with open(filepath) as f:
        return json.load(f)


def save_pickle(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)
