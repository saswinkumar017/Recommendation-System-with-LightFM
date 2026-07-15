import os
import zipfile
import urllib.request

import pandas as pd

from src.utils import CONFIG


def download_movielens():
    url = CONFIG["dataset_url"]
    data_dir = CONFIG["data_dir"]
    zip_path = os.path.join(data_dir, "ml-100k.zip")
    extract_path = os.path.join(data_dir, "ml-100k")

    if os.path.exists(extract_path):
        print(f"Dataset already exists at {extract_path}")
        return extract_path

    print(f"Downloading MovieLens 100K from {url}...")
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete. Extracting...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    os.remove(zip_path)
    print(f"Dataset extracted to {extract_path}")
    return extract_path


def load_ratings(data_path):
    filepath = os.path.join(data_path, "u.data")
    columns = ["user_id", "item_id", "rating", "timestamp"]
    df = pd.read_csv(filepath, sep="\t", names=columns)
    return df


def load_items(data_path):
    filepath = os.path.join(data_path, "u.item")
    columns = [
        "item_id", "title", "release_date", "video_release_date",
        "imdb_url", "unknown", "action", "adventure", "animation",
        "childrens", "comedy", "crime", "documentary", "drama",
        "fantasy", "film_noir", "horror", "musical", "mystery",
        "romance", "sci_fi", "thriller", "war", "western",
    ]
    df = pd.read_csv(filepath, sep="|", encoding="latin-1", names=columns)
    df = df[["item_id", "title", "release_date"]]
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    return df


def load_users(data_path):
    filepath = os.path.join(data_path, "u.user")
    columns = ["user_id", "age", "gender", "occupation", "zip_code"]
    df = pd.read_csv(filepath, sep="|", names=columns)
    return df


def load_all():
    data_path = download_movielens()
    ratings = load_ratings(data_path)
    items = load_items(data_path)
    users = load_users(data_path)
    return ratings, items, users
