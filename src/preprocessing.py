import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix


def clean_ratings(df):
    df = df.dropna()
    df = df.drop_duplicates()
    df = df[df["rating"] > 0]
    df = df[df["user_id"] >= 0]
    df = df[df["item_id"] >= 0]
    df = df.reset_index(drop=True)
    return df


def map_ids(df):
    unique_users = sorted(df["user_id"].unique())
    unique_items = sorted(df["item_id"].unique())

    user_map = {uid: i for i, uid in enumerate(unique_users)}
    item_map = {iid: i for i, iid in enumerate(unique_items)}

    df["user_idx"] = df["user_id"].map(user_map)
    df["item_idx"] = df["item_id"].map(item_map)

    return df, user_map, item_map


def build_interaction_matrix(df, n_users, n_items):
    matrix = lil_matrix((n_users, n_items), dtype=np.float32)
    for _, row in df.iterrows():
        matrix[row["user_idx"], row["item_idx"]] = row["rating"]
    return matrix.tocsr()


def train_test_split(matrix, test_size=0.2, random_state=42):
    rng = np.random.RandomState(random_state)
    matrix = matrix.tocoo()

    n_total = matrix.nnz
    n_test = int(n_total * test_size)

    indices = rng.permutation(n_total)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    train = csr_matrix(
        (matrix.data[train_idx], (matrix.row[train_idx], matrix.col[train_idx])),
        shape=matrix.shape,
        dtype=np.float32,
    )
    test = csr_matrix(
        (matrix.data[test_idx], (matrix.row[test_idx], matrix.col[test_idx])),
        shape=matrix.shape,
        dtype=np.float32,
    )

    return train, test


def binarize(matrix, threshold=1.0):
    data = np.ones(matrix.nnz, dtype=np.float32)
    return csr_matrix((data, matrix.nonzero()), shape=matrix.shape, dtype=np.float32)
