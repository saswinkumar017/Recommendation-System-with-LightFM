import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.utils import CONFIG, ensure_dirs, save_json, save_pickle
from src.data_loader import load_all
from src.preprocessing import clean_ratings, map_ids, build_interaction_matrix, \
    train_test_split, binarize
from src.train import LightFM
from src.recommend import recommend_for_user, format_recommendations, \
    save_recommendations
from src.evaluate import evaluate_model, compare_losses, compare_epochs
from src.visualization import generate_all_plots


def step1_load_dataset():
    print("\n" + "="*60)
    print("STEP 1: Loading Dataset")
    print("="*60)
    ratings, items, users = load_all()
    print(f"\nRatings: {ratings.shape[0]} records")
    print(f"Users: {users.shape[0]} unique")
    print(f"Items: {items.shape[0]} unique")
    return ratings, items, users


def step2_explore_dataset(ratings, items, users):
    print("\n" + "="*60)
    print("STEP 2: Exploratory Data Analysis")
    print("="*60)

    n_users = ratings["user_id"].nunique()
    n_items = ratings["item_id"].nunique()
    n_interactions = len(ratings)
    total_possible = n_users * n_items
    sparsity = (1 - n_interactions / total_possible) * 100

    print(f"\nNumber of Users: {n_users}")
    print(f"Number of Items:  {n_items}")
    print(f"Interactions:     {n_interactions}")
    print(f"Matrix Sparsity:  {sparsity:.2f}%")
    print(f"Density:          {100 - sparsity:.4f}%")
    print(f"\nRating Statistics:")
    print(ratings["rating"].describe())
    print(f"\nSample Ratings:")
    print(ratings.head(10))

    if items is not None:
        print(f"\nSample Items:")
        print(items.head(5))

    if users is not None:
        print(f"\nSample Users:")
        print(users.head(5))

    return n_users, n_items, sparsity


def step3_clean_data(ratings):
    print("\n" + "="*60)
    print("STEP 3: Data Cleaning")
    print("="*60)
    df = clean_ratings(ratings)
    print(f"Records after cleaning: {len(df)}")
    print(f"Removed {len(ratings) - len(df)} records")
    return df


def step4_prepare_data(df, n_users, n_items):
    print("\n" + "="*60)
    print("STEP 4: Data Preparation")
    print("="*60)

    df, user_map, item_map = map_ids(df)
    print(f"User ID mapping: {len(user_map)} users")
    print(f"Item ID mapping: {len(item_map)} items")

    interaction_matrix = build_interaction_matrix(df, n_users, n_items)
    print(f"Interaction matrix shape: {interaction_matrix.shape}")
    print(f"Non-zero entries: {interaction_matrix.nnz}")

    train, test = train_test_split(
        interaction_matrix,
        test_size=CONFIG["eval_params"]["test_size"],
        random_state=CONFIG["model_params"]["random_state"],
    )
    print(f"Train interactions: {train.nnz}")
    print(f"Test interactions:  {test.nnz}")

    train_binary = binarize(train)
    test_binary = binarize(test)

    user_reverse_map = {v: k for k, v in user_map.items()}
    item_reverse_map = {v: k for k, v in item_map.items()}

    mapping_info = {
        "user_map": user_map,
        "item_map": item_map,
        "user_rev": user_reverse_map,
        "item_rev": item_reverse_map,
    }

    return train_binary, test_binary, mapping_info


def step5_train_model(train_interactions, loss="warp", epochs=30):
    print("\n" + "="*60)
    print(f"STEP 5-7: Training LightFM Model (loss={loss})")
    print("="*60)

    params = CONFIG["model_params"]
    model = LightFM(
        no_components=params["no_components"],
        learning_rate=params["learning_rate"],
        max_sampled=params["max_sampled"],
        random_state=params["random_state"],
        loss=loss,
    )
    model.fit(train_interactions, epochs=epochs, verbose=True)
    return model


def step6_recommend(model, train_interactions, items_df, mapping_info, top_k=10):
    print("\n" + "="*60)
    print("STEP 8: Generating Recommendations")
    print("="*60)

    item_ids = mapping_info["item_rev"]
    item_titles = dict(zip(items_df["item_id"], items_df["title"]))

    all_recs = {}
    sample_users = list(mapping_info["user_rev"].keys())[:5]

    for user_idx in sample_users:
        user_id = mapping_info["user_rev"][user_idx]
        recs = recommend_for_user(model, user_idx, item_ids, item_titles, top_k)
        all_recs[int(user_id)] = recs

    print(format_recommendations(all_recs))

    rec_path = os.path.join(CONFIG["recommendations_dir"], "top_recommendations.csv")
    df_recs = save_recommendations(all_recs, rec_path)
    print(f"\nRecommendations saved to: {rec_path}")

    return all_recs


def step7_evaluate(model, train_interactions, test_interactions):
    print("\n" + "="*60)
    print("STEP 9: Evaluation")
    print("="*60)

    k = CONFIG["eval_params"]["k"]
    metrics = evaluate_model(model, test_interactions, train_interactions, k)
    print(f"\nEvaluation Results (K={k}):")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric:15} : {value:.4f}")

    save_json(metrics, os.path.join(CONFIG["outputs_dir"], "evaluation_metrics.json"))
    return metrics


def step8_compare_losses_and_epochs(train_interactions, test_interactions):
    print("\n" + "="*60)
    print("Comparing Loss Functions")
    print("="*60)

    loss_results = compare_losses(
        train_interactions, test_interactions,
        epochs=CONFIG["train_params"]["epochs"],
    )
    print("\nLoss Comparison Results:")
    print(loss_results.to_string(index=False))

    loss_path = os.path.join(CONFIG["outputs_dir"], "loss_comparison.csv")
    loss_results.to_csv(loss_path, index=False)
    print(f"Saved to: {loss_path}")

    print("\n" + "="*60)
    print("Comparing Training Epochs (WARP loss)")
    print("="*60)

    epoch_results = compare_epochs(
        train_interactions, test_interactions,
        loss="warp", max_epochs=CONFIG["train_params"]["epochs"],
    )

    epoch_path = os.path.join(CONFIG["outputs_dir"], "epoch_comparison.csv")
    epoch_results.to_csv(epoch_path, index=False)
    print(f"Saved to: {epoch_path}")

    return loss_results, epoch_results


def step9_save_model(model, mapping_info):
    print("\n" + "="*60)
    print("STEP 11: Saving Model")
    print("="*60)

    model_dir = CONFIG["trained_model_dir"]
    save_pickle(model, os.path.join(model_dir, "lightfm_model.pkl"))
    save_pickle(mapping_info, os.path.join(model_dir, "mapping_info.pkl"))

    model_json = {
        "no_components": model.no_components,
        "learning_rate": model.learning_rate,
        "max_sampled": model.max_sampled,
        "random_state": model.random_state,
        "loss": model.loss,
        "n_users": int(model.user_embeddings.shape[0]),
        "n_items": int(model.item_embeddings.shape[0]),
    }
    save_json(model_json, os.path.join(model_dir, "model_config.json"))
    print(f"Model saved to: {model_dir}")


def main():
    print("="*60)
    print("  RECOMMENDATION SYSTEM WITH LIGHTFM")
    print("  CodTech IT Solutions - Machine Learning Internship")
    print("  Intern ID: CITS5452")
    print("="*60)

    ensure_dirs()

    ratings, items, users = step1_load_dataset()
    n_users, n_items, sparsity = step2_explore_dataset(ratings, items, users)
    df_clean = step3_clean_data(ratings)
    train_binary, test_binary, mapping_info = step4_prepare_data(
        df_clean, n_users, n_items
    )

    model = step5_train_model(
        train_binary, loss="warp",
        epochs=CONFIG["train_params"]["epochs"],
    )

    step6_recommend(model, train_binary, items, mapping_info)
    metrics = step7_evaluate(model, train_binary, test_binary)

    loss_results, epoch_results = step8_compare_losses_and_epochs(
        train_binary, test_binary
    )

    generate_all_plots(
        ratings, items, epoch_results, loss_results, model
    )

    step9_save_model(model, mapping_info)

    print("\n" + "="*60)
    print("  PROJECT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nOutputs saved in: {CONFIG['outputs_dir']}")
    print(f"  - Graphs:      {CONFIG['graphs_dir']}")
    print(f"  - Model:       {CONFIG['trained_model_dir']}")
    print(f"  - Metrics:     {CONFIG['outputs_dir']}/evaluation_metrics.json")
    print(f"  - Recs CSV:    {CONFIG['recommendations_dir']}")


if __name__ == "__main__":
    main()
