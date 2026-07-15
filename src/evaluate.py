import numpy as np
import pandas as pd
from src.utils import CONFIG


def precision_at_k(model, test_interactions, train_interactions, k=10):
    n_users = test_interactions.shape[0]
    precision_scores = []

    for user_idx in range(n_users):
        test_items = set(test_interactions[user_idx].indices)
        if len(test_items) == 0:
            continue

        train_items = set(train_interactions[user_idx].indices)
        n_items = test_interactions.shape[1]
        all_scores = model.predict(user_idx, np.arange(n_items))
        top_k_indices = np.argsort(-all_scores)[:k]

        recommended = set(top_k_indices) - train_items
        hits = len(recommended & test_items)
        precision_scores.append(hits / k)

    return np.mean(precision_scores) if precision_scores else 0.0


def recall_at_k(model, test_interactions, train_interactions, k=10):
    n_users = test_interactions.shape[0]
    recall_scores = []

    for user_idx in range(n_users):
        test_items = set(test_interactions[user_idx].indices)
        if len(test_items) == 0:
            continue

        train_items = set(train_interactions[user_idx].indices)
        n_items = test_interactions.shape[1]
        all_scores = model.predict(user_idx, np.arange(n_items))
        top_k_indices = np.argsort(-all_scores)[:k]

        recommended = set(top_k_indices) - train_items
        hits = len(recommended & test_items)
        recall_scores.append(hits / len(test_items))

    return np.mean(recall_scores) if recall_scores else 0.0


def auc_score(model, test_interactions, train_interactions):
    n_users = test_interactions.shape[0]
    auc_scores = []

    for user_idx in range(n_users):
        test_items = set(test_interactions[user_idx].indices)
        if len(test_items) == 0:
            continue

        train_items = set(train_interactions[user_idx].indices)
        n_items = test_interactions.shape[1]
        all_scores = model.predict(user_idx, np.arange(n_items))
        sorted_indices = np.argsort(-all_scores)

        test_items_remaining = set(test_items)
        n_pos = len(test_items_remaining)
        if n_pos == 0:
            continue

        n_neg = 0
        rank_sum = 0

        for item_idx in sorted_indices:
            if item_idx in train_items:
                continue
            if item_idx in test_items_remaining:
                rank_sum += n_neg
                test_items_remaining.remove(item_idx)
            else:
                n_neg += 1

            if not test_items_remaining:
                break

        if n_neg > 0:
            auc_scores.append(rank_sum / (n_pos * n_neg))

    return np.mean(auc_scores) if auc_scores else 0.0


def ranking_loss(model, test_interactions, train_interactions):
    n_users = test_interactions.shape[0]
    losses = []

    for user_idx in range(n_users):
        test_items = set(test_interactions[user_idx].indices)
        if len(test_items) < 2:
            continue

        train_items = set(train_interactions[user_idx].indices)
        n_items = test_interactions.shape[1]
        all_scores = model.predict(user_idx, np.arange(n_items))

        test_list = list(test_items)
        incorrect_pairs = 0
        total_pairs = 0

        for i in range(len(test_list)):
            for j in range(i + 1, len(test_list)):
                if all_scores[test_list[i]] < all_scores[test_list[j]]:
                    incorrect_pairs += 1
                total_pairs += 1

        if total_pairs > 0:
            losses.append(incorrect_pairs / total_pairs)

    return np.mean(losses) if losses else 0.0


def evaluate_model(model, test_interactions, train_interactions, k=10):
    precision = precision_at_k(model, test_interactions, train_interactions, k)
    recall = recall_at_k(model, test_interactions, train_interactions, k)
    auc = auc_score(model, test_interactions, train_interactions)
    rl = ranking_loss(model, test_interactions, train_interactions)

    return {
        "Precision@K": round(precision, 4),
        "Recall@K": round(recall, 4),
        "AUC": round(auc, 4),
        "RankingLoss": round(rl, 4),
    }


def compare_losses(train_interactions, test_interactions, epochs=30):
    losses = ["warp", "bpr", "logistic"]
    results = []

    for loss in losses:
        print(f"\n{'='*50}")
        print(f"Training with loss: {loss}")
        print(f"{'='*50}")

        from src.train import LightFM
        params = CONFIG["model_params"]
        model = LightFM(
            no_components=params["no_components"],
            learning_rate=params["learning_rate"],
            max_sampled=params["max_sampled"],
            random_state=params["random_state"],
            loss=loss,
        )
        model.fit(train_interactions, epochs=epochs, verbose=True)

        metrics = evaluate_model(
            model, test_interactions, train_interactions, k=CONFIG["eval_params"]["k"]
        )
        metrics["Loss"] = loss
        results.append(metrics)

    return pd.DataFrame(results)


def compare_epochs(train_interactions, test_interactions, loss="warp", max_epochs=30):
    from src.train import LightFM
    params = CONFIG["model_params"]
    model = LightFM(
        no_components=params["no_components"],
        learning_rate=params["learning_rate"],
        max_sampled=params["max_sampled"],
        random_state=params["random_state"],
        loss=loss,
    )

    results = []
    for epoch in range(1, max_epochs + 1):
        model.fit(train_interactions, epochs=1, verbose=False)
        metrics = evaluate_model(
            model, test_interactions, train_interactions,
            k=CONFIG["eval_params"]["k"]
        )
        metrics["Epoch"] = epoch
        results.append(metrics)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: P@K={metrics['Precision@K']}, "
                  f"R@K={metrics['Recall@K']}, AUC={metrics['AUC']}")

    return pd.DataFrame(results)
