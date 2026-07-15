import numpy as np
import pandas as pd


def recommend_for_user(model, user_idx, item_ids, item_titles, top_k=10):
    n_items = model.item_embeddings.shape[0]
    all_scores = model.predict(user_idx, np.arange(n_items))
    top_indices = np.argsort(-all_scores)[:top_k]

    recommendations = []
    for rank, item_idx in enumerate(top_indices, 1):
        original_id = item_ids[item_idx]
        title = item_titles.get(original_id, f"Item {original_id}")
        score = float(all_scores[item_idx])
        recommendations.append({
            "rank": rank,
            "item_id": int(original_id),
            "title": title,
            "score": round(score, 4),
        })

    return recommendations


def generate_recommendations(model, interactions, item_df, user_map, top_k=10):
    n_users = interactions.shape[0]
    item_ids = user_map["item_rev"]
    item_titles = dict(zip(item_df["item_id"], item_df["title"]))

    all_recommendations = {}
    for user_idx in range(min(n_users, 10)):
        user_id = user_map["user_rev"][user_idx]
        recs = recommend_for_user(
            model, user_idx, item_ids, item_titles, top_k
        )
        all_recommendations[int(user_id)] = recs

    return all_recommendations


def format_recommendations(recs_dict):
    lines = []
    for user_id, recs in recs_dict.items():
        lines.append(f"\n=== Top {len(recs)} Recommendations for User {user_id} ===")
        lines.append(f"{'Rank':<6} {'Item ID':<10} {'Score':<10} Title")
        lines.append("-" * 60)
        for r in recs:
            lines.append(
                f"{r['rank']:<6} {r['item_id']:<10} {r['score']:<10} {r['title']}"
            )
    return "\n".join(lines)


def save_recommendations(recs_dict, filepath):
    rows = []
    for user_id, recs in recs_dict.items():
        for r in recs:
            rows.append({
                "user_id": user_id,
                "rank": r["rank"],
                "item_id": r["item_id"],
                "title": r["title"],
                "score": r["score"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    return df
