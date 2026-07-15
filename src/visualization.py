import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import CONFIG


def set_style():
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["font.size"] = 12


def plot_rating_distribution(ratings_df):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(ratings_df["rating"], bins=5, edgecolor="black",
                 color="steelblue", alpha=0.7)
    axes[0].set_title("Rating Distribution", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Rating")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3)

    rating_counts = ratings_df["rating"].value_counts().sort_index()
    axes[1].bar(rating_counts.index, rating_counts.values,
                color="coral", edgecolor="black", alpha=0.7)
    axes[1].set_title("Rating Frequency", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Rating")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(CONFIG["graphs_dir"], "rating_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_interaction_sparsity(ratings_df, title="Interaction Matrix Sparsity"):
    set_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    n_users = ratings_df["user_id"].nunique()
    n_items = ratings_df["item_id"].nunique()
    n_interactions = len(ratings_df)
    total_possible = n_users * n_items
    sparsity = (1 - n_interactions / total_possible) * 100

    labels = ["Interactions", "No Interaction"]
    sizes = [n_interactions, total_possible - n_interactions]
    colors = ["#2ecc71", "#ecf0f1"]
    explode = (0.05, 0)

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.2f%%",
        colors=colors, explode=explode, startangle=90,
        textprops={"fontsize": 12},
    )
    for at in autotexts:
        at.set_fontweight("bold")

    ax.set_title(
        f"{title}\n({n_users} users, {n_items} items, {n_interactions} interactions)",
        fontsize=14, fontweight="bold",
    )

    save_path = os.path.join(CONFIG["graphs_dir"], "interaction_sparsity.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_top_users(ratings_df, top_n=15):
    set_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    user_counts = ratings_df["user_id"].value_counts().head(top_n)
    bars = ax.bar(
        range(len(user_counts)), user_counts.values,
        color=plt.cm.viridis(np.linspace(0.2, 0.8, top_n)),
        edgecolor="black", linewidth=0.5,
    )

    ax.set_title(f"Top {top_n} Most Active Users", fontsize=14, fontweight="bold")
    ax.set_xlabel("User ID")
    ax.set_ylabel("Number of Ratings")
    ax.set_xticks(range(len(user_counts)))
    ax.set_xticklabels(user_counts.index, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, user_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(val), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(CONFIG["graphs_dir"], "top_active_users.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_top_items(ratings_df, item_df, top_n=15):
    set_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    item_counts = ratings_df["item_id"].value_counts().head(top_n)
    item_counts = item_counts.reset_index()
    item_counts.columns = ["item_id", "count"]
    item_counts = item_counts.merge(
        item_df[["item_id", "title"]], on="item_id", how="left"
    )
    item_counts["label"] = item_counts["title"].fillna(
        "Item " + item_counts["item_id"].astype(str)
    )
    item_counts["short"] = item_counts["label"].str.slice(0, 30)

    bars = ax.barh(
        range(len(item_counts)), item_counts["count"],
        color=plt.cm.plasma(np.linspace(0.2, 0.8, top_n)),
        edgecolor="black", linewidth=0.5,
    )

    ax.set_title(f"Top {top_n} Most Rated Items", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Ratings")
    ax.set_yticks(range(len(item_counts)))
    ax.set_yticklabels(item_counts["short"], fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    for bar, val in zip(bars, item_counts["count"]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(val), ha="left", va="center", fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(CONFIG["graphs_dir"], "top_rated_items.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_training_comparison(epoch_results_df):
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = ["Precision@K", "Recall@K", "AUC", "RankingLoss"]
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

    for ax, metric, color in zip(axes.flat, metrics, colors):
        ax.plot(epoch_results_df["Epoch"], epoch_results_df[metric],
                color=color, linewidth=2, marker="o", markersize=3)
        ax.set_title(f"{metric} over Epochs", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(CONFIG["graphs_dir"], "training_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_loss_comparison(loss_results_df):
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = ["Precision@K", "Recall@K", "AUC", "RankingLoss"]
    palette = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

    for ax, metric, col in zip(axes.flat, metrics, palette):
        losses = loss_results_df["Loss"]
        values = loss_results_df[metric]
        bars = ax.bar(losses, values, color=[col] * len(losses),
                      edgecolor="black", linewidth=1.2)
        ax.set_title(f"{metric} by Loss Function", fontsize=13, fontweight="bold")
        ax.set_xlabel("Loss Function")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=10,
                    fontweight="bold")

    plt.tight_layout()
    save_path = os.path.join(CONFIG["graphs_dir"], "loss_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_embedding_visualization(model, title="User & Item Embeddings (PCA)"):
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("sklearn not available for PCA visualization")
        return

    set_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    user_emb = model.user_embeddings
    item_emb = model.item_embeddings
    combined = np.vstack([user_emb, item_emb])

    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(combined)

    n_users = user_emb.shape[0]
    user_2d = reduced[:n_users]
    item_2d = reduced[n_users:]

    ax.scatter(user_2d[:, 0], user_2d[:, 1], c="#3498db", label="Users",
               alpha=0.6, s=30, edgecolors="black", linewidth=0.3)
    ax.scatter(item_2d[:, 0], item_2d[:, 1], c="#e74c3c", label="Items",
               alpha=0.6, s=30, edgecolors="black", linewidth=0.3)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(CONFIG["graphs_dir"], "embedding_visualization.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def generate_all_plots(ratings_df, item_df, epoch_results=None, loss_results=None,
                       model=None):
    print("\nGenerating visualizations...")

    plot_rating_distribution(ratings_df)
    plot_interaction_sparsity(ratings_df)
    plot_top_users(ratings_df)
    plot_top_items(ratings_df, item_df)

    if epoch_results is not None:
        plot_training_comparison(epoch_results)

    if loss_results is not None:
        plot_loss_comparison(loss_results)

    if model is not None:
        plot_embedding_visualization(model)
