import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr

from src.utils import CONFIG


class LightFM:
    def __init__(self, no_components=30, learning_rate=0.05, max_sampled=10,
                 random_state=42, loss="warp"):
        self.no_components = no_components
        self.learning_rate = learning_rate
        self.max_sampled = max_sampled
        self.random_state = random_state
        self.loss = loss

        self.user_embeddings = None
        self.user_biases = None
        self.item_embeddings = None
        self.item_biases = None
        self._rng = np.random.RandomState(random_state)

    def _init(self, n_users, n_items):
        self.user_embeddings = self._rng.normal(
            0, 0.01, (n_users, self.no_components)
        ).astype(np.float32)
        self.item_embeddings = self._rng.normal(
            0, 0.01, (n_items, self.no_components)
        ).astype(np.float32)
        self.user_biases = np.zeros(n_users, dtype=np.float32)
        self.item_biases = np.zeros(n_items, dtype=np.float32)

    def _predict_one(self, user_idx, item_idx):
        score = (
            self.user_biases[user_idx]
            + self.item_biases[item_idx]
            + np.dot(self.user_embeddings[user_idx], self.item_embeddings[item_idx])
        )
        return score

    def predict(self, user_idx, item_indices):
        scores = (
            self.user_biases[user_idx]
            + self.item_biases[item_indices]
            + np.dot(
                self.item_embeddings[item_indices],
                self.user_embeddings[user_idx],
            )
        )
        return scores

    def _sample_negative(self, n_items, positive_items):
        while True:
            neg = self._rng.randint(0, n_items)
            if neg not in positive_items:
                return neg

    def _update_warp(self, user_idx, pos_idx, neg_idx):
        pos_score = self._predict_one(user_idx, pos_idx)
        neg_score = self._predict_one(user_idx, neg_idx)

        margin = 1.0 + neg_score - pos_score

        if margin > 0.0:
            pos_grad = -1.0
            neg_grad = 1.0

            self.user_embeddings[user_idx] -= (
                self.learning_rate
                * (pos_grad * self.item_embeddings[pos_idx]
                   + neg_grad * self.item_embeddings[neg_idx])
            )
            self.item_embeddings[pos_idx] -= (
                self.learning_rate * pos_grad * self.user_embeddings[user_idx]
            )
            self.item_embeddings[neg_idx] -= (
                self.learning_rate * neg_grad * self.user_embeddings[user_idx]
            )

            self.user_biases[user_idx] -= self.learning_rate * pos_grad
            self.item_biases[pos_idx] -= self.learning_rate * pos_grad
            self.item_biases[neg_idx] -= self.learning_rate * neg_grad

            return True
        return False

    def _update_bpr(self, user_idx, pos_idx, neg_idx):
        pos_score = self._predict_one(user_idx, pos_idx)
        neg_score = self._predict_one(user_idx, neg_idx)

        sigmoid = 1.0 / (1.0 + np.exp(pos_score - neg_score))

        pos_grad = -sigmoid
        neg_grad = sigmoid

        self.user_embeddings[user_idx] -= (
            self.learning_rate
            * (pos_grad * self.item_embeddings[pos_idx]
               + neg_grad * self.item_embeddings[neg_idx])
        )
        self.item_embeddings[pos_idx] -= (
            self.learning_rate * pos_grad * self.user_embeddings[user_idx]
        )
        self.item_embeddings[neg_idx] -= (
            self.learning_rate * neg_grad * self.user_embeddings[user_idx]
        )

        self.user_biases[user_idx] -= self.learning_rate * pos_grad
        self.item_biases[pos_idx] -= self.learning_rate * pos_grad
        self.item_biases[neg_idx] -= self.learning_rate * neg_grad

    def _update_logistic(self, user_idx, pos_idx, neg_idx):
        pos_score = self._predict_one(user_idx, pos_idx)
        neg_score = self._predict_one(user_idx, neg_idx)

        pos_sigmoid = 1.0 / (1.0 + np.exp(-pos_score))
        neg_sigmoid = 1.0 / (1.0 + np.exp(-neg_score))

        pos_grad = -(1.0 - pos_sigmoid)
        neg_grad = neg_sigmoid

        self.user_embeddings[user_idx] -= (
            self.learning_rate
            * (pos_grad * self.item_embeddings[pos_idx]
               + neg_grad * self.item_embeddings[neg_idx])
        )
        self.item_embeddings[pos_idx] -= (
            self.learning_rate * pos_grad * self.user_embeddings[user_idx]
        )
        self.item_embeddings[neg_idx] -= (
            self.learning_rate * neg_grad * self.user_embeddings[user_idx]
        )

        self.user_biases[user_idx] -= self.learning_rate * pos_grad
        self.item_biases[pos_idx] -= self.learning_rate * pos_grad
        self.item_biases[neg_idx] -= self.learning_rate * neg_grad

    def fit(self, interactions, epochs=10, num_threads=1, verbose=True):
        if not isspmatrix_csr(interactions):
            interactions = interactions.tocsr()

        n_users, n_items = interactions.shape
        self._init(n_users, n_items)

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_updates = 0

            for user_idx in range(n_users):
                item_indices = interactions[user_idx].indices
                if len(item_indices) == 0:
                    continue

                positive_items = set(item_indices)

                for pos_idx in item_indices:
                    neg_idx = self._sample_negative(n_items, positive_items)

                    if self.loss == "warp":
                        success = self._update_warp(user_idx, pos_idx, neg_idx)
                        if success:
                            num_updates += 1
                    elif self.loss == "bpr":
                        self._update_bpr(user_idx, pos_idx, neg_idx)
                        num_updates += 1
                    elif self.loss == "logistic":
                        self._update_logistic(user_idx, pos_idx, neg_idx)
                        num_updates += 1

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} completed")
                print(f"  Updates: {num_updates}")

        return self

    def predict_rank(self, interactions, k=10):
        n_users, n_items = interactions.shape
        ranks = np.zeros(n_users, dtype=np.int32)

        for user_idx in range(n_users):
            item_indices = interactions[user_idx].indices
            if len(item_indices) == 0:
                ranks[user_idx] = n_items
                continue

            all_scores = self.predict(user_idx, np.arange(n_items))
            sorted_items = np.argsort(-all_scores)

            pos = 0
            for i, item_idx in enumerate(sorted_items[:k]):
                if item_idx in item_indices:
                    pos += 1

            ranks[user_idx] = pos

        return ranks.mean()


def train_model(interactions, loss="warp", epochs=30):
    params = CONFIG["model_params"]
    model = LightFM(
        no_components=params["no_components"],
        learning_rate=params["learning_rate"],
        max_sampled=params["max_sampled"],
        random_state=params["random_state"],
        loss=loss,
    )
    model.fit(interactions, epochs=epochs, verbose=True)
    return model
