import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error


def build_user_item_matrix(ratings):
    return ratings.pivot_table(index='user_id', columns='item_id',
                               values='rating', fill_value=0)


# ── Collaborative Filtering ─────────────────────────────────────────────────

class UserBasedCF:
    def __init__(self, k_neighbors=20):
        self.k = k_neighbors
        self.matrix = None
        self.sim = None

    def fit(self, ratings):
        self.matrix = build_user_item_matrix(ratings)
        mat = self.matrix.values.astype(float)
        # Mean-center rows
        row_mean = np.where(mat != 0, mat, np.nan)
        self.user_mean = np.nanmean(row_mean, axis=1, keepdims=True)
        centered = np.where(mat != 0, mat - self.user_mean, 0)
        self.sim = cosine_similarity(centered)
        return self

    def predict(self, user_id, item_id):
        if user_id not in self.matrix.index or item_id not in self.matrix.columns:
            return self.matrix.values[self.matrix.values != 0].mean()
        u_idx  = self.matrix.index.get_loc(user_id)
        i_idx  = self.matrix.columns.get_loc(item_id)
        sim_row = self.sim[u_idx].copy()
        sim_row[u_idx] = 0
        top_k  = np.argsort(sim_row)[-self.k:]
        top_sim = sim_row[top_k]
        top_rat = self.matrix.values[top_k, i_idx]
        rated   = top_rat != 0
        if not rated.any():
            return float(self.user_mean[u_idx])
        num = np.sum(top_sim[rated] * top_rat[rated])
        den = np.sum(np.abs(top_sim[rated])) + 1e-9
        return float(self.user_mean[u_idx] + num / den)

    def recommend(self, user_id, n=10):
        u_idx = self.matrix.index.get_loc(user_id)
        rated = set(self.matrix.columns[self.matrix.values[u_idx] != 0])
        unseen = [i for i in self.matrix.columns if i not in rated]
        scores = [(i, self.predict(user_id, i)) for i in unseen]
        return sorted(scores, key=lambda x: -x[1])[:n]


# ── Matrix Factorization (SVD) ───────────────────────────────────────────────

class SVDRecommender:
    def __init__(self, n_components=20):
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.matrix = None

    def fit(self, ratings):
        self.matrix = build_user_item_matrix(ratings)
        self.matrix_approx = pd.DataFrame(
            self.svd.fit_transform(self.matrix) @ self.svd.components_,
            index=self.matrix.index,
            columns=self.matrix.columns,
        )
        return self

    def recommend(self, user_id, n=10):
        u_idx = self.matrix.index.get_loc(user_id)
        rated = set(self.matrix.columns[self.matrix.values[u_idx] != 0])
        scores = self.matrix_approx.loc[user_id]
        unseen = scores.drop(index=list(rated & set(scores.index)), errors='ignore')
        return list(unseen.nlargest(n).items())

    def explained_variance(self):
        return self.svd.explained_variance_ratio_.sum()


# ── Content-Based ────────────────────────────────────────────────────────────

def content_based_recommend(user_id, ratings, items, n=10):
    user_ratings = ratings[ratings['user_id'] == user_id]
    if user_ratings.empty:
        return items.sample(n)['item_id'].tolist()

    liked = user_ratings[user_ratings['rating'] >= 4]['item_id'].values
    liked_genres = items[items['item_id'].isin(liked)]['genre'].value_counts()
    if liked_genres.empty:
        return items.sample(n)['item_id'].tolist()

    seen = set(user_ratings['item_id'].values)
    unseen = items[~items['item_id'].isin(seen)].copy()
    top_genre = liked_genres.index[0]
    unseen['score'] = (unseen['genre'] == top_genre).astype(int) * 2 + unseen['score_critique'] / 10
    return unseen.nlargest(n, 'score')['item_id'].tolist()


# ── Evaluation ───────────────────────────────────────────────────────────────

def train_test_split_ratings(ratings, test_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ratings), size=int(len(ratings) * test_frac), replace=False)
    test  = ratings.iloc[idx].reset_index(drop=True)
    train = ratings.drop(index=ratings.index[idx]).reset_index(drop=True)
    return train, test


def evaluate_cf(model, test):
    preds, actuals = [], []
    for _, row in test.iterrows():
        p = model.predict(row['user_id'], row['item_id'])
        preds.append(p)
        actuals.append(row['rating'])
    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(np.mean((np.array(actuals) - np.array(preds)) ** 2))
    print(f'MAE={mae:.4f}  RMSE={rmse:.4f}')
    return dict(mae=mae, rmse=rmse)


def plot_rating_distribution(ratings, items):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ratings['rating'].value_counts().sort_index().plot.bar(ax=axes[0], color='#2196F3')
    axes[0].set_title('Distribution des notes'); axes[0].set_xlabel('Note')

    ratings_per_user = ratings.groupby('user_id').size()
    axes[1].hist(ratings_per_user, bins=30, color='#4CAF50', edgecolor='white')
    axes[1].set_title('Notes par utilisateur'); axes[1].set_xlabel('Nb notes')

    items['genre'].value_counts().plot.bar(ax=axes[2], color='#FF9800', edgecolor='white')
    axes[2].set_title('Films par genre'); axes[2].tick_params(axis='x', rotation=45)

    plt.suptitle('Exploration — Système de recommandation', fontweight='bold')
    plt.tight_layout(); plt.show()
