import numpy as np
import pandas as pd
from pathlib import Path


def generate_ratings_data(n_users=500, n_items=200, density=0.05, seed=42):
    rng = np.random.default_rng(seed)

    genres = ['Action', 'Comedie', 'Drame', 'Thriller', 'SF', 'Romance', 'Animation', 'Documentaire']

    items = pd.DataFrame({
        'item_id': range(n_items),
        'titre': [f'Film_{i:03d}' for i in range(n_items)],
        'genre': rng.choice(genres, n_items),
        'annee': rng.integers(1990, 2024, n_items),
        'duree_min': rng.integers(75, 180, n_items),
        'score_critique': np.clip(rng.normal(6.5, 1.5, n_items), 1, 10).round(1),
    })

    # Latent user tastes (K=5 hidden factors)
    K = 5
    user_factors = rng.normal(0, 1, (n_users, K))
    item_factors = rng.normal(0, 1, (n_items, K))

    n_ratings = int(n_users * n_items * density)
    user_ids  = rng.integers(0, n_users, n_ratings)
    item_ids  = rng.integers(0, n_items, n_ratings)

    latent_score = np.sum(user_factors[user_ids] * item_factors[item_ids], axis=1)
    raw_rating = latent_score + rng.normal(0, 0.5, n_ratings)
    # Scale to 1-5
    r_min, r_max = raw_rating.min(), raw_rating.max()
    ratings_cont = 1 + 4 * (raw_rating - r_min) / (r_max - r_min)
    ratings_disc = np.clip(np.round(ratings_cont).astype(int), 1, 5)

    ratings = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings_disc,
    }).drop_duplicates(subset=['user_id', 'item_id'])

    return ratings, items


def load_or_generate(csv_path, **kwargs):
    path = Path(csv_path)
    items_path = path.parent / 'items_simulated.csv'
    if path.exists() and items_path.exists():
        return pd.read_csv(path), pd.read_csv(items_path)
    ratings, items = generate_ratings_data(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(path, index=False)
    items.to_csv(items_path, index=False)
    return ratings, items
