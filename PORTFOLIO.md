# CAS D'USAGE 14 — Système de Recommandation
## Recommander des produits pertinents par filtrage collaboratif et contenu

> **Auteur :** TSAGUE EMMANUEL — Data Scientist / Data Analyst  
> **Domaine :** Recommandation, NLP basique, Algèbre linéaire appliquée  
> **Repository GitHub :** `recommendation-system-portfolio`  
> **Statut :** Portfolio — données simulées  
> **Date :** Juin 2026

---
## 1. TITRE ET RÉSUMÉ EXÉCUTIF

**"Système de recommandation e-commerce : filtrage collaboratif user-based + filtrage contenu + gestion du cold start"**

> **Système de recommandation :** algorithme qui prédit les préférences d'un utilisateur pour des items qu'il n'a pas encore vus, afin de lui suggérer les plus pertinents. Amazon, Netflix et Spotify s'en servent à grande échelle.

Ce projet construit un système de recommandation pour un site e-commerce simulé (10 000 utilisateurs, 500 produits). Il couvre deux familles d'approches : filtrage collaboratif (basé sur les comportements) et filtrage contenu (basé sur les attributs des produits).

**Résultats simulés :** Precision@10 = 0,31 | Recall@10 = 0,24 | NDCG@10 = 0,38.

---
## 2. LES DEUX GRANDES FAMILLES D'APPROCHES

> **Filtrage collaboratif (Collaborative Filtering — CF) :** "Les utilisateurs qui ont aimé les mêmes produits que toi ont aussi aimé X → je te recommande X." Ne nécessite pas de connaître le contenu des produits. Requiert des données d'interaction (achats, notes, clics).

> **Filtrage par contenu (Content-Based Filtering) :** "Tu as aimé ce produit avec ces caractéristiques → je te recommande d'autres produits avec des caractéristiques similaires." Ne dépend pas des autres utilisateurs. Requiert des attributs descriptifs des produits.

| Aspect | Filtrage Collaboratif | Filtrage Contenu |
|--------|----------------------|-----------------|
| Données | Interactions utilisateurs | Attributs produits |
| Cold Start utilisateur | Problème | OK |
| Cold Start produit | Problème | OK |
| Diversité | Forte | Faible (surspécialisation) |
| Interprétabilité | Faible | Forte |

> **Cold Start :** problème des nouveaux utilisateurs (pas d'historique) ou nouveaux produits (pas d'interactions). Algorithme hybride ou règles métier pour pallier.

---
## 3. GÉNÉRATION DES DONNÉES SIMULÉES

```python
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

np.random.seed(42)
N_USERS    = 10_000
N_ITEMS    = 500
N_RATINGS  = 150_000  # ~15 interactions par utilisateur en moyenne

# Matrice d'interactions (ratings simulés 1-5)
user_ids    = np.random.randint(0, N_USERS, N_RATINGS)
item_ids    = np.random.randint(0, N_ITEMS, N_RATINGS)

# Simuler des goûts : certains utilisateurs préfèrent certaines catégories
categories = np.random.randint(0, 10, N_ITEMS)  # 10 catégories
user_cat_affinity = np.random.dirichlet(np.ones(10), N_USERS)  # Affinité par catégorie

ratings = []
for u, i in zip(user_ids, item_ids):
    base = 3.0
    affinity = user_cat_affinity[u, categories[i]]
    rating = base + affinity * 2 + np.random.normal(0, 0.5)
    rating = np.clip(round(rating), 1, 5)
    ratings.append(rating)

df_ratings = pd.DataFrame({"user_id": user_ids, "item_id": item_ids, "rating": ratings})
df_ratings = df_ratings.drop_duplicates(subset=["user_id", "item_id"])

print(f"Interactions uniques : {len(df_ratings):,}")
print(f"Densité de la matrice : {len(df_ratings)/(N_USERS*N_ITEMS):.2%}")

# Catalogue produits
df_items = pd.DataFrame({
    "item_id":   range(N_ITEMS),
    "categorie": categories,
    "prix":      np.random.uniform(5, 500, N_ITEMS).round(2),
    "note_moy":  np.random.uniform(2.5, 5.0, N_ITEMS).round(1),
    "desc":      [f"Produit {i} catégorie {categories[i]}" for i in range(N_ITEMS)],
})
```

---
## 4. FILTRAGE COLLABORATIF — USER-BASED (KNN)

> **User-Based CF :** pour recommander à l'utilisateur A, on cherche les utilisateurs "similaires" à A (voisins), puis on recommande ce qu'ils ont aimé et que A n'a pas encore vu.

> **Cosine Similarity :** mesure la similarité entre deux vecteurs en calculant le cosinus de l'angle entre eux. Valeur entre -1 (opposés) et 1 (identiques). Très utilisée pour les systèmes de recommandation.

> **Sparse Matrix (matrice creuse) :** matrice avec beaucoup de zéros. Une matrice utilisateurs × produits est typiquement creuse (un utilisateur note rarement 1 % des produits disponibles). Les représentations creuses économisent la mémoire.

```python
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import numpy as np

# Construire la matrice utilisateur × item
user_item_matrix = df_ratings.pivot_table(
    index="user_id", columns="item_id", values="rating", fill_value=0
)

# Matrice creuse pour l'efficacité mémoire
user_item_sparse = csr_matrix(user_item_matrix.values)
print(f"Matrice : {user_item_sparse.shape} | "
      f"Mémoire sparse : {user_item_sparse.data.nbytes/1e6:.1f} MB")

def recommander_user_based(user_id, n_voisins=20, n_reco=10):
    """
    Recommande N items à un utilisateur par filtrage collaboratif user-based.

    Étapes :
    1. Calculer la similarité entre user_id et tous les autres utilisateurs
    2. Sélectionner les N voisins les plus proches
    3. Pondérer les ratings de ces voisins par leur similarité
    4. Recommander les items non encore vus avec le score pondéré le plus élevé
    """
    if user_id not in user_item_matrix.index:
        return []  # Cold start : pas d'historique

    # Similarité cosinus avec tous les autres utilisateurs
    user_vector = user_item_sparse[user_item_matrix.index.get_loc(user_id)]
    similarities = cosine_similarity(user_vector, user_item_sparse)[0]

    # Exclure l'utilisateur lui-même
    similarities[user_item_matrix.index.get_loc(user_id)] = 0

    # Top N voisins
    top_voisins_idx = np.argsort(similarities)[-n_voisins:]
    top_voisins_sim = similarities[top_voisins_idx]

    # Items déjà vus par l'utilisateur
    items_vus = set(df_ratings[df_ratings["user_id"] == user_id]["item_id"])

    # Score pondéré par voisin
    scores = {}
    for voisin_idx, sim in zip(top_voisins_idx, top_voisins_sim):
        voisin_id = user_item_matrix.index[voisin_idx]
        voisin_ratings = df_ratings[df_ratings["user_id"] == voisin_id]
        for _, row in voisin_ratings.iterrows():
            item = int(row["item_id"])
            if item not in items_vus:
                scores[item] = scores.get(item, 0) + sim * row["rating"]

    # Top N recommandations
    reco = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_reco]
    return [item_id for item_id, _ in reco]

# Test
reco = recommander_user_based(user_id=42, n_voisins=20, n_reco=10)
print(f"Recommandations pour l'utilisateur 42 : {reco}")
```

---
## 5. FILTRAGE COLLABORATIF — FACTORISATION MATRICIELLE

> **Factorisation matricielle (Matrix Factorization) :** décompose la matrice utilisateurs × items en deux matrices plus petites : une matrice utilisateurs × facteurs_latents et une matrice items × facteurs_latents. Chaque utilisateur et item est représenté par un vecteur de facteurs latents (embeddings). La recommandation = produit scalaire de ces vecteurs.

> **SVD (Singular Value Decomposition) :** décomposition mathématique qui factorise une matrice en trois matrices : U × Σ × Vᵀ. En recommandation, on conserve uniquement les K plus grandes valeurs singulières pour la compression.

```python
from sklearn.decomposition import TruncatedSVD

# TruncatedSVD sur la matrice user-item (équivalent SVD creux)
n_factors = 50
svd = TruncatedSVD(n_components=n_factors, random_state=42)
user_factors = svd.fit_transform(user_item_sparse)  # (N_users, 50)
item_factors = svd.components_.T                      # (N_items, 50)

print(f"Variance expliquée par {n_factors} facteurs : "
      f"{svd.explained_variance_ratio_.sum():.1%}")

def recommander_svd(user_id, n_reco=10):
    """Recommande via factorisation matricielle (SVD)."""
    if user_id not in user_item_matrix.index:
        return []

    user_idx    = user_item_matrix.index.get_loc(user_id)
    user_vec    = user_factors[user_idx]

    # Scores = produit scalaire entre vecteur utilisateur et tous les items
    scores_items = item_factors @ user_vec
    items_vus    = set(df_ratings[df_ratings["user_id"] == user_id]["item_id"])

    reco = []
    for item_id in np.argsort(scores_items)[::-1]:
        if item_id not in items_vus:
            reco.append(item_id)
        if len(reco) == n_reco:
            break
    return reco
```

---
## 6. FILTRAGE PAR CONTENU

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise        import cosine_similarity

# TF-IDF sur les descriptions produits
# TF-IDF (Term Frequency-Inverse Document Frequency) :
# pondère les mots importants dans un document par rapport au corpus entier
tfidf = TfidfVectorizer(max_features=200, ngram_range=(1, 2))
item_matrix_tfidf = tfidf.fit_transform(df_items["desc"])

# Enrichissement avec features numériques
from scipy.sparse import hstack
from sklearn.preprocessing import MinMaxScaler

item_num = df_items[["prix", "note_moy", "categorie"]].values
scaler = MinMaxScaler()
item_num_scaled = scaler.fit_transform(item_num)

# Matrice item hybride : TF-IDF + numériques
from scipy.sparse import csr_matrix
item_content_matrix = hstack([item_matrix_tfidf, csr_matrix(item_num_scaled)])

def recommander_content_based(item_id, n_reco=10):
    """Recommande des items similaires à item_id (filtrage contenu)."""
    item_vec = item_content_matrix[item_id]
    scores   = cosine_similarity(item_vec, item_content_matrix)[0]
    scores[item_id] = 0  # Exclure l'item lui-même
    top_items = np.argsort(scores)[-n_reco:][::-1]
    return list(top_items)
```

---
## 7. ÉVALUATION — PRECISION@K ET RECALL@K

> **Precision@K :** parmi les K items recommandés, quelle proportion est vraiment pertinente ?

> **Recall@K :** parmi tous les items pertinents pour l'utilisateur, quelle proportion est dans les K recommandations ?

> **NDCG@K (Normalized Discounted Cumulative Gain) :** mesure la qualité du classement — un item pertinent en position 1 vaut plus qu'en position 10.

```python
from sklearn.model_selection import train_test_split as tts

# Split temporel : entraîner sur 80%, évaluer sur 20%
df_sorted = df_ratings.sort_values("item_id")
train_size = int(len(df_sorted) * 0.8)
df_train = df_sorted.iloc[:train_size]
df_test  = df_sorted.iloc[train_size:]

# Items de test par utilisateur
test_items_by_user = df_test.groupby("user_id")["item_id"].apply(set).to_dict()

def precision_recall_at_k(recommender, users, k=10):
    precisions, recalls = [], []
    for user_id in users[:200]:  # Sous-ensemble pour la rapidité
        if user_id not in test_items_by_user:
            continue
        recos       = set(recommender(user_id, n_reco=k))
        relevant    = test_items_by_user[user_id]
        hits        = recos & relevant
        precisions.append(len(hits) / k)
        recalls.append(len(hits) / len(relevant) if relevant else 0)
    return np.mean(precisions), np.mean(recalls)

users_test = list(test_items_by_user.keys())
p_ub, r_ub = precision_recall_at_k(recommander_user_based, users_test, k=10)
p_svd, r_svd = precision_recall_at_k(recommander_svd, users_test, k=10)

print(f"\n=== ÉVALUATION @10 ===")
print(f"User-Based CF  | Precision : {p_ub:.4f} | Recall : {r_ub:.4f}")
print(f"SVD            | Precision : {p_svd:.4f} | Recall : {r_svd:.4f}")
```

---
## 8. COLD START — STRATÉGIE HYBRIDE

```python
def recommander_hybride(user_id, n_reco=10):
    """
    Stratégie hybride : gestion du cold start.
    - Nouvel utilisateur (< 5 interactions) → Popularité globale
    - Utilisateur établi → SVD
    """
    n_interactions = len(df_ratings[df_ratings["user_id"] == user_id])

    if n_interactions == 0:
        # Froid total : recommander les items les plus populaires
        top_items = (df_ratings
                     .groupby("item_id")["rating"]
                     .agg(["mean", "count"])
                     .query("count >= 10")
                     .sort_values("mean", ascending=False)
                     .head(n_reco)
                     .index.tolist())
        return top_items, "popularite"

    elif n_interactions < 5:
        # Warm start : filtrage contenu basé sur les derniers items vus
        derniers_items = df_ratings[df_ratings["user_id"] == user_id]["item_id"].tolist()
        scores = np.zeros(N_ITEMS)
        for item in derniers_items[-3:]:  # 3 derniers items
            recos = recommander_content_based(item, n_reco=20)
            for r in recos:
                scores[r] += 1
        top_items = np.argsort(scores)[-n_reco:][::-1].tolist()
        return top_items, "content_warm_start"

    else:
        # Utilisateur établi : SVD collaboratif
        return recommander_svd(user_id, n_reco), "svd_collaboratif"
```

---
## 9. ARCHITECTURE GITHUB

```
recommendation-system-portfolio/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_eda_interactions.ipynb
│   ├── 02_collaborative_filtering.ipynb
│   ├── 03_matrix_factorization_svd.ipynb
│   ├── 04_content_based_filtering.ipynb
│   ├── 05_evaluation_metrics.ipynb
│   └── 06_cold_start_strategy.ipynb
└── src/
    ├── collaborative.py
    ├── content.py
    ├── hybrid.py
    └── evaluation.py
```

---
## 15. COMPÉTENCES DÉMONTRÉES

| Compétence | Preuve |
|-----------|--------|
| Filtrage collaboratif | User-based cosine similarity |
| SVD / Embeddings | TruncatedSVD facteurs latents |
| Filtrage contenu | TF-IDF + features numériques |
| Precision@K / NDCG | Évaluation offline rigoureuse |
| Cold Start | Stratégie hybride multi-niveaux |

---

*Fin du document — TSAGUE EMMANUEL — CAS 14 — Système de Recommandation*
---

## Contact & Liens

**TSAGUE EMMANUEL** - Data Scientist

| | |
|---|---|
| Email | [emmatsague@yahoo.fr](mailto:emmatsague@yahoo.fr) |
| GitHub | [github.com/TSAGUE25](https://github.com/TSAGUE25) |
| Formation | Datascientest 2024 |
| Experience | EDF MAD EDVANCE |
| Domaines | Machine Learning - Data Analysis - Energie |

---

> Toutes les donnees de ce depot sont simulees et anonymisees.  
> Aucune donnee reelle ou confidentielle n'est presente.
