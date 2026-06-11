# Système de Recommandation E-commerce

> **Filtrage collaboratif, SVD et stratégie cold start sur 10 000 utilisateurs simulés**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Domaine](https://img.shields.io/badge/Domaine-E-commerce-green)
![Statut](https://img.shields.io/badge/Statut-Portfolio-orange)
![Données](https://img.shields.io/badge/Données-Simulées%2FAnonymisées-lightgrey)

---

## Contexte métier

Les systèmes de recommandation sont au cœur des stratégies de personnalisation d'Amazon, Netflix et Spotify. Ils permettent d'augmenter le panier moyen, l'engagement et la rétention via des suggestions pertinentes.

---

## Problème traité

10 000 utilisateurs, 500 produits, 150 000 interactions simulées. Recommander les produits pertinents à chaque utilisateur en gérant le problème du cold start pour les nouveaux utilisateurs.

---

## Solution proposée

User-Based CF (similarité cosinus), factorisation matricielle SVD (50 facteurs latents), filtrage contenu TF-IDF, stratégie hybride cold start : popularité → contenu (warm) → collaboratif (établi).

---

## Technologies utilisées

| Outil | Usage |
|-------|-------|
| Python 3.10+ | Langage principal |
| pandas / numpy | Manipulation des données |
| scikit-learn | Machine Learning & preprocessing |
| matplotlib / seaborn | Visualisation |
| Jupyter Notebook | Exploration interactive |

> Voir `requirements.txt` pour la liste complète.

---

## Structure du projet

```
recommendation-system-portfolio/
├── README.md              ← Ce fichier
├── PORTFOLIO.md           ← Documentation complète du cas d'usage
├── .gitignore
├── requirements.txt
├── notebooks/             ← Jupyter Notebooks d'exploration
├── src/                   ← Code Python modulaire
├── data_sample/           ← Données simulées (anonymisées)
├── figures/               ← Graphiques et visualisations
├── reports/               ← Rapports et synthèses
└── docs/                  ← Documentation complémentaire
```

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/TSAGUE25/recommendation-system-portfolio.git
cd recommendation-system-portfolio

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate    # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer Jupyter
jupyter notebook
```

---

## Métriques clés (données simulées)

```
Precision@10 = 0.31 | Recall@10 = 0.24 | NDCG@10 = 0.38 (simulés)
```

---

## Valeur métier

Personnalisation des recommandations. Augmentation du taux de conversion simulée.

---

## Limites

SVD sans mise à jour incrémentale. Pas de diversification temporelle.

---

## Prochaines améliorations

ALS (Alternating Least Squares). Neural CF. Déploiement API temps réel.

---

## Avertissement — Confidentialité

> **Toutes les données utilisées dans ce projet sont simulées, synthétiques ou anonymisées.**
> Aucune donnée réelle, confidentielle ou propriétaire n'est présente dans ce dépôt.
> Ce projet est un cas d'usage pédagogique à destination du portfolio professionnel d'Emmanuel TSAGUE.

---

## Contributors

**TSAGUE EMMANUEL** - Data Scientist  
Specialise en Machine Learning, Data Analysis et systemes decisionnels.  
Formation Datascientest 2024 | EDF MAD EDVANCE  
Email : [emmatsague@yahoo.fr](mailto:emmatsague@yahoo.fr)  
LinkedIn : [emmanuel-tsague-114295414](https://www.linkedin.com/in/emmanuel-tsague-114295414)  
GitHub : [github.com/TSAGUE25](https://github.com/TSAGUE25)

