# Machine_learning - Projet Baleine
Projet de Machine Learning pour classifier des spectrogrammes de vocalisations de baleines d'Antarctique avec scikit-learn.

## Architecture
├── 1-wav_to_imagette.py                        Extrait les imagettes des portions representatives d'un son caracteristique a partir des fichiers audio et des annotations.
├── 2-imagette_padding.py                       Standardise les imagettes avec un fond noir pour uniformiser leur taille.
├── 3-classification.py                         Entraîne et évalue les modèles supervisés (Random Forest, SVM, MLP, Bayes...). Génère les matrices de confusion.
├── 3-clustering.py                             Explore les données via apprentissage non-supervisé (K-Means, DBSCAN, Mean Shift) et calcule le score ARI.
├── config.py                                   Fichier de configuration regroupant les modeles utilises lors de la classification ou des clustering
├── biodcase_development_set/                   Dossier contenant les données brutes et les imagettes.
├── results/
└── README.md

## Dependances
```bash
pip install scikit-learn numpy pillow matplotlib tqdm
```

## Exécution
- Lancer les scripts .py dans l'ordre des numéros.
- Un fois les scripts 1 et 2 lancé les imagettes sont crées et il n'est plus nécéssaire de les relancer.
- Pour choisir les modeles a utiliser pour les test il suffit de modifier le fichier config.py
- Lancer classification.py pour l'évaluation supervisée.
- Lancer clustering.py pour l'analyse des clusters.
