import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, adjusted_rand_score
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, MeanShift
from tqdm.auto import tqdm


# =============================
# ===       CONFIG          ===
# =============================

modeles_clustering = {
    "K-Means (3 clusters)": KMeans(n_clusters=3, random_state=42, n_init='auto'),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    "HDBSCAN": HDBSCAN(min_cluster_size=5),
    "Mean Shift": MeanShift(n_jobs=-1)
}

base_path = "biodcase_development_set"

# =============================
# === Fonctions utilitaires ===
# =============================

def load_data_sklearn(base_dir, split, max_images=None):
    input_dir = Path(base_dir) / "imagettes_padded" / split
    X, y = [], []

    classes = [d for d in input_dir.iterdir() if d.is_dir()]

    for class_dir in tqdm(classes, desc=f"Chargement {split}"):
        class_name = class_dir.name
        files = list(class_dir.glob("*.png"))

        if max_images:
            files = files[:max_images]

        for file in files:
            with Image.open(file) as img:
                img_resized = img.convert('L').resize((64, 64), Image.Resampling.BILINEAR)
                X.append(np.array(img_resized).flatten())
                y.append(class_name)

    return np.array(X), np.array(y)


# =========================
# === Boucle Principale ===
# =========================

if __name__ == "__main__":
    print("=== Chargement des données ===")
    X_train, y_train = load_data_sklearn(base_path, "train", max_images=100) # On a pris moins pour l'instant parceuqe c'est vraiment long

    print(f"Entraînement : {X_train.shape[0]} images ({X_train.shape[1]} pixels/image)")

    print("\n=== Lancement des méthodes de Clustering ===")
    with open("results/comparatif_clustering.txt", "w", encoding="utf-8") as f:
        f.write("=== Comparatif des modèles de Clustering ===\n\n")

    for nom, modele in modeles_clustering.items():
        print(f"\nEntraînement de {nom}...")
        modele.fit(X_train)
        labels = modele.labels_

        nb_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        score_ari = adjusted_rand_score(y_train, labels)

        texte_resultat = f"Modèle : {nom} | Clusters trouvés : {nb_clusters} | Score ARI : {score_ari:.4f}\n"
        print(texte_resultat.strip())

        with open("results/comparatif_clustering.txt", "a", encoding="utf-8") as f:
            f.write(texte_resultat)
