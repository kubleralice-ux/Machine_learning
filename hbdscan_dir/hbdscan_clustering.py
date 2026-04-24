import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from sklearn.cluster import HDBSCAN, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

# --- 1. CHARGEMENT DES DONNEES ---

def load_data_sklearn(base_dir, split, max_images=None):
    input_dir = Path(base_dir) / "imagettes_padded" / split
    X, y = [], []
    classes = [d for d in input_dir.iterdir() if d.is_dir()]

    for class_dir in tqdm(classes, desc=f"Chargement {split}"):
        class_name = class_dir.name
        files = list(class_dir.glob("*.png"))
        if max_images: files = files[:max_images]
        for file in files:
            with Image.open(file) as img:
                img_resized = img.convert('L').resize((64, 64), Image.Resampling.BILINEAR)
                X.append(np.array(img_resized).flatten())
                y.append(class_name)
    return np.array(X), np.array(y)

if __name__ == '__main__':
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    base_path = PROJECT_ROOT / "biodcase_development_set"

    print("--- Étape 1 : Chargement des données ---")
    X_data, y_true = load_data_sklearn(base_path, "train", max_images=200)

    print("\n--- Étape 2 : Réduction de dimension (PCA 2D) ---")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_data)

    print("\n--- Étape 3 : Clustering (HDBSCAN) ---")
    hdb = HDBSCAN(min_cluster_size=10)
    clusters_hdb = hdb.fit_predict(X_pca)

    # CALCUL DU SCORE ARI
    ari_hdb = adjusted_rand_score(y_true, clusters_hdb)
    print(f"Score ARI global pour HDBSCAN : {ari_hdb:.4f}")

    plt.figure(figsize=(10, 6))

    mask_bruit = (clusters_hdb == -1)
    mask_valide = (clusters_hdb != -1)

    plt.scatter(X_pca[mask_bruit, 0], X_pca[mask_bruit, 1], c='gray', marker='x', alpha=0.5, label='Bruit (-1)')
    scatter = plt.scatter(X_pca[mask_valide, 0], X_pca[mask_valide, 1], c=clusters_hdb[mask_valide], cmap='viridis',
                          alpha=0.7)

    plt.title("Clustering HDBSCAN (Les croix grises sont des outliers)")
    plt.xlabel("Composante Principale 1")
    plt.ylabel("Composante Principale 2")
    plt.colorbar(scatter, label="Numéro de Cluster")
    plt.legend()

    output_path = SCRIPT_DIR / "hdbscan_result.png"
    plt.savefig(output_path, dpi=300)
    print(f"Graphique sauvegardé sous : {output_path}")