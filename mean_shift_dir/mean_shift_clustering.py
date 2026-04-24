import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift, estimate_bandwidth


# --- 1. FONCTION DE CHARGEMENT ---
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


if __name__ == "__main__":
    # Détection automatique du bon chemin
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    base_path = PROJECT_ROOT / "biodcase_development_set"

    print("--- Étape 1 : Chargement des données ---")
    X_data, y_true = load_data_sklearn(base_path, "train", max_images=200)

    print("\n--- Étape 2 : Réduction de dimension (PCA 2D) ---")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_data)

    print("\n--- Étape 3 : Clustering (Mean Shift) ---")
    rayon_estime = estimate_bandwidth(X_pca, quantile=0.1)
    ms = MeanShift(bandwidth=rayon_estime)
    clusters_ms = ms.fit_predict(X_pca)

    n_clusters = len(np.unique(clusters_ms))
    print(f"Mean Shift a trouvé {n_clusters} clusters tout seul !")

    # --- LE CALCUL DU SCORE ARI ---
    ari_ms = adjusted_rand_score(y_true, clusters_ms)
    print(f" Score ARI pour Mean Shift : {ari_ms:.4f}")

    # --- Étape 4 : Visualisation ---
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_ms, cmap='plasma', alpha=0.6)
    plt.title(f"Clustering Mean Shift (Trouvé : {n_clusters} clusters)")
    plt.xlabel("Composante Principale 1")
    plt.ylabel("Composante Principale 2")
    plt.colorbar(scatter, label="Numéro de Cluster")

    output_path = SCRIPT_DIR / "mean_shift_result.png"
    plt.savefig(output_path, dpi=300)
    print(f"Graphique sauvegardé sous : {output_path}")