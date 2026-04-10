import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, MeanShift



# --- 1. FONCTION DE CHARGEMENT (Identique) ---
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
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    base_path = PROJECT_ROOT / "biodcase_development_set"

    print("--- Étape 1 : Chargement des données ---")
    X_data, y_true = load_data_sklearn(base_path, "train", max_images=200)

    print("\n--- Étape 2 : Réduction de dimension (PCA) ---")
    # On écrase les 4096 pixels en seulement 2 dimensions (X et Y) pour pouvoir faire un graphique
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_data)
    print(f"Les données sont passées de {X_data.shape[1]} à {X_pca.shape[1]} dimensions.")

    print("\n--- Étape 3 : Clustering (K-Means) ---")
    kmeans = KMeans(n_clusters=7, random_state=42, n_init="auto")
    clusters_kmeans = kmeans.fit_predict(X_pca)

    # --- Étape 4 : Visualisation ---
    plt.figure(figsize=(10, 6))

    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_kmeans, cmap='viridis', alpha=0.6)
    plt.title("Clustering K-Means après PCA (2D)")
    plt.xlabel("Composante Principale 1")
    plt.ylabel("Composante Principale 2")
    plt.colorbar(scatter, label="Numéro de Cluster")

    output_path = SCRIPT_DIR / "kmeans_clusters.png"
    plt.savefig(output_path, dpi=300)
    print(f"Graphique de clustering sauvegardé sous : {output_path}")