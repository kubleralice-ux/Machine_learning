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

    # ====================================================================
    # ---  INTERPRÉTATION PHYSIQUE DE LA PCA  ---
    # ====================================================================

    print("\n--- Analyse des composantes principales ---")

    # ---------------------------------------------------------
    # MÉTHODE 1 : Les images extrêmes de l'Axe X (Composante 1)
    # ---------------------------------------------------------
    # 1. On trouve l'index du point le plus à gauche et le plus à droite
    index_extreme_gauche = np.argmin(X_pca[:, 0])
    index_extreme_droite = np.argmax(X_pca[:, 0])

    # 2. On récupère les lignes de pixels d'origine (4096 valeurs) et on reforme le carré de 64x64
    img_gauche = X_data[index_extreme_gauche].reshape((64, 64))
    img_droite = X_data[index_extreme_droite].reshape((64, 64))

    # 3. On affiche les deux images côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_gauche, cmap='viridis', origin='lower')
    axes[0].set_title(f"Point extrême GAUCHE\n(Vraie étiquette : {y_true[index_extreme_gauche]})")

    axes[1].imshow(img_droite, cmap='viridis', origin='lower')
    axes[1].set_title(f"Point extrême DROITE\n(Vraie étiquette : {y_true[index_extreme_droite]})")
    plt.suptitle("Ce qui sépare le plus nos données (Axe X)")

    # Sauvegarde
    plt.savefig(SCRIPT_DIR / "pca_extremes.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # MÉTHODE 2 : L'image "Fantôme" (Loadings de la PCA)
    # ---------------------------------------------------------
    # pca.components_ contient les "recettes" de nos axes.
    # La ligne [0] est la recette de la Composante 1. On la reforme en 64x64.
    image_fantome_pc1 = pca.components_[0].reshape((64, 64))

    plt.figure(figsize=(8, 6))
    # On utilise 'coolwarm' : Rouge = pixels très importants en positif, Bleu = pixels très importants en négatif
    plt.imshow(image_fantome_pc1, cmap='coolwarm', origin='lower')
    plt.colorbar(label="Poids (Importance du pixel pour la PCA)")
    plt.title("Image Fantôme (PC1) : Quelles fréquences la machine regarde-t-elle ?")

    # Sauvegarde
    plt.savefig(SCRIPT_DIR / "pca_fantome.png", dpi=300)
    plt.close()

    print("Graphiques d'analyse de la PCA générés : 'pca_extremes.png' et 'pca_fantome.png'")

    # ---------------------------------------------------------
    # MÉTHODE 1 bis : Les images extrêmes de l'Axe Y (Composante 2)
    # ---------------------------------------------------------
    # On regarde la colonne 1 (qui correspond à l'axe Y en Python)
    index_extreme_bas = np.argmin(X_pca[:, 1])
    index_extreme_haut = np.argmax(X_pca[:, 1])

    img_bas = X_data[index_extreme_bas].reshape((64, 64))
    img_haut = X_data[index_extreme_haut].reshape((64, 64))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_bas, cmap='viridis', origin='lower')
    axes[0].set_title(f"Point extrême BAS (Y négatif)\n(Vraie étiquette : {y_true[index_extreme_bas]})")

    axes[1].imshow(img_haut, cmap='viridis', origin='lower')
    axes[1].set_title(f"Point extrême HAUT (Y positif)\n(Vraie étiquette : {y_true[index_extreme_haut]})")
    plt.suptitle("Ce qui sépare le plus nos données (Axe Y)")

    plt.savefig(SCRIPT_DIR / "pca_extremes_Y.png", dpi=300)
    plt.close()

    # ====================================================================
    # ---  VÉRITÉ TERRAIN (GROUND TRUTH) ---
    # ====================================================================
    print("\n--- Étape 5 : Génération de la Vérité Terrain ---")

    # On récupère la liste des noms de classes uniques
    classes_uniques = np.unique(y_true)
    cmap = plt.cm.get_cmap('tab10', len(classes_uniques))

    plt.figure(figsize=(10, 6))

    for i, nom_classe in enumerate(classes_uniques):
        masque = (y_true == nom_classe)
        plt.scatter(X_pca[masque, 0], X_pca[masque, 1],
                    label=nom_classe, color=cmap(i), alpha=0.6)

    plt.title("Vérité Terrain : Les vraies classes biologiques (Après PCA)")
    plt.xlabel("Composante Principale 1 (Continu vs Impulsif)")
    plt.ylabel("Composante Principale 2 (Faible vs Puissant)")

    plt.legend(title="Classes réelles", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Sauvegarde du graphique
    output_path_gt = SCRIPT_DIR / "verite_terrain.png"
    plt.savefig(output_path_gt, dpi=300)
    print(f"Graphique de la vérité terrain sauvegardé sous : {output_path_gt}")