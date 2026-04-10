import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from tqdm.auto import tqdm

from config import modeles_classification, base_path

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

def evaluer_et_sauvegarder(modele, nom_modele, X_val, y_val):
    print(f"\n=== Évaluation : {nom_modele} ===")
    y_pred = modele.predict(X_val)

    rapport = classification_report(y_val, y_pred)
    print(rapport)

    nom_fichier = nom_modele.replace(" ", "_").lower()

    with open(f"results/performances_{nom_fichier}.txt", "w", encoding="utf-8") as f:
        f.write(f"=== Rapport de Classification : {nom_modele} ===\n\n")
        f.write(rapport)

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(y_val, y_pred, cmap="Blues", xticks_rotation='vertical', ax=ax)
    plt.title(f"Matrice de Confusion : {nom_modele}")
    plt.tight_layout()
    plt.savefig(f"results/matrice_{nom_fichier}.png", dpi=300)
    plt.close()


# =========================
# === Boucle Principale ===
# =========================

if __name__ == "__main__":
    print("=== Chargement des données ===")
    X_train, y_train = load_data_sklearn(base_path, "train", max_images=None)
    X_val, y_val = load_data_sklearn(base_path, "validation", max_images=None)

    print(f"Entraînement : {X_train.shape[0]} images ({X_train.shape[1]} pixels/image)")
    print(f"Validation : {X_val.shape[0]} images ({X_val.shape[1]} pixels/image)")

    print("\n=== Entrainement et évaluation des modèles ===")
    for nom, modele in modeles_classification.items():
        print(f"\nEntraînement de {nom}...")
        modele.fit(X_train, y_train)
        evaluer_et_sauvegarder(modele, nom, X_val, y_val)

