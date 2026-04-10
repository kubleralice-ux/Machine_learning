import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score

''' 1. Chargement des données'''


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

if __name__ == "__main__":
    '''2. Préparation des données'''

    SCRIPT_DIR = Path(__file__).resolve().parent

    # On vérifie si on est déjà dans Machine_learning ou au-dessus
    if SCRIPT_DIR.parent.name == "Machine_learning":
        base_path = SCRIPT_DIR.parent / "biodcase_development_set"
    else:
        # Si on est dans Projet ML, on descend dans Machine_learning
        base_path = SCRIPT_DIR.parent / "Machine_learning" / "biodcase_development_set"

    # Dossier de sortie pour les graphiques (qui se créera dans phase2_classification)
    output_dir = SCRIPT_DIR / "confusion_matrix"
    output_dir.mkdir(exist_ok=True)

    print("--- Étape 1 : Chargement des données ---")
    X_train, y_train = load_data_sklearn(base_path, "train", max_images=None)
    X_val, y_val = load_data_sklearn(base_path, "validation", max_images=None)

    print(f"Entraînement : {X_train.shape[0]} images de {X_train.shape[1]} pixels.")

    ''' 3. Définition des modèles'''
    models = {
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Naive_Bayes": GaussianNB(),
        "SVM_Lineaire": SVC(kernel='linear', random_state=42),
        "Reseau_Neurones_MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    }

    ''' 4. Entrainement et évalutation'''
    results = {}

    for name, model in models.items():
        print(f"\n--- Modèle : {name} ---")
        print("Entraînement en cours...")
        model.fit(X_train, y_train)

        print("Prédiction et Évaluation...")
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        results[name] = acc

        print(f"Précision globale (Accuracy) : {acc:.2%}")

        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay.from_predictions(
            y_val, y_pred,
            cmap="Blues",
            xticks_rotation='vertical',
            ax=ax
        )
        plt.title(f"Matrice de Confusion : {name}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{name}_confusion.png", dpi=300)
        plt.close(fig)

    # --- 5. BILAN ---
    print("\n=== BILAN DES PERFORMANCES ===")
    for name, acc in results.items():
        print(f"{name} : {acc:.2%}")