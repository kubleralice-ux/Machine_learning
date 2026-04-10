import zipfile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from tqdm.auto import tqdm

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
    base_path = "biodcase_development_set"
    
    print("--- Étape 1 : Chargement des données ---")
    X_train, y_train = load_data_sklearn(base_path, "train", max_images=None)
    X_val, y_val = load_data_sklearn(base_path, "validation", max_images=None)

    print(f"Entraînement : {X_train.shape[0]} images ({X_train.shape[1]} pixels/image)")
    print(f"Validation : {X_val.shape[0]} images ({X_val.shape[1]} pixels/image)")

    print("\n--- Étape 2 : Entraînement du Random Forest ---")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    print("Entraînement terminé !")

    print("\n--- Étape 3 : Évaluation ---")
    y_pred = clf.predict(X_val)

    print(classification_report(y_val, y_pred))

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_val, y_pred, 
        cmap="Blues", 
        xticks_rotation='vertical',
        ax=ax
    )
    plt.title("Matrice de Confusion: Random Forest")
    plt.tight_layout()
    plt.savefig("confusion_matrix/random_forest.png", dpi=300)
    print("Matrice enregistrée sous le nom 'random_forest.png'")
