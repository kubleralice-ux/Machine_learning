import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from tqdm import tqdm

def load_data_sklearn(base_dir, split):
    input_dir = Path(base_dir) / "imagettes_padded" / split
    
    X = [] # Contiendra les images aplaties
    y = [] # Contiendra les étiquettes (les noms des classes)
    
    # Parcourir chaque sous-dossier (bma, zcl, etc.)
    classes = [d for d in input_dir.iterdir() if d.is_dir()]
    
    for class_dir in tqdm(classes, desc=f"Chargement {split}"):
        class_name = class_dir.name
        
        for file in class_dir.glob("*.png"):
            with Image.open(file) as img:
                # Convertir l'image en tableau numpy
                img_array = np.array(img)
                # Aplatir l'image (de 2D à 1D)
                img_flat = img_array.flatten()
                
                X.append(img_flat)
                y.append(class_name)
                
    return np.array(X), np.array(y)

# 1. Chargement des données
print("--- Étape 1 : Chargement des données ---")
base_path = "biodcase_development_set"
X_train, y_train = load_data_sklearn(base_path, "train")
X_val, y_val = load_data_sklearn(base_path, "validation")

print(f"Entraînement : {X_train.shape[0]} images de {X_train.shape[1]} pixels.")
print(f"Validation : {X_val.shape[0]} images de {X_val.shape[1]} pixels.")

# 2. Création et entraînement du modèle
print("\n--- Étape 2 : Entraînement du Random Forest ---")
# n_jobs=-1 permet d'utiliser tous les cœurs de ton processeur
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
print("Entraînement terminé !")

# 3. Évaluation du modèle
print("\n--- Étape 3 : Évaluation sur le set de validation ---")
y_pred = clf.predict(X_val)

# Afficher les scores (Précision, Rappel, F1-score)
print(classification_report(y_val, y_pred))

# Afficher la matrice de confusion
print("Génération de la matrice de confusion...")
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(
    y_val, y_pred, 
    cmap="Blues", 
    xticks_rotation='vertical',
    ax=ax
)
plt.title("Matrice de Confusion - Random Forest")
plt.tight_layout()
plt.show()
