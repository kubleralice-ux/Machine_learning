import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def apply_padding(base_dir):
    input_dir = Path(base_dir) / "imagettes"
    output_dir = Path(base_dir) / "imagettes_padded"
    
    all_files = list(input_dir.rglob("*.png"))
    if not all_files:
        print("Aucune image trouvée.")
        return

    # Étape 1 : Trouver la taille maximale
    print("Recherche des dimensions maximales...")
    max_w, max_h = 0, 0
    for file in tqdm(all_files, desc="Analyse"):
        with Image.open(file) as img:
            w, h = img.size
            if w > max_w: max_w = w
            if h > max_h: max_h = h
            
    print(f"Taille cible : {max_w}x{max_h} pixels.")

    # Étape 2 : Appliquer le padding (fond noir)
    for file in tqdm(all_files, desc="Padding"):
        with Image.open(file) as img:
            w, h = img.size
            
            # Création d'un fond noir (0)
            new_img = Image.new(img.mode, (max_w, max_h), color=0)
            
            # Calcul pour centrer l'imagette
            offset_x = (max_w - w) // 2
            offset_y = (max_h - h) // 2
            
            # Collage
            new_img.paste(img, (offset_x, offset_y))
            
            # Sauvegarde dans le nouveau dossier
            relative_path = file.relative_to(input_dir)
            save_path = output_dir / relative_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            new_img.save(save_path)

apply_padding("biodcase_development_set")
print("Toutes les images sont maintenant de taille identique.")
