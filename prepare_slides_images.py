import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import spectrogram
from PIL import Image
import re

# Configuration
OUTPUT_DIR = Path("IMAGES_SLIDES")
RESULTS_DIR = Path("results")
CLUSTERING_DIR = Path("hbdscan_dir")
DATA_DIR = Path("biodcase_development_set")

def setup():
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)
    print(f"Dossier {OUTPUT_DIR} prêt.")

def generate_spectrogram_example():
    print("Génération du spectrogramme pour Slide 2...")
    # Fichier audio trouvé précédemment
    wav_path = DATA_DIR / "train/audio/ballenyislands2015/2015-06-23T13-00-00_000.wav"
    
    if not wav_path.exists():
        print(f"Erreur : Fichier {wav_path} introuvable.")
        return

    fs, x = wavfile.read(wav_path)
    if x.dtype == np.int16: x = x / 32768.
    
    # Paramètres identiques à 1-wav_to_imagette.py
    nperseg = 256
    noverlap = 192
    
    freqs, times, power = spectrogram(x, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap)
    power = np.maximum(power, 1e-20)
    log_psd = 10 * np.log10(power / (2e-5 ** 2))
    
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(times, freqs, log_psd, shading='gouraud', cmap='viridis')
    plt.ylabel('Fréquence [Hz]')
    plt.xlabel('Temps [sec]')
    plt.title('Spectrogramme brut (Fs=250Hz, Antarctique)')
    plt.colorbar(label='Intensité [dB]')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "slide2_spectrogramme.png")
    plt.close()

def generate_padding_comparison():
    print("Génération de la comparaison Padding pour Slide 3...")
    # Imagette trouvée précédemment
    img_path = DATA_DIR / "imagettes/validation/bp20plus/kerguelen2015_2015-04-20T07-00-00_000_5010.png"
    
    if not img_path.exists():
        print(f"Erreur : Imagette {img_path} introuvable.")
        return

    with Image.open(img_path) as img:
        # 1. Redimensionnement brut (étiré)
        img_stretched = img.resize((64, 64), Image.LANCZOS)
        
        # 2. Padding (centré)
        target_size = 64
        w, h = img.size
        # Si l'image est plus grande que 64, on la réduit en gardant l'aspect ratio d'abord ?
        # Dans le projet, max_w et max_h étaient trouvés sur tout le dataset. 
        # Ici on va simuler un padding sur 64x64 pour l'exemple.
        ratio = min(target_size/w, target_size/h)
        new_w, new_h = int(w*ratio), int(h*ratio)
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        
        img_padded = Image.new("L", (target_size, target_size), color=0)
        img_padded.paste(img_resized, ((target_size - new_w) // 2, (target_size - new_h) // 2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img_stretched, cmap='gray')
    ax1.set_title("Étiré (Mauvais : Perte d'aspect ratio)")
    ax1.axis('off')
    
    ax2.imshow(img_padded, cmap='gray')
    ax2.set_title("Padding (Bon : Conservation de la forme)")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "slide3_padding_vs_resize.png")
    plt.close()

def generate_accuracy_plot():
    print("Génération du graphique des performances pour Slide 6...")
    accuracies = {}
    
    for file in RESULTS_DIR.glob("performances_*.txt"):
        model_name = file.stem.replace("performances_", "").replace("_", " ").title()
        with open(file, 'r') as f:
            content = f.read()
            match = re.search(r"accuracy\s+([\d\.]+)", content)
            if match:
                accuracies[model_name] = float(match.group(1))
    
    if not accuracies:
        print("Aucune donnée de performance trouvée.")
        return

    # Tri par accuracy
    sorted_acc = dict(sorted(accuracies.items(), key=lambda item: item[1], reverse=True))
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(sorted_acc)))
    bars = plt.bar(sorted_acc.keys(), sorted_acc.values(), color=colors)
    plt.ylim(0, 1.0)
    plt.ylabel('Accuracy')
    plt.title('Comparaison des performances des modèles de classification')
    plt.xticks(rotation=45, ha='right')
    
    # Ajout des valeurs sur les barres
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "slide6_accuracy_comparison.png")
    plt.close()

def copy_existing_results():
    print("Copie des résultats existants pour Slide 7 et 8...")
    
    # Slide 7 : Matrice SVM
    svm_matrix = RESULTS_DIR / "matrice_svm.png"
    if svm_matrix.exists():
        shutil.copy(svm_matrix, OUTPUT_DIR / "slide7_matrice_svm.png")
    else:
        print("Avertissement : matrice_svm.png introuvable.")

    # Slide 8 : Clustering PCA
    hdbscan_res = CLUSTERING_DIR / "hdbscan_result.png"
    if hdbscan_res.exists():
        shutil.copy(hdbscan_res, OUTPUT_DIR / "slide8_clustering_pca.png")
    else:
        # Fallback sur mean shift
        ms_res = Path("mean_shift_dir/mean_shift_result.png")
        if ms_res.exists():
            shutil.copy(ms_res, OUTPUT_DIR / "slide8_clustering_pca.png")
        else:
            print("Avertissement : graphique de clustering introuvable.")

if __name__ == "__main__":
    setup()
    generate_spectrogram_example()
    generate_padding_comparison()
    generate_accuracy_plot()
    copy_existing_results()
    print("\nToutes les images ont été préparées dans le dossier IMAGES_SLIDES.")
