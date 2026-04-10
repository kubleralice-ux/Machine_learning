import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import os
from datetime import datetime, timezone
import librosa
import soundfile as sf

def lire_csv(fichier):
    df = pd.read_csv(fichier, sep=",")
    return df.to_numpy()

def extraire_donnees_interet(array):
    L = []
    L.append(array[0])
    L.append(array[1])
    L.append(datetime.fromisoformat(array[-2]))
    L.append(datetime.fromisoformat(array[-1]))
    # L = np.array(L)
    return L

def extraire_pour_un_excel(data):
    donnees_interet = []
    for k in range(len(data)):
        donnees_interet.append(extraire_donnees_interet(data[k]))
    # return np.array(donnees_interet)
    return donnees_interet
    
def extract(audio_path, start_iso, end_iso, out_path):
    filename = os.path.basename(audio_path)
    base = filename.replace(".wav", "")
    date_part, time_part = base.split("T")
    time_part = time_part.replace("-", ":", 2)
    base = f"{date_part}T{time_part}"
    audio_start = datetime.strptime(base, "%Y-%m-%dT%H:%M:%S_%f")
    audio_start = audio_start.replace(tzinfo=timezone.utc)
    start_dt = start_iso
    end_dt   = end_iso
    offset_start = (start_dt - audio_start).total_seconds()
    offset_end   = (end_dt - audio_start).total_seconds()
    if offset_end <= 0:
        raise ValueError("Le segment demandé est entièrement avant le début de l'audio.")
    y, sr = librosa.load(audio_path, sr=None)
    start_sample = max(0, int(offset_start * sr))
    end_sample   = min(len(y), int(offset_end * sr))
    segment = y[start_sample:end_sample]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, segment, sr)
    return segment

def nmbr_cri_par_audio(data):
    dico = {}

    for ligne in data:
        audio = ligne[1]
        if audio not in dico:
            dico[audio] = 1
        else:
            dico[audio] += 1

    return dico


def retrouve_chemin_audio(folder_name, file_name):
    return os.path.join("biodcase_development_set", "train", "audio", folder_name, file_name)

def new_file_name(file_name, folder_name):
    return os.path.join("biodcase_development_set", "train", "audio_extraits", folder_name, file_name)

def chercher_tous_les_excel(folder_names):
    liste_excel = []
    for folder_name in folder_names :
        folder_name_csv = f"{folder_name}.csv"
        path = os.path.join("biodcase_development_set", "train", "annotations", folder_name_csv)
        excel = lire_csv(path)
        liste_excel.append(excel)
    # array_excel = np.array(liste_excel)
    return liste_excel

def extraire_donnees_interet_boucle(data):
    """ data  : liste de taille 8 (8 classes dans train)
    chaque item contient beaucoup de lignes (exemple : 2222)
    chaque ligne a une longueur de 8"""
    N = len(data)
    new_data = []
    for k in range(N):
        data_interet = extraire_pour_un_excel(data[k])
        new_data.append(data_interet)
    return new_data

def liste_dico(data):
    L = []
    for k in range(len(data)):
        L.append(nmbr_cri_par_audio(data[k]))
    return L


def new_name_added(data):
    data = sorted(data, key=lambda x: (x[1], x[2]))
    numero = 0
    ancien_filename = None
    for ligne in data:
        file_name = ligne[1]
        if file_name == ancien_filename:
            numero += 1
        else:
            numero = 1
        base = file_name.replace(".wav", "")
        newfile_name = f"{base}_{numero}.wav"
        ligne.append(newfile_name)
        ancien_filename = file_name
    return data

def new_name_boucle(data):
    n = len(data)
    for k in range(n):
        data[k] = new_name_added(data[k])

def extraire_tous_les_audios( data):
    N = len(data) #nombre de classes
    Ns = []
    for k in range(N):
        n = len(data[k]) #nombre de son de baleine dans chaque excel
        for i in range(n):
            excel = data[k][i]
            folder_name, file_name, start, end, new_name =excel[0], excel[1],excel[2], excel[3], excel[4]
            audio_path = retrouve_chemin_audio(folder_name, file_name)
            new_audio_path = new_file_name(new_name, folder_name)
            extract(audio_path, start, end, new_audio_path)

if __name__ == "__main__":
    nom_classes = ["ballenyislands2015","casey2014","elephantisland2013","elephantisland2014","greenwich2015","kerguelen2005","maudrise2014","rosssea2014"]
    N = len(nom_classes) # nombre de bases de données dans "train"
    liste_data = chercher_tous_les_excel(nom_classes)
    donnees_interet = extraire_donnees_interet_boucle(liste_data)
    # print(donnees_interet[0][:6])
    dico = liste_dico(donnees_interet)
    new_name_boucle(donnees_interet)
    extraire_tous_les_audios(donnees_interet)



