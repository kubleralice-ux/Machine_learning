import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import spectrogram
from tqdm import tqdm

def extract_imagettes(base_dir):
    base_path = Path(base_dir)
    output_dir = base_path / "imagettes"
    
    for split in ['train', 'validation']:
        audio_dir = base_path / split / 'audio'
        anno_dir = base_path / split / 'annotations'
        
        if not audio_dir.exists() or not anno_dir.exists():
            continue
            
        csv_files = list(anno_dir.glob('*.csv'))
        
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, parse_dates=['start_datetime', 'end_datetime'])
            dataset_name = csv_file.stem
            
            for wav_name, annotations in tqdm(df.groupby('filename'), desc=f"{split} - {dataset_name}"):
                wav_path = audio_dir / dataset_name / wav_name
                if not wav_path.exists():
                    continue
                    
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fs, x = wavfile.read(wav_path)
                    
                if x.dtype == np.uint8: x = (x - 128) / 128.
                elif x.dtype == np.int16: x = x / 32768.
                elif x.dtype == np.int32: x = x / 2147483648.

                # Paramètres de spectrogramme adaptés à 250 Hz
                nperseg = 256
                noverlap = 192
                nfft = 256
                
                freqs, times, power = spectrogram(
                    x, fs=fs, window='hann', 
                    nperseg=nperseg, noverlap=noverlap, nfft=nfft
                )
                
                power = np.maximum(power, 1e-20)
                log_psd = 10 * np.log10(power / (2e-5 ** 2))
                
                wav_start_time = pd.to_datetime(wav_name.split('.')[0], format='%Y-%m-%dT%H-%M-%S_%f').tz_localize('UTC')
                
                for idx, row in annotations.iterrows():
                    start_sec = (row['start_datetime'] - wav_start_time).total_seconds()
                    end_sec = (row['end_datetime'] - wav_start_time).total_seconds()
                    
                    t_mask = (times >= start_sec) & (times <= end_sec)
                    f_mask = (freqs >= row['low_frequency']) & (freqs <= row['high_frequency'])
                    
                    imagette = log_psd[f_mask][:, t_mask]
                    
                    if imagette.size == 0:
                        continue
                        
                    imagette = np.nan_to_num(imagette)
                    if imagette.max() > imagette.min():
                        imagette = (imagette - imagette.min()) / (imagette.max() - imagette.min())
                    else:
                        imagette = np.zeros_like(imagette)
                        
                    class_name = str(row['annotation']).strip()
                    save_dir = output_dir / split / class_name
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    save_path = save_dir / f"{dataset_name}_{wav_name.replace('.wav', '')}_{idx}.png"
                    plt.imsave(save_path, imagette, cmap='Greys', origin='lower')

extract_imagettes("biodcase_development_set")
