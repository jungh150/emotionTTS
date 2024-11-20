import librosa
import numpy as np
import os
import pandas as pd

# CSV 파일 경로
csv_path = "datasets/emotion_melpath_dataset.csv"

# Mel Spectrum 저장 경로
output_dir = "datasets/melspecs/"
os.makedirs(output_dir, exist_ok=True)  # 저장 디렉토리 생성

# CSV 파일 읽기
df = pd.read_csv(csv_path)

for _, row in df.iterrows():
    wav_path = row['file_path']
    output_path = os.path.join(output_dir, os.path.basename(wav_path).replace(".wav", ".npy"))

    # .wav 파일에서 Mel Spectrum 추출
    try:
        y, sr = librosa.load(wav_path, sr=22050)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Mel Spectrum 저장
        np.save(output_path, mel_spec_db)
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")