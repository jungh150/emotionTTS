import librosa
import numpy as np
import os
import pandas as pd
import torch

def generate_mel_spectrograms(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # 저장 디렉토리 생성

    # CSV 파일 읽기
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        wav_path = row['file_path']
        output_path = os.path.join(output_dir, os.path.basename(wav_path).replace(".wav", ".npy"))

        try:
            # 오디오 파일 로드 및 Mel Spectrum 생성
            y, sr = librosa.load(wav_path, sr=22050)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # .npy 파일로 저장
            np.save(output_path, mel_spec_db)
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")

class MelSpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mel_path = self.data.iloc[idx, 0]
        mel_spectrogram = np.load(mel_path)
        mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32)

        label = int(self.data.iloc[idx, 1])
        label = torch.tensor(label, dtype=torch.long)

        mask = torch.ones(mel_spectrogram.shape[1], dtype=torch.float32)
        return mel_spectrogram, label, mask

def collate_fn(batch):
    mel_spectrograms, labels, masks = zip(*batch)

    max_len = max(mel.shape[1] for mel in mel_spectrograms)
    mel_spectrograms_padded = torch.zeros(len(batch), mel_spectrograms[0].shape[0], max_len)
    masks_padded = torch.zeros(len(batch), max_len)

    for i, mel in enumerate(mel_spectrograms):
        mel_spectrograms_padded[i, :, :mel.shape[1]] = mel
        masks_padded[i, :mel.shape[1]] = 1

    return mel_spectrograms_padded, torch.stack(labels), masks_padded

# 실행 코드
if __name__ == "__main__":
    csv_path = "datasets/emotion_melpath_dataset.csv"
    output_dir = "datasets/melspecs/"
    generate_mel_spectrograms(csv_path, output_dir)