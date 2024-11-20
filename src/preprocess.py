import librosa
import numpy as np
import os
import torch
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, data_dir, sample_rate=22050, n_mels=128, max_len=128):
        self.data = []
        self.labels = []
        self.sr = sample_rate
        self.n_mels = n_mels
        self.max_len = max_len

        # 감정별 오디오 파일 로드
        emotions = {'angry': 0, 'happy': 1, 'sad': 2, 'neutral': 3}
        for emotion, label in emotions.items():
            folder_path = os.path.join(data_dir, emotion)
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                self.data.append(file_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        label = self.labels[idx]

        # 오디오 파일 로드
        y, sr = librosa.load(file_path, sr=self.sr)

        # Mel Spectrum 추출
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # 패딩/트리밍
        if mel_db.shape[1] > self.max_len:
            mel_db = mel_db[:, :self.max_len]
        else:
            pad_width = self.max_len - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')

        return torch.tensor(mel_db, dtype=torch.float32), label