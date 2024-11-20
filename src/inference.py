import torch
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from train import EmotionTTS
from preprocess import AudioDataset

def transform_audio(input_file, target_emotion, model_path='models/emotion_tts_model.pth'):
    # 모델 로드
    model = EmotionTTS()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 오디오 파일 로드
    y, sr = librosa.load(input_file, sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # 모델을 통한 Mel Spectrum 변형
    mel_tensor = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        emotion_vector = model(mel_tensor)
        # 여기서 emotion_vector를 통해 변형된 Mel Spectrum 생성 (추가 논리 필요)

    # 변형된 Mel Spectrum을 오디오로 변환
    transformed_audio = librosa.feature.inverse.mel_to_audio(mel_db, sr=sr)
    librosa.output.write_wav('outputs/transformed.wav', transformed_audio, sr)
    print(f"Transformed audio saved at outputs/transformed.wav")

# 예시 실행
transform_audio(input_file='data/neutral/example.wav', target_emotion='happy')