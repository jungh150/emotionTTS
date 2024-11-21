import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from emotionTTS import EmotionTTS

# 감정 레이블
emotion_labels = ["neutral", "happy", "sad", "angry", "fear", "disgust"]

def evaluate_emotion_transform(input_file, model_path='models/emotion_tts_model.pth'):
    # 모델 로드
    emotion_embedding_dim = 32  # 학습 시 설정한 값
    input_dim = 128
    output_dim = 128
    model = EmotionTTS(emotion_embedding_dim=emotion_embedding_dim, input_dim=input_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 오디오 파일 로드
    y, sr = librosa.load(input_file, sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # 입력 데이터 준비
    mel_tensor = torch.tensor(mel_db).unsqueeze(0).float()  # (1, input_dim, seq_len)

    # 모든 감정에 대해 변환
    results = {}
    for i, emotion in enumerate(emotion_labels):
        emotion_label = torch.tensor([i])  # 감정 레이블
        with torch.no_grad():
            # 모델에 입력 전달
            transformed = model(mel_tensor, emotion_label).squeeze(0).numpy()

        # 변환 결과 저장
        results[emotion] = transformed

        # 차이를 계산 (MSE와 코사인 유사도)
        mse = np.mean((mel_db - transformed) ** 2)
        cosine_similarity = np.dot(mel_db.flatten(), transformed.flatten()) / (
            np.linalg.norm(mel_db.flatten()) * np.linalg.norm(transformed.flatten())
        )
        print(f"Emotion: {emotion}, MSE: {mse:.4f}, Cosine Similarity: {cosine_similarity:.4f}")

        # 시각화
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", cmap="coolwarm")
        plt.title("Original")
        plt.colorbar()
        plt.subplot(1, 2, 2)
        librosa.display.specshow(transformed, sr=sr, x_axis="time", y_axis="mel", cmap="coolwarm")
        plt.title(f"Transformed ({emotion})")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    return results

# 테스트 실행
input_file = "datasets/test/audio_segment_0041.wav"
evaluate_emotion_transform(input_file)