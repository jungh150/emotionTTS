import torch
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import os
from emotionTTS import EmotionTTS

# 경로 설정
MODEL_PATH = "models/emotion_tts_model.pth"
OUTPUT_DIR = "outputs/test_results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 출력 디렉토리 생성

# 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
emotion_embedding_dim = 32
input_dim = 128
output_dim = 128

model = EmotionTTS(emotion_embedding_dim=emotion_embedding_dim, input_dim=input_dim, output_dim=output_dim)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
model.eval()

def visualize_and_save_mel_spectrum(mel, sr, filename, title):
    """Mel Spectrum을 시각화하고 이미지로 저장."""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel, sr=sr, x_axis='time', y_axis='mel', cmap='coolwarm')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved Mel Spectrum Image: {filename}")

def process_audio(input_audio_path, target_emotion_label, output_audio_name):
    # 1. 원래 음성 파일을 Mel Spectrum으로 변환
    y, sr = librosa.load(input_audio_path, sr=22050)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # 2. 모델 입력 준비
    mel_tensor = torch.tensor(mel_spectrogram_db).unsqueeze(0).to(device).float()
    emotion_label = torch.tensor([target_emotion_label]).to(device)

    # 3. 모델 추론
    with torch.no_grad():
        transformed_mel = model(mel_tensor, emotion_label)

    # 4. Mel Spectrum 시각화 및 저장
    input_mel_path = os.path.join(OUTPUT_DIR, output_audio_name.replace(".wav", "_input_mel.png"))
    output_mel_path = os.path.join(OUTPUT_DIR, output_audio_name.replace(".wav", "_output_mel.png"))

    visualize_and_save_mel_spectrum(mel_spectrogram_db, sr, input_mel_path, "Input Mel Spectrum")
    output_mel = transformed_mel[0].cpu().numpy()
    visualize_and_save_mel_spectrum(output_mel, sr, output_mel_path, f"Output Mel Spectrum (Emotion {target_emotion_label})")

    # 5. Output Mel Spectrum -> 음성 변환
    from librosa.core import db_to_power

    # 로그 스케일(dB)을 파워 스케일로 변환
    output_power = db_to_power(output_mel)

    # 파워 스케일 Mel Spectrum -> 음성 변환
    transformed_audio = librosa.feature.inverse.mel_to_audio(output_power, sr=sr, n_iter=64)

    # 6. 볼륨 정규화
    transformed_audio = transformed_audio / np.max(np.abs(transformed_audio))  # Normalize
    transformed_audio = transformed_audio * 0.95  # Scale to 95% of max amplitude

    # 7. 변환된 음성 저장
    output_audio_path = os.path.join(OUTPUT_DIR, output_audio_name)
    sf.write(output_audio_path, transformed_audio, samplerate=sr)
    print(f"Transformed Audio Saved: {output_audio_path}")

    # 8. 로그 출력
    print(f"Input Audio: {input_audio_path}")
    print(f"Target Emotion Label: {target_emotion_label}")
    print(f"Input Mel Spectrum Image: {input_mel_path}")
    print(f"Output Mel Spectrum Image: {output_mel_path}")

# 테스트 예제 실행
process_audio(input_audio_path="datasets/test/audio_segment_0041.wav", 
              target_emotion_label=3,  # 예: 2 = Sad
              output_audio_name="neutral_to_angry.wav")