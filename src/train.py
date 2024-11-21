import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocess import MelSpectrogramDataset, collate_fn  # CSV 기반 데이터셋 사용
from emotionTTS import EmotionTTS  # 모델 임포트

# CSV 파일 경로 설정
CSV_PATH = "datasets/emotion_melpath_dataset_updated.csv"
# CSV 구조: [Mel Spectrum .npy 파일 경로, 감정 레이블]

# 데이터셋 및 DataLoader 준비
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file '{CSV_PATH}' does not exist. Please check the path.")

dataset = MelSpectrogramDataset(csv_file=CSV_PATH)  # 데이터셋 클래스
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)  # DataLoader

# 모델 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_channels = 128  # Mel Spectrum의 채널 수
emotion_embedding_dim = 32
output_dim = 128  # 출력 Mel Spectrum 크기

model = EmotionTTS(emotion_embedding_dim=emotion_embedding_dim, input_dim=input_channels, output_dim=output_dim)
model.to(device)

# 손실 함수 및 최적화 함수
criterion = nn.MSELoss()  # Mel Spectrum의 차이를 최소화
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 학습률을 낮게 설정하여 안정적 학습

# 훈련 루프
epochs = 30
for epoch in range(epochs):
    model.train()
    total_train_loss = 0

    for mel_spectrograms, labels, masks in train_loader:  # masks 포함
        mel_spectrograms, labels, masks = mel_spectrograms.to(device), labels.to(device), masks.to(device)

        optimizer.zero_grad()

        # 모델에 입력
        outputs = model(mel_spectrograms, labels)  # 감정 레이블과 함께 모델에 전달

        # 출력 크기를 입력 크기에 맞게 조정
        mel_spectrograms_resized = F.interpolate(mel_spectrograms, size=outputs.size(2), mode='linear', align_corners=False)

        # 손실 계산 (마스크를 적용하여 패딩된 부분 무시)
        loss = criterion(outputs, mel_spectrograms_resized)
        masked_loss = (loss * masks).sum() / masks.sum()  # 마스크 적용
        total_train_loss += masked_loss.item()

        # 역전파 및 최적화
        masked_loss.backward()
        optimizer.step()

    # 에포크별 손실 출력
    print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {total_train_loss / len(train_loader)}")

# 모델 저장
MODEL_PATH = "models/emotion_tts_model.pth"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)  # 경로 생성
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")