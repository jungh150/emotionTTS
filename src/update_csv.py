import pandas as pd
import os

# 원본 CSV 파일 경로
input_csv_path = "datasets/emotion_melpath_dataset.csv"

# 수정된 CSV 파일 저장 경로
output_csv_path = "datasets/emotion_melpath_dataset_updated.csv"

# Mel Spectrum 저장 디렉토리
mel_dir = "datasets/melspecs/"

# CSV 파일 읽기
df = pd.read_csv(input_csv_path)

# 파일 경로를 .npy로 변환
def update_to_npy_path(wav_path):
    # .wav 파일 경로를 .npy 파일 경로로 변환
    npy_file_name = os.path.basename(wav_path).replace(".wav", ".npy")
    return os.path.join(mel_dir, npy_file_name)

# DataFrame의 file_path 필드 업데이트
df['file_path'] = df['file_path'].apply(update_to_npy_path)

# 업데이트된 CSV 저장
df.to_csv(output_csv_path, index=False)
print(f"Updated CSV saved to: {output_csv_path}")