import os
import pandas as pd

# 감정 레이블 매핑
emotion_mapping = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "angry": 3,
    "fear": 4,
    "disgust": 5
}

# 데이터셋 경로 설정
ravdess_path = "datasets/RAVDESS/"
tess_path = "datasets/TESS/"
cremad_path = "datasets/CREMA-D/"

# 결과 저장 리스트
result = []

# RAVDESS 처리
for root, _, files in os.walk(ravdess_path):
    for file in files:
        if file.endswith(".wav"):
            # 파일 이름에서 감정 번호 추출 (예: "03-01-01-01-01-01-01.wav")
            parts = file.split("-")
            if len(parts) > 2:
                emotion_code = parts[2]
                if emotion_code == "01":
                    result.append((os.path.join(root, file), emotion_mapping["neutral"]))
                elif emotion_code == "03":
                    result.append((os.path.join(root, file), emotion_mapping["happy"]))
                elif emotion_code == "04":
                    result.append((os.path.join(root, file), emotion_mapping["sad"]))
                elif emotion_code == "05":
                    result.append((os.path.join(root, file), emotion_mapping["angry"]))
                elif emotion_code == "06":
                    result.append((os.path.join(root, file), emotion_mapping["fear"]))
                elif emotion_code == "07":
                    result.append((os.path.join(root, file), emotion_mapping["disgust"]))

# TESS 처리
for root, _, files in os.walk(tess_path):
    for file in files:
        if file.endswith(".wav"):
            # 파일 경로에서 감정 추출 (예: "OAF_angry/OAF_back_angry.wav")
            emotion = root.split("/")[-1].split("_")[-1].lower()
            if emotion in emotion_mapping:
                result.append((os.path.join(root, file), emotion_mapping[emotion]))

# CREMA-D 처리
for root, _, files in os.walk(cremad_path):
    for file in files:
        if file.endswith(".wav"):
            # 파일 이름에서 감정 추출 (예: "1001_DFA_ANG_XX.wav")
            parts = file.split("_")
            if len(parts) > 2:
                emotion = parts[2].lower()
                if emotion == "neu":
                    result.append((os.path.join(root, file), emotion_mapping["neutral"]))
                elif emotion == "hap":
                    result.append((os.path.join(root, file), emotion_mapping["happy"]))
                elif emotion == "sad":
                    result.append((os.path.join(root, file), emotion_mapping["sad"]))
                elif emotion == "ang":
                    result.append((os.path.join(root, file), emotion_mapping["angry"]))
                elif emotion == "fea":
                    result.append((os.path.join(root, file), emotion_mapping["fear"]))
                elif emotion == "dis":
                    result.append((os.path.join(root, file), emotion_mapping["disgust"]))

# 결과를 DataFrame으로 변환
df = pd.DataFrame(result, columns=["file_path", "emotion_label"])

# CSV로 저장
output_csv_path = "datasets/emotion_melpath_dataset.csv"
df.to_csv(output_csv_path, index=False)
print(f"CSV 파일이 '{output_csv_path}'에 저장되었습니다.")