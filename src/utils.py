import librosa

def load_audio(file_path, sr=22050):
    y, _ = librosa.load(file_path, sr=sr)
    return y

def save_audio(file_path, y, sr=22050):
    librosa.output.write_wav(file_path, y, sr)