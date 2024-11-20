import torch
import torch.nn as nn

class EmotionTTS(nn.Module):
    def __init__(self, emotion_embedding_dim, input_dim, output_dim):
        """
        감정 정보를 반영하여 Mel Spectrum을 생성하는 TTS 모델.
        
        Parameters:
        - emotion_embedding_dim (int): 감정 임베딩 벡터의 차원.
        - input_dim (int): 입력 Mel Spectrum의 차원.
        - output_dim (int): 출력 Mel Spectrum의 차원.
        """
        super(EmotionTTS, self).__init__()
        
        # 감정 임베딩 레이어 (6개의 감정 레이블)
        self.emotion_embedding = nn.Embedding(6, emotion_embedding_dim)
        
        # Encoder: Mel Spectrum과 감정 임베딩을 처리
        self.encoder = nn.Conv1d(input_dim + emotion_embedding_dim, 512, kernel_size=3, stride=1, padding=1)
        
        # Decoder: 감정 반영 Mel Spectrum 생성
        self.decoder = nn.Conv1d(512, output_dim, kernel_size=3, stride=1, padding=1)
        
        # 활성화 함수
        self.relu = nn.ReLU()
    
    def forward(self, x, emotion_label):
        """
        Forward pass
        
        Parameters:
        - x (Tensor): 입력 Mel Spectrum, shape = (batch_size, input_dim, seq_len)
        - emotion_label (Tensor): 감정 레이블, shape = (batch_size,)
        
        Returns:
        - Tensor: 출력 Mel Spectrum, shape = (batch_size, output_dim, seq_len)
        """
        # 감정 레이블을 임베딩 벡터로 변환
        emotion_embedding = self.emotion_embedding(emotion_label).unsqueeze(2)  # (batch_size, embedding_dim, 1)
        
        # Mel Spectrum에 감정 임베딩 추가
        x = torch.cat((x, emotion_embedding.expand(-1, -1, x.size(2))), dim=1)  # (batch_size, input_dim + embedding_dim, seq_len)
        
        # Encoder: 입력 Mel Spectrum 처리
        encoded = self.relu(self.encoder(x))
        
        # Decoder: 감정 반영 Mel Spectrum 생성
        decoded = self.decoder(encoded)
        
        return decoded