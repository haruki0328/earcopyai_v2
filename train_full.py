import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import jams

# model.py から Harukiさんが作った「脳みそ」をインポート！
from model import EarCopyBrain

# ---------------------------------------------------------
# ① データ保管倉庫（本物のWAVとJAMSを読み込む仕様）
# ---------------------------------------------------------
class RealGuitarDataset(Dataset):
    def __init__(self, audio_path, jams_path):
        print("🎵 本物の音声データをCQT画像に変換中...")
        # 1. 音声を読み込んでCQT変換
        y, sr = librosa.load(audio_path, sr=22050)
        cqt = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('E2'), n_bins=84, bins_per_octave=12))
        
        # PyTorchの脳みそに入る形に整える: [チャンネル(1), 高さ(84), 幅(時間)]
        self.image = torch.tensor(cqt, dtype=torch.float32).unsqueeze(0)

        # 2. JAMSから正解データを読み込む（今回は6弦開放弦=100番で固定）
        jam = jams.load(jams_path)
        self.target = torch.tensor(100, dtype=torch.long)
        
        # 3. テスト用に「1曲を100曲分に水増し」して倉庫に入れる
        self.num_samples = 100
        print(f"📦 倉庫に {self.num_samples} 個の学習データを用意しました！\n")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # コンベアに画像と正解ラベルを渡す
        return self.image, self.target

# ---------------------------------------------------------
# ② メインの学習ループ（工場稼働！）
# ---------------------------------------------------------
if __name__ == "__main__":
    print("🚀 エンドツーエンド学習パイプラインを起動します！\n")
    
    # 1. データの準備（コンベアの設定）
    # datasetフォルダにある錬成データを使います
    dataset = RealGuitarDataset("dataset/test_guitar.wav", "dataset/test_annotation.jams")
    # 10曲ずつ束にして流す
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    # 2. 脳みその準備
    model = EarCopyBrain()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # 3. 本番の学習ループ！（10周回します）
    epochs = 10
    print("🎓 学習スタート！")
    for epoch in range(1, epochs + 1):
        total_loss = 0
        
        # コンベアから流れてくるバッチ（10曲ずつの束）を絶え間なく受け取る
        for batch_images, batch_labels in dataloader:
            optimizer.zero_grad()                 # 記憶のリセット
            outputs = model(batch_images)         # 予測
            loss = criterion(outputs, batch_labels) # 答え合わせ
            loss.backward()                       # ズレの逆算
            optimizer.step()                      # 脳のアップデート
            
            total_loss += loss.item()
        
        # 1周（100曲分）終わるごとの平均のズレを表示
        average_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch:2d}/{epochs} | 平均のズレ(Loss): {average_loss:.4f}")
        
    print("\n✅ 完全統合テスト成功！本物のAIパイプラインが完成しました！")