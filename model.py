import torch
import torch.nn as nn

class EarCopyBrain(nn.Module):
    """
    EarCopyAIの心臓部となる、畳み込みニューラルネットワーク（CNN）
    """
    def __init__(self):
        super(EarCopyBrain, self).__init__()
        
        # ① 特徴抽出レイヤー（CQT画像から「倍音」などの波形の形を見つけ出す）
        self.features = nn.Sequential(
            # 1層目: 最初の特徴を見つける
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            # 2層目: さらに複雑な特徴（弦ごとの音色の違いなど）を見つける
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # ② 分類レイヤー（見つけた特徴から、120通りの運指のどれかを当てる）
        self.classifier = nn.Sequential(
            # どんな長さの入力（曲の長さ）が来ても、1つの塊に圧縮する魔法の関数
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(),
            
            # 最終出力: 32個の特徴を、120個（6弦 × 20フレット）の確率に変換する！
            nn.Linear(in_features=32, out_features=120)
        )

    def forward(self, x):
        """
        データが流れるルート（順伝播）
        画像(x) -> 特徴抽出 -> 分類 -> 出力
        """
        x = self.features(x)
        x = self.classifier(x)
        return x