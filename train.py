import torch
import torch.nn as nn
import torch.optim as optim
from model import EarCopyBrain

print("🎓 AIの学習（トレーニング）を開始します...\n")

# 1. モデル（脳みそ）を呼び出す
model = EarCopyBrain()

# 2. 誤差を計算する関数（CrossEntropy）と、脳をアップデートする最適化ツール（Adam）を準備
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01) # lrは学習スピード

# 3. テスト用のデータを準備
# 入力: ダミーのCQT画像 [1枚, 白黒, 高さ84, 幅100]
dummy_cqt = torch.randn(1, 1, 84, 100)

# 正解ラベル: 「6弦の0フレット（開放弦）」を正解とする
# ※ 120通りのインデックス計算: (弦番号 - 1) * 20 + フレット
# 6弦0フレットは (6-1)*20 + 0 = 100番目のインデックス
target_label = torch.tensor([100])

print("【目標】AIに「この画像が来たら『100番（6弦開放）』と答えろ！」と教え込みます。\n")

# 4. 学習ループ（同じ問題を50回解かせて、反省させる）
for epoch in range(1, 51):
    # ① まずは予測してみる
    optimizer.zero_grad()
    predictions = model(dummy_cqt)
    
    # ② 正解との「ズレ（Loss）」を計算する
    loss = criterion(predictions, target_label)
    
    # ③ ズレの原因を逆算して、脳みそをアップデート！（バックプロパゲーション）
    loss.backward()
    optimizer.step()
    
    # 10回ごとに成長の記録を表示
    if epoch % 10 == 0 or epoch == 1:
        # 現在のAIが、100番（6弦開放）に対してどれくらいの確信度を持っているか計算
        probabilities = torch.softmax(predictions, dim=1)
        confidence = probabilities[0][100].item() * 100
        
        print(f"学習 {epoch:2d}回目 | ズレ(Loss): {loss.item():.4f} | 6弦開放と答える確率: {confidence:.2f} %")

print("\n✅ 学習完了！AIが完全に特徴を覚えました。")