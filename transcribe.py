import torch
import librosa
import numpy as np
from model import EarCopyBrain

print("🎸 EarCopyAI V2: 自動採譜システムを起動します...\n")

# ---------------------------------------------------------
# 1. 音源の読み込みとAIの目（CQT）の準備
# ---------------------------------------------------------
audio_path = "dataset/test_guitar.wav"
print(f"🎵 対象ファイル: {audio_path} を解析中...")

y, sr = librosa.load(audio_path, sr=22050)
cqt = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('E2'), n_bins=84, bins_per_octave=12))

# AIが処理しやすいようにテンソルに変換 [バッチ(1), チャンネル(1), 高さ(84), 幅(時間)]
cqt_tensor = torch.tensor(cqt, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# ---------------------------------------------------------
# 2. AIの脳みそ（推論モード）を起動
# ---------------------------------------------------------
model = EarCopyBrain()
# ★超重要: 学習ではなく「推論（テスト）モード」に切り替えるおまじない
model.eval() 

print("🧠 AIが波形を視覚的に読み取っています...")

# 推論時はメモリ節約と速度向上のために、AIの「学習用メモ機能」をオフにします
with torch.no_grad(): 
    ai_output = model(cqt_tensor)
    
    # AIの出力スコアを「0〜100%の確率」に変換する
    probabilities = torch.softmax(ai_output, dim=1)[0]

# ---------------------------------------------------------
# 3. AIの出した答え（Top3）を発表！
# ---------------------------------------------------------
# 確率が最も高い上位3つのインデックス（0〜119）を取得
top3_probs, top3_indices = torch.topk(probabilities, 3)

print("\n✨ 【AIの解析結果】")
print("AIが導き出した、この音の運指候補Top3です:\n")

for i in range(3):
    idx = top3_indices[i].item()
    # 0〜119のインデックスを「弦」と「フレット」に逆算
    string_num = (idx // 20) + 1
    fret_num = idx % 20
    prob_percent = top3_probs[i].item() * 100
    
    print(f"  🏆 候補{i+1}: 【 {string_num}弦 {fret_num}フレット 】 (AIの確信度: {prob_percent:.2f}%)")