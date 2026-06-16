import numpy as np

print("🎸 物理法則アルゴリズム（ビタビ補正）のテスト\n")

# 運指の選択肢 [弦, フレット]
labels = ["1弦5F (人差し指)", "1弦6F (中指)", "6弦20F (罠)"]
positions = [[1, 5], [1, 6], [6, 20]]

# ---------------------------------------------------------
# ① AIが弾き出した「3フレーム分の確率スコア」
# ---------------------------------------------------------
# フレーム0: 1弦5Fがダントツで正解っぽい
# フレーム1: 6弦20Fが一番スコアが高い（AIが罠に引っかかった！）
# フレーム2: 再び1弦5Fがダントツ
ai_scores = np.array([
    [10.0,  1.0,   0.0],  # フレーム0
    [ 2.0,  8.0,   9.0],  # フレーム1
    [10.0,  1.0,   0.0],  # フレーム2
])

print("❌ 生のAI出力（確率スコアだけを信じた場合）:")
raw_path = np.argmax(ai_scores, axis=1)
print(" -> ".join([labels[i] for i in raw_path]))
print(" ⚠️ 警告: 指が瞬間移動しないと弾けません！\n")

# ---------------------------------------------------------
# ② 物理法則（指の疲れ）を計算して補正する！
# ---------------------------------------------------------
print("🔄 物理法則アルゴリズムで補正中...\n")

best_score = -9999
best_path = []

# 今回は分かりやすく、3フレーム×3候補の全ルート（27通り）を計算して比較します
for i in range(3):
    for j in range(3):
        for k in range(3):
            # 1. AIの自信度（スコアの合計）
            ai_total = ai_scores[0][i] + ai_scores[1][j] + ai_scores[2][k]
            
            # 2. 移動ペナルティ（弦の差 × 5 ＋ フレットの差）
            # 遠くに移動するほどペナルティが大きくなる！
            penalty1 = abs(positions[i][0] - positions[j][0])*5 + abs(positions[i][1] - positions[j][1])
            penalty2 = abs(positions[j][0] - positions[k][0])*5 + abs(positions[j][1] - positions[k][1])
            
            # 3. 最終評価 ＝ AIの自信度 － 指の疲れ（ペナルティ）
            final_score = ai_total - (penalty1 + penalty2)
            
            # 一番「現実的でスコアが高い」ルートを記録
            if final_score > best_score:
                best_score = final_score
                best_path = [i, j, k]

print("✅ 補正後の出力（AIの自信度 － 指の移動コスト）:")
print(" -> ".join([labels[i] for i in best_path]))
print(" ✨ 成功: 人間が自然に弾ける運指になりました！")