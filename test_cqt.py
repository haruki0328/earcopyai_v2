import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 1. 音声ファイルの読み込み
# 用意したWAVファイルの名前を指定してください
audio_path = 'test_guitar.wav' 

print(f"音源を読み込んでいます: {audio_path}")
# sr=22050 はサンプリングレート（1秒間に何回音を切り取るか）
y, sr = librosa.load(audio_path, sr=22050)

# 2. CQT（定Q変換）の実行
print("音を画像（CQTスペクトログラム）に変換中...")
# fmin: ギターの最低音(6弦開放=E2)の周波数を基準にする
fmin = librosa.note_to_hz('E2')
# ギターの音階（半音）に合わせて周波数を分解する
C = np.abs(librosa.cqt(y, sr=sr, fmin=fmin, n_bins=84, bins_per_octave=12))

# 3. 人間の耳の感覚に合わせるため、デシベル（dB）表記に変換
C_db = librosa.amplitude_to_db(C, ref=np.max)

# 4. 画像として表示する
plt.figure(figsize=(10, 6))
librosa.display.specshow(C_db, sr=sr, x_axis='time', y_axis='cqt_note', fmin=fmin)
plt.colorbar(format='%+2.0f dB')
plt.title('Guitar Sound - CQT Spectrogram')
plt.tight_layout()

print("画像を描画します！")
plt.show()