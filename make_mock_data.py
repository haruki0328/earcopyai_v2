import os
import jams
import soundfile as sf
import numpy as np

print("🎸 テスト用の『神データ（WAVとJAMS）』を錬成中...")

# datasetフォルダを自動作成
os.makedirs('dataset', exist_ok=True)

# 1. 音源ファイル (WAV) の作成（6弦開放 E2 を想定）
sr = 22050
t = np.linspace(0, 2.0, sr * 2)
freq = 82.41  
wave = (np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * freq * 2 * t)) * np.exp(-3 * t) * 0.5
wav_path = 'dataset/test_guitar.wav'
sf.write(wav_path, wave, sr)

# 2. 正解データ (JAMS) の作成
jam = jams.JAMS()

# ★ここを追加：曲全体の長さをメタデータに登録！
jam.file_metadata.duration = 2.0  

ann = jams.Annotation(namespace='note_midi')

# ギターの弦を指定（GuitarSetのルール: 0=1弦, ..., 5=6弦）
ann.annotation_metadata.data_source = '5' 

# 「0秒から2秒間、MIDI番号40(E2)の音が鳴った」という正解データを追加
ann.append(time=0.0, duration=2.0, value=40.0, confidence=1.0)

jam.annotations.append(ann)

jams_path = 'dataset/test_annotation.jams'
jam.save(jams_path)

print(f"✅ 成功: {wav_path} と {jams_path} を作成しました！")