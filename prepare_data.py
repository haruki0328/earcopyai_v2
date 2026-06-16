import jams
import librosa
import numpy as np

def extract_guitarset_data(audio_path, jams_path):
    print(f"🎵 音声を読み込んでいます: {audio_path}")
    # 1. 音声を読み込み、CQTスペクトログラム（AIの目）に変換
    y, sr = librosa.load(audio_path, sr=22050)
    cqt = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('E2'), n_bins=84, bins_per_octave=12))
    
    print(f"📄 正解データ(JAMS)を解剖しています: {jams_path}")
    # 2. JAMSファイルを読み込む
    jam = jams.load(jams_path)
    
    # 3. 弦ごとの「正解データ」を抽出する
    # GuitarSetは 0=1弦, 1=2弦... 5=6弦 としてデータが分かれています
    print("\n【抽出された正解データ（最初の5音だけ表示）】")
    note_count = 0
    
    # JAMSの中にある「MIDIノート（音符）」のデータを弦ごとに探す
    for anno in jam.annotations.search(namespace='note_midi'):
        # どの弦のデータかを取得
        string_idx = anno.annotation_metadata.data_source
        string_num = int(string_idx) + 1  # 1弦〜6弦に変換
        
        # その弦で弾かれた音符を1つずつ取り出す
        for note in anno.data:
            start_time = note.time          # 弾き始めの時間（秒）
            duration = note.duration        # 鳴っている長さ（秒）
            midi_pitch = note.value         # 音の高さ（MIDI番号）
            
            # MIDI番号と弦番号から、フレットを逆算する！
            # 各弦の開放弦のMIDI番号: 1弦=64, 2弦=59, 3弦=55, 4弦=50, 5弦=45, 6弦=40
            open_strings = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}
            fret = round(midi_pitch - open_strings[string_num])
            
            if note_count < 5:
                print(f"時間: {start_time:.2f}秒 ~ | {string_num}弦 {fret}フレット (MIDI: {midi_pitch:.1f})")
                note_count += 1

    print(f"\n✅ 成功！ CQT画像のサイズ: {cqt.shape}")
    print("AIに「この画像（時間）の時は、この弦・フレットが正解だ！」と教える準備が整いました。")

# 実行部分
if __name__ == "__main__":
    # ★ここを先ほど錬成したテストデータの名前に書き換えます！
    audio_file = "dataset/test_guitar.wav"
    jams_file = "dataset/test_annotation.jams"
    
    try:
        extract_guitarset_data(audio_file, jams_file)
    except FileNotFoundError:
        print("エラー: datasetフォルダの中に音声かjamsファイルが見つかりません。")