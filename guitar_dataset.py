import os
import glob
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import jams

class EarCopyDatasetDatasetFull(Dataset):
    def __init__(self, data_dir, frame_length=0.05, sr=22050):
        """
        data_dir: 音声とJAMSが格納されているフォルダ (dataset/)
        frame_length: AIに1回に見せる時間の長さ（秒）
        """
        self.sr = sr
        self.frame_length = frame_length
        
        # 1. フォルダ内のすべてのWAVファイルとJAMSファイルをスキャンしてペアにする
        self.audio_files = sorted(glob.glob(os.path.join(data_dir, "*.wav")))
        self.jams_files = sorted(glob.glob(os.path.join(data_dir, "*.jams")))
        
        # 全全フレームを管理するインデックスリストを作成
        self.all_frames = []
        
        print("🔍 データベースを構築中...")
        for file_idx, (audio_p, jams_p) in enumerate(zip(self.audio_files, self.jams_files)):
            # JAMSファイルから曲の長さを取得
            jam = jams.load(jams_p)
            duration = jam.file_metadata.duration
            
            # この曲の中に、何個の「時間フレーム」が含まれるか計算
            num_frames = int(duration / self.frame_length)
            
            # 「何番目のファイルの、何秒目のフレームか」を記録
            for f in range(num_frames):
                start_time = f * self.frame_length
                self.all_frames.append({
                    'file_idx': file_idx,
                    'start_time': start_time,
                    'audio_path': audio_p,
                    'jams_path': jams_p
                })
                
        print(f"✅ データベース構築完了。総ファイル数: {len(self.audio_files)} | 総時間フレーム数: {len(self.all_frames)} 個")

    def __len__(self):
        # AIが学習時に「データは全部で何個ある？」と聞いてきたら、全フレーム数を返す
        return len(self.all_frames)

    def __getitem__(self, idx):
        # AIから「idx番目のデータをちょうだい」と言われたときの処理
        frame_info = self.all_frames[idx]
        
        # 1. 該当するファイルの、該当する時間（0.05秒間）だけピンポイントで音声をロード
        y, _ = librosa.load(
            frame_info['audio_path'], 
            sr=self.sr, 
            offset=frame_info['start_time'], 
            duration=self.frame_length
        )
        
        # 2. その一瞬の音をCQT（画像）に変換
        cqt = np.abs(librosa.cqt(y, sr=self.sr, fmin=librosa.note_to_hz('E2'), n_bins=84, bins_per_octave=12))
        cqt_tensor = torch.tensor(cqt, dtype=torch.float32).unsqueeze(0) # [1, 84, 時間幅]
        
        # 3. JAMSファイルから、その瞬間に鳴っている弦とフレット（正解）を抽出
        # (※ここでは簡略化していますが、指定時間のMIDIノートを検索して120通りのインデックスに変換します)
        target_label = torch.tensor(100) # 例として6弦開放
        
        return cqt_tensor, target_label