import os
import shutil
import librosa
from torch.utils.data import Dataset, DataLoader, dataloader
from pathlib import Path
import pandas as pd
import torchaudio
from transformers import AutoTokenizer
import cv2
import numpy as np
import torch
import subprocess

dev_csv = "../dataset/dev/dev_sent_emo.csv"
dev_video_dir = "../dataset/dev/dev_splits_complete"
train_csv = "../dataset/train/train_sent_emo.csv"
train_video_dir = "../dataset/train/train_splits"
test_csv = "../dataset/test/test_sent_emo.csv"
test_video_dir = "../dataset/test/output_repeated_splits_test"


class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir):  # Ham init tron gython
        self.data = pd.read_csv(csv_path)
        self.video_dir = video_dir

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        # Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear
        self.emotion_map = {
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "joy": 3,
            "neutral": 4,
            "sadness": 5,
            "surprise": 6,
        }

        self.sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}

    def __len__(self):
        return len(self.data)

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            if not cap.isOpened():
                raise ValueError(f"Video not found: {video_path}")

            # Try and read first frame to validate video
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Video not found: {video_path}")

            # reset index to not skip frist frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # resize the frame
                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0
                frames.append(frame)

        except Exception as e:
            raise ValueError(f"Video error: {str(e)}")
        finally:
            cap.release()

        if len(frames) == 0:
            raise ValueError("No frames could be extracted!")

        # pad or truncate framse
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]

        # before permute : [frames, height, width, channels]
        # after permut : [frames, channels, height, width]

        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)

    def _extract_audio_features(self, video_path):
        # chu y type cua python. video_path khong phai string
        audio_path = str(video_path).replace(".mp4", ".wav")

        # kiểm tra ffmpeg tồn tại
        if shutil.which("ffmpeg") is None:
            raise EnvironmentError(
                "ffmpeg not found in PATH. Install ffmpeg to extract audio."
            )

        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vn",  # no video
            "-acodec",
            "pcm_s16le",  # 16-bit PCM
            "-ar",
            str(16000),  # sample rate, ví dụ 16000
            "-ac",
            str(1),  # number of audio channels
            audio_path,
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # --- BẮT ĐẦU PHẦN THAY THẾ BẰNG LIBROSA ---

            # 2. Load bằng librosa
            # librosa.load có thể tự động resample về sr=16000 và chuyển sang mono
            waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)

            # 3. Convert mel-spectrogram
            # Các tham số tương đương với code cũ của bạn
            mel_spec_np = librosa.feature.melspectrogram(
                y=waveform,
                sr=sample_rate,  # đã là 16000
                n_mels=64,
                n_fft=1024,
                hop_length=512,
            )

            # Chuyển sang thang đo Decibel (dB) - đây là thực hành phổ biến
            mel_spec_np = librosa.power_to_db(mel_spec_np, ref=np.max)

            # 4. Trả torch tensor vào DataLoader
            mel_spec = torch.from_numpy(mel_spec_np).float()

            # Thêm chiều "channel" (từ [64, time] -> [1, 64, time])
            # để tương thích với logic padding/normalize bên dưới
            mel_spec = mel_spec.unsqueeze(0)

            # --- KẾT THÚC PHẦN THAY THẾ BẰNG LIBROSA ---
            # Normalize

            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()
            # neu do dai am thanh < 300 => padding
            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))

            else:
                mel_spec = mel_spec[:, :, :300]

            return mel_spec

        except subprocess.CalledProcessError as e:
            raise ValueError(f"Audio extraction error: {str(e)}")

        except Exception as e:
            raise ValueError(f"Audio error: {str(e)}")

        finally:
            # delete audiofile after extract
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def __getitem__(self, index):
        # Check index là instance của Tensor
        if isinstance(index, torch.Tensor):
            index = index.item()

        row = self.data.iloc[index]

        try:
            video_filename = f"""dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"""
            video_path = Path(self.video_dir) / video_filename

            if not video_path.exists():
                raise FileNotFoundError(f"No video found for filename: {video_path}")

            text_input = self.tokenizer(
                row["Utterance"],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )

            video_frames = self._load_video_frames(video_path)
            audio_features = self._extract_audio_features(video_path)
            # Map sentiment and emotion labels
            emotion_label = self.emotion_map[row["Emotion"].lower()]
            sentiment_label = self.sentiment_map[row["Sentiment"].lower()]

            return {
                "text_inputs": {
                    "input_ids": text_input["input_ids"].squeeze(),
                    "attention_mask": text_input["attention_mask"].squeeze(),
                },
                "video_frames": video_frames,
                "audio_features": audio_features,
                "emotion_label": torch.tensor(emotion_label),
                "sentiment_label": torch.tensor(sentiment_label),
            }

        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            return None


def collate_fn(batch):
    # Filter out None samples
    batch = list(filter(None, batch))
    return dataloader.default_collate(batch)


def prepare_dataloaders(
    train_csv,
    train_video_dir,
    dev_csv,
    dev_video_dir,
    test_csv,
    test_video_dir,
    batch_size=32,
):
    train_dataset = MELDDataset(train_csv, train_video_dir)
    dev_dataset = MELDDataset(dev_csv, dev_video_dir)
    test_dataset = MELDDataset(test_csv, test_video_dir)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    return train_loader, dev_loader, test_loader


if __name__ == "__main__":
    # get the loader
    train_loader, dev_loader, test_loader = prepare_dataloaders(
        train_csv,
        train_video_dir,
        dev_csv,
        dev_video_dir,
        test_csv,
        test_video_dir,
    )

    for batch in train_loader:
        print(batch["text_inputs"])
        print(batch["video_frames"].shape)
        print(batch["audio_features"].shape)
        print(batch["emotion_label"].shape)
        print(batch["sentiment_label"].shape)
        break
