from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer
import cv2
import numpy as np
import torch

csv_path = "../dataset/dev/dev_sent_emo.csv"
video_path = "../dataset/dev/dev_splits_complete"


class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dỉr):  # Ham init tron gython
        self.data = pd.read_csv(csv_path)
        self.video_dir = video_dỉr

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

        self.sentiment_map = {"negative": 0, "neutral": 1, "positvie": 2}

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

    def __getitem__(self, index):
        row = self.data.iloc[index]
        video_filename = f"""dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"""

        video_path = Path(self.video_dir) / video_filename

        if not video_path.exists():
            raise FileNotFoundError(f"No video found for filename: {path}")

        text_input = self.tokenizer(
            row["Utterance"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        video_frames = self._load_video_frames(video_path)

        # print(video_frames)


if __name__ == "__main__":
    meld = MELDDataset(csv_path, video_path)
    print(meld[0])
