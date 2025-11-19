import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models
from meld_dataset import MELDDataset


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Sử dụng BERT: Mô hình ngôn ngữ cực mạnh của Google
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # 2. Đóng băng (Freeze) BERT
        for param in self.bert.parameters():
            param.requires_grad = False

        # 3. Lớp chiếu (Projection)
        self.projection = nn.Linear(768, 120)

    def forward(self, input_ids, attention_mask):
        # Extract Bert embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # use [CLS] token representation
        pooler_output = outputs.pooler_output

        return self.projection(pooler_output)


class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Sử dụng R3D_18 (ResNet 3D): Mô hình chuyên xử lý video
        self.backbone = vision_models.video.r3d_18(
            weights=vision_models.video.R3D_18_Weights.KINETICS400_V1
        )

        # ... Đóng băng backbone ...
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Thay thế lớp cuối cùng (Head)
        num_fts = self.backbone.fc.in_features
        # define layer trainable
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_fts, 128), nn.ReLU(), nn.Dropout(0.2)
        )

    def forward(self, x):
        # switch frames and channels
        # [batch_size, frames, channels, height, width]-> [batch_size, channels, frames, height, width]
        x = x.transpose(1, 2)
        return self.backbone(x)


class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Lower level features
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Higher level features
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Keep all pretrain param
        for param in self.conv_layers.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2))

    def forward(self, x):
        # [batch_size, 1,64,300]
        x = x.squeeze(1)

        # Features output: [batch_size, 128, 1]

        features = self.conv_layers(x)
        return self.projection(features.squeeze(-1))


# if main run
class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoders
        self.text_encoder = TextEncoder()
        self.video_endcoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        # fusion_layer

        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 3, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3)
        )

        # Classification heads
        self.emo_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7),  # 7 emotions
        )

        self.sentiment_calssifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3),  # 3 sentiment, nagative, positive, neutral
        )

    def forward(self, text_input, video_frames, audio_features):
        text_features = self.text_encoder(
            text_input["input_ids"], text_input["attention_mask"]
        )

        video_features = self.video_endcoder(video_frames)
        audio_features = self.audio_encoder(audio_features)
        # Concatenate multimodal features
        combined_features = torch.cat(
            [text_features, video_features, audio_features], dim=1
        )  # [batch_size, 128 * 3]

        fusion_features = self.fusion_layer(combined_features)

        emotion_output = self.emo_classifier(fusion_features)
        sentiment_output = self.sentiment_calssifier(fusion_features)

        return {"emotion": emotion_output, "sentiment": sentiment_output}


if __name__ == "__main__":
    train_csv = "../dataset/train/train_sent_emo.csv"
    train_video_dir = "../dataset/train/train_splits"

    dataset = MELDDataset(train_csv, train_video_dir)

    sample = dataset[0]
    model = MultimodalSentimentModel()
    model.eval()

    text_inputs = {
        "input_ids": sample["text_inputs"].unsqueeze(0),
        "attention_mask": sample["text_inputs"]["attention_mask"].unsqueeze(0),
    }

    video_frames = sample["video_frames"].unsqueeze(0)
    audio_features = sample["audio_features"].unsqueeze(0)

    with torch.inference_mode():
