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
        self.projection = nn.Linear(768, 128)

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

        return {"emotions": emotion_output, "sentiments": sentiment_output}


class MultimodalTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # log dataset sized
        train_size = len(train_loader.dataset())
        val_size = len(val_loader)
        print("\nDataset sizes: ")
        print(f"Training samples: {train_size}")
        print(f"Validation samples: {val_size}")
        print(f"Batches per epoch: {len(train_loader):,}")

        self.optimizer = torch.optim.Adam(
            [
                {"params": model.text_encoder.parameters(), "lr": 8e-6},
                {"params": model.video_encoder.parameters(), "lr": 8e-5},
                {"params": model.audio_encoder.parameters(), "lr": 8e-5},
                {"params": model.fusion_layer.parameters(), "lr": 5e-4},
                {"params": model.emotion_classifier.parameters(), "lr": 5e-4},
                {"params": model.sentiment_classifier.parameters(), "lr": 5e-4},
            ],
            weight_decay=1e-5,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=2
        )

        self.emotion_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.sentiment_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    def train_epoch(self):
        # trong ML: epoch chỉ 1 vòng lặp mà mô hình sẽ duyệt toàn bộ tập huấn luyện để cập nhật trọng số của nó.

        # Cross-entropy: hàm loss
        self.model.train()
        running_loss = {"total": 0, "emotion": 0, "sentiment": 0}

        for batch in self.train_loader:
            device = next(self.model.parameters()).device
            text_inputs = {
                "input_ids": batch["text_input"]["input_ids"].to(device),
                "attention_mask": batch["text_input"]["attention_mask"].to(device),
            }
            video_frames = batch["video_frames"].to(device)
            audio_features = batch["audio_features"].to(device)
            emotion_labels = batch["emotion_label"].to(device)
            sentiment_labels = batch["sentiment_labels"].to(device)

            # Zero gradient
            self.optimizer.zero_grad()

            # Forward paass:
            outputs = self.model(text_inputs, video_frames, audio_features)

            # calculate losses using raw logits
            emotion_loss = self.emotion_criterion(outputs["emotions"], emotion_labels)
            sentiment_loss = self.sentiment_criterion(
                outputs["sentiments"], sentiment_labels
            )

            total_loss = emotion_loss + sentiment_loss

            # backward pass. Calculate gradients
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # track losses
            running_loss["total"] += total_loss.item()
            running_loss["emotion"] += emotion_loss.item()
            running_loss["sentiment"] += sentiment_loss.item()

            # return {k,v} {k,v in running_loss()}

        return {k: v / len(self.train_loader) for k, v in running_loss.item()}

    def validate(self):
        self.model.eval()
        val_loss = {"total": 0, "emotion": 0, "sentiment": 0}
        all_emotion_preds = []
        all_emotion_labels = []
        all_sentiment_preds = []
        all_sentiment_labels = []

        with torch.inference_mode():
            for batch in self.val_loader:
                device = next(self.model.parameters()).device
                text_inputs = {
                    "input_ids": batch["text_input"]["input_ids"].to(device),
                    "attention_mask": batch["text_input"]["attention_mask"].to(device),
                }
                video_frames = batch["video_frames"].to(device)
                audio_features = batch["audio_features"].to(device)
                emotion_labels = batch["emotion_label"].to(device)
                sentiment_labels = batch["sentiment_labels"].to(device)

                # Forward paass:
                outputs = self.model(text_inputs, video_frames, audio_features)

                # calculate losses using raw logits
                emotion_loss = self.emotion_criterion(
                    outputs["emotions"], emotion_labels
                )
                sentiment_loss = self.sentiment_criterion(
                    outputs["sentiments"], sentiment_labels
                )

                total_loss = emotion_loss + sentiment_loss

                all_emotion_preds.extend(
                    outputs["emotions"].argmax(dim=1).cpu().numpy()
                )

                all_emotion_labels.extend(emotion_labels.cpu().numpy())
                all_sentiment_preds.extend(
                    outputs["sentiments"].argmax(dim=1).cpu().numpy()
                )

                all_sentiment_labels.extend(sentiment_labels.cpu().numpy())

                # Track losses
                val_loss["total"] += total_loss.item()
                val_loss["emotion"] += emotion_loss.item()
                val_loss["sentiment"] += sentiment_loss.item()

        avg_loss = {k: v / len(self.val_loader) for k, v in val_loss.item()}


if __name__ == "__main__":
    train_csv = "../dataset/train/train_sent_emo.csv"
    train_video_dir = "../dataset/train/train_splits"

    dataset = MELDDataset(train_csv, train_video_dir)

    sample = dataset[0]
    model = MultimodalSentimentModel()
    model.eval()

    text_inputs = {
        "input_ids": sample["text_inputs"]["input_ids"].unsqueeze(0),
        "attention_mask": sample["text_inputs"]["attention_mask"].unsqueeze(0),
    }

    video_frames = sample["video_frames"].unsqueeze(0)
    audio_features = sample["audio_features"].unsqueeze(0)

    with torch.inference_mode():
        outputs = model(text_inputs, video_frames, audio_features)

        emotion_probs = torch.softmax(outputs["emotions"], dim=1)[0]
        sentiment_probs = torch.softmax(outputs["sentiments"], dim=1)[0]

    emotion_map = {
        0: "anger",
        1: "disgust",
        2: "fear",
        3: "joy",
        4: "neutral",
        5: "sadness",
        6: "surprise",
    }

    sentiment_map = {
        0: "negative",
        1: "neutral",
        2: "positive",
    }

    for i, prob in enumerate(emotion_probs):
        print(f"{emotion_map[i]} : {prob: 2f}")

    for i, prob in enumerate(sentiment_probs):
        print(f"{sentiment_map[i]} : {prob: 2f}")

    print("Predictions for utterance:")
