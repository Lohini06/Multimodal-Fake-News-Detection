import torch
import torch.nn as nn
import open_clip

class MultimodalClassifier(nn.Module):
    def __init__(self, config: dict):
        super(MultimodalClassifier, self).__init__()

        # Load pretrained CLIP
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            config["model"]["clip_model"],
            pretrained=config["model"]["clip_pretrained"]
        )
        self.tokenizer = open_clip.get_tokenizer(config["model"]["clip_model"])

        # Freeze CLIP weights — we only train the classifier head
        for param in self.clip_model.parameters():
            param.requires_grad = False

        embedding_dim = config["model"]["embedding_dim"]
        hidden_dim = config["model"]["hidden_dim"]
        dropout = config["model"]["dropout"]
        num_classes = config["model"]["num_classes"]

        # Fusion MLP: takes concatenated image + text embeddings
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
        # L2 normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.float()

    def encode_text(self, texts: list) -> torch.Tensor:
        tokens = self.tokenizer(texts)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens)
        # L2 normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.float()

    def forward(self, images: torch.Tensor, texts: list) -> torch.Tensor:
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)

        # Concatenate both modalities
        fused = torch.cat([image_features, text_features], dim=-1)

        # Classify
        logits = self.classifier(fused)
        return logits

    def get_embeddings(self, images: torch.Tensor, texts: list) -> torch.Tensor:
        """Returns fused embeddings — useful for SHAP explainability."""
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)
        return torch.cat([image_features, text_features], dim=-1)