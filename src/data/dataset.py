import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import open_clip
import os

class FakeNewsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, preprocess, config: dict):
        self.df = df.reset_index(drop=True)
        self.preprocess = preprocess
        self.max_text_length = config["data"]["max_text_length"]
        self.image_size = config["data"]["image_size"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- Text ---
        text = str(row["text"])

        # --- Image ---
        image_path = row.get("image_path", "")
        if image_path and os.path.exists(str(image_path)):
            image = Image.open(image_path).convert("RGB")
        else:
            # No image available — use blank white image as placeholder
            image = Image.new("RGB", (self.image_size, self.image_size), color=(255, 255, 255))

        image = self.preprocess(image)

        # --- Label ---
        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return image, text, label


def create_dataloaders(df: pd.DataFrame, preprocess, config: dict):
    """Split dataframe into train/val/test and return DataLoaders."""
    from sklearn.model_selection import train_test_split

    val_split = config["training"]["val_split"]
    test_split = config["training"]["test_split"]
    seed = config["training"]["seed"]
    batch_size = config["training"]["batch_size"]

    # First split off test set
    train_val_df, test_df = train_test_split(
        df, test_size=test_split, random_state=seed, stratify=df["label"]
    )

    # Then split train and val
    relative_val = val_split / (1 - test_split)
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val, random_state=seed, stratify=train_val_df["label"]
    )

    print(f"✅ Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_dataset = FakeNewsDataset(train_df, preprocess, config)
    val_dataset   = FakeNewsDataset(val_df,   preprocess, config)
    test_dataset  = FakeNewsDataset(test_df,  preprocess, config)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader