import pandas as pd
import os
from pathlib import Path
from PIL import Image
import requests
from tqdm import tqdm

def create_sample_dataset(processed_dir: str):
    """
    Creates a small sample CSV dataset for testing the pipeline
    before plugging in the real FakeNewsNet dataset.
    """
    data = {
        "id": list(range(20)),
        "text": [
            "Scientists discover cure for cancer using AI",
            "Local man wins lottery twice in same week",
            "New climate report warns of rising sea levels",
            "Government announces free education for all",
            "Celebrity found alive after reported death",
            "Stock market hits all time high today",
            "Researchers develop new COVID variant vaccine",
            "City council approves new public transport plan",
            "Aliens spotted near Area 51 says insider",
            "President signs new healthcare reform bill",
            "Doctors warn against new social media diet trend",
            "Tech giant launches revolutionary battery technology",
            "Study shows chocolate prevents heart disease always",
            "NASA confirms water found on Mars surface",
            "Local school wins national science competition",
            "Politician caught accepting bribes on camera",
            "New study links 5G towers to health issues",
            "Olympic athlete breaks 100m world record",
            "Miracle drug cures diabetes overnight claim",
            "University opens new artificial intelligence lab",
        ],
        "label": [1, 1, 0, 0, 1, 0, 0, 0, 1, 0,
                  1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
        "image_path": [""] * 20,
    }
    df = pd.DataFrame(data)
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "sample_data.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Sample dataset saved to {out_path}")
    return df

def load_and_validate(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = ["id", "text", "label"]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
    print(f"✅ Loaded {len(df)} samples. Labels: {df['label'].value_counts().to_dict()}")
    return df