import pandas as pd
import os
from datasets import load_dataset

def load_real_dataset(processed_dir: str):
    """Load real fake news dataset from HuggingFace."""
    print("Loading FakeNews dataset from HuggingFace...")
    ds = load_dataset('GonzaloA/fake_news')

    df = ds['train'].to_pandas()

    df = df[['title', 'label']].copy()
    df = df.rename(columns={'title': 'text'})
    df['id'] = range(len(df))
    df['image_path'] = ''

    df = df.dropna(subset=['text', 'label'])
    df = df[df['text'].str.strip() != '']

    real = df[df['label'] == 0].sample(1000, random_state=42)
    fake = df[df['label'] == 1].sample(1000, random_state=42)
    df = pd.concat([real, fake]).sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, 'sample_data.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} samples to {out_path}")
    print(f"Labels: {df['label'].value_counts().to_dict()}")
    return df

def create_sample_dataset(processed_dir: str):
    return load_real_dataset(processed_dir)

def load_and_validate(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = ['id', 'text', 'label']
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
    print(f"Loaded {len(df)} samples. Labels: {df['label'].value_counts().to_dict()}")
    return df