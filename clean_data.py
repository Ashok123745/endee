import pandas as pd
import numpy as np

def fix_cleaning():
    print("Reading data...")
    df = pd.read_csv("data/news.csv")
    
    # 1. Force everything to string to avoid the .str accessor error
    df['text'] = df['text'].astype(str).str.strip()
    df['label'] = df['label'].astype(str).str.upper().str.strip()

    # 2. Flexible Mapping: Handles 1, 0, TRUE, FALSE, REAL, FAKE
    # This ensures no rows are dropped just because of format
    valid_labels = {
        'REAL': 'REAL', '1': 'REAL', '1.0': 'REAL', 'TRUE': 'REAL',
        'FAKE': 'FAKE', '0': 'FAKE', '0.0': 'FAKE', 'FALSE': 'FAKE'
    }
    df['label'] = df['label'].map(valid_labels)

    # 3. Drop rows that actually have no text or no valid label
    df.dropna(subset=['text', 'label'], inplace=True)
    df = df[df['text'] != '']
    
    # 4. Save to a NEW file so we don't break the original
    df.to_csv("data/news_cleaned.csv", index=False)
    
    print(f"Cleaning done ✅. Rows remaining: {len(df)}")
    print("Sample labels in cleaned file:", df['label'].unique())

if __name__ == "__main__":
    fix_cleaning()