import pandas as pd
from pathlib import Path

DATA_PATH = "data/raw/twcs.csv"
OUTPUT_PATH = "data/processed/amazonhelp_tweets.csv"

brand = "AmazonHelp"

print(f"Extracting tweets related to {brand}...")

chunks = []

for chunk in pd.read_csv(DATA_PATH, chunksize=100_000):
    brand_tweets = chunk[chunk["author_id"] == brand]

    if not brand_tweets.empty:
        chunks.append(brand_tweets)

amazonhelp_df = pd.concat(chunks, ignore_index=True)

Path("data/processed").mkdir(parents=True, exist_ok=True)

amazonhelp_df.to_csv(OUTPUT_PATH, index=False)

print("\n--- Extraction Complete ---")
print(f"Rows extracted: {len(amazonhelp_df)}")
print(f"Saved to: {OUTPUT_PATH}")
print("\n--- Sample Rows ---")
print(amazonhelp_df[["tweet_id", "author_id", "inbound", "text"]].head())