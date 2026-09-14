import pandas as pd
from pathlib import Path

DATA_PATH = "data/raw/twcs.csv"
OUTPUT_PATH = "data/processed/amazonhelp_conversations.csv"

BRAND = "AmazonHelp"

print("Pass 1: Finding customer tweet IDs answered by AmazonHelp...")

customer_tweet_ids = set()

for chunk in pd.read_csv(DATA_PATH, chunksize=100_000):
    brand_rows = chunk[chunk["author_id"] == BRAND]

    parent_ids = (
        brand_rows["in_response_to_tweet_id"]
        .dropna()
        .astype("int64")
        .astype(str)
    )

    customer_tweet_ids.update(parent_ids.tolist())

print(f"Customer tweets linked to AmazonHelp: {len(customer_tweet_ids)}")

print("\nPass 2: Extracting AmazonHelp replies and customer messages...")

related_chunks = []

for chunk in pd.read_csv(DATA_PATH, chunksize=100_000):

    # AmazonHelp support replies
    brand_rows = chunk[
        chunk["author_id"] == BRAND
    ]

    # Customer tweets that AmazonHelp replied to
    customer_rows = chunk[
        chunk["tweet_id"]
        .astype(str)
        .isin(customer_tweet_ids)
    ]

    related_rows = pd.concat(
        [brand_rows, customer_rows],
        ignore_index=True
    )

    if not related_rows.empty:
        related_chunks.append(related_rows)

conversations = (
    pd.concat(related_chunks, ignore_index=True)
    .drop_duplicates(subset="tweet_id")
)

Path("data/processed").mkdir(
    parents=True,
    exist_ok=True
)

conversations.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n--- Extraction Complete ---")
print(f"Total related tweets: {len(conversations)}")

print("\n--- Inbound Distribution ---")
print(conversations["inbound"].value_counts())

print("\n--- Author Distribution ---")
print(conversations["author_id"].value_counts().head(10))

print(f"\nSaved to: {OUTPUT_PATH}")