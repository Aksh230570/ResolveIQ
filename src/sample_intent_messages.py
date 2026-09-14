import pandas as pd
from pathlib import Path

INPUT_PATH = "data/processed/amazonhelp_reconstructed.csv"
OUTPUT_PATH = "data/processed/intent_review_samples.csv"

SAMPLE_SIZE = 100

print("Loading reconstructed conversations...")

df = pd.read_csv(INPUT_PATH)

# Keep only customer messages
customer_messages = df[
    df["role"] == "customer"
].copy()

# Remove empty messages
customer_messages = customer_messages[
    customer_messages["clean_text"].notna()
    & (customer_messages["clean_text"].str.strip() != "")
]

# Select a reproducible random sample
samples = customer_messages.sample(
    n=min(SAMPLE_SIZE, len(customer_messages)),
    random_state=42
)

# Keep the fields useful for manual review
samples = samples[
    [
        "conversation_id",
        "turn_index",
        "tweet_id",
        "clean_text",
    ]
].reset_index(drop=True)

Path("data/processed").mkdir(
    parents=True,
    exist_ok=True
)

samples.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n--- Sampling Complete ---")
print(f"Customer messages available: {len(customer_messages)}")
print(f"Messages sampled: {len(samples)}")
print(f"Saved to: {OUTPUT_PATH}")

print("\n--- First 20 Sample Messages ---")

for index, row in samples.head(20).iterrows():
    print(f"\n[{index + 1}]")
    print(row["clean_text"])