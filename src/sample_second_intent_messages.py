import pandas as pd
from pathlib import Path


INPUT_PATH = "data/processed/amazonhelp_reconstructed.csv"
OUTPUT_PATH = "data/processed/intent_review_samples_2.csv"


print("Loading reconstructed AmazonHelp conversations...")

df = pd.read_csv(INPUT_PATH)

# Keep only customer messages.
customer_df = df[df["role"] == "customer"].copy()

# Remove messages without usable text.
customer_df = customer_df[
    customer_df["clean_text"].notna()
    & (customer_df["clean_text"].str.strip() != "")
]

print(f"Customer messages available: {len(customer_df)}")

# Create a different sample using a different random seed.
sample_df = customer_df.sample(
    n=100,
    random_state=123
).copy()

# Add an empty intent column.
sample_df["intent"] = ""

# Ensure the output directory exists.
Path("data/processed").mkdir(parents=True, exist_ok=True)

# Save the second batch.
sample_df.to_csv(OUTPUT_PATH, index=False)

print(f"Messages sampled: {len(sample_df)}")
print(f"Saved to: {OUTPUT_PATH}")