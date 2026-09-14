import re
from pathlib import Path

import pandas as pd


INPUT_PATH = "data/processed/amazonhelp_conversations.csv"
OUTPUT_PATH = "data/processed/amazonhelp_reconstructed.csv"


def clean_text(text):
    """Clean common Twitter text artifacts."""
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove artifacts such as ^J and ^CH
    text = re.sub(r"\^[A-Z]+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


print("Loading extracted AmazonHelp data...")

df = pd.read_csv(
    INPUT_PATH,
    dtype={
        "tweet_id": "string",
        "author_id": "string",
        "in_response_to_tweet_id": "string",
        "response_tweet_id": "string",
        "text": "string",
    },
)

print(f"Initial rows: {len(df)}")


def normalize_tweet_id(value):
    """Convert IDs such as '621.0' into '621'."""
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value


# Normalize tweet IDs and parent tweet IDs
df["tweet_id"] = df["tweet_id"].apply(normalize_tweet_id)

df["in_response_to_tweet_id"] = (
    df["in_response_to_tweet_id"]
    .apply(normalize_tweet_id)
)

# Clean text
df["clean_text"] = df["text"].apply(clean_text)

# Remove empty messages
df = df[df["clean_text"].str.len() > 0].copy()

# Remove very short messages
df = df[df["clean_text"].str.len() >= 10].copy()

# Remove duplicate tweet IDs
df = df.drop_duplicates(subset="tweet_id").copy()

print(f"Rows after cleaning: {len(df)}")


# Create a lookup from tweet ID to its parent tweet ID
parent_lookup = (
    df.set_index("tweet_id")["in_response_to_tweet_id"]
    .dropna()
    .to_dict()
)


def find_root(tweet_id):
    """Find the root tweet of a conversation thread."""
    visited = set()
    current_id = tweet_id

    while current_id in parent_lookup:
        if current_id in visited:
            break

        visited.add(current_id)

        parent_id = parent_lookup[current_id]

        if pd.isna(parent_id):
            break

        current_id = str(parent_id)

    return current_id


print("Reconstructing conversation threads...")

df["conversation_id"] = df["tweet_id"].apply(find_root)

# Assign a role
df["role"] = df["inbound"].map(
    {
        True: "customer",
        False: "support",
    }
)

# Parse timestamps
df["created_at_parsed"] = pd.to_datetime(
    df["created_at"],
    format="%a %b %d %H:%M:%S %z %Y",
    errors="coerce",
    utc=True,
)

# Sort messages inside each conversation
df = df.sort_values(
    by=["conversation_id", "created_at_parsed", "tweet_id"]
)

# Assign turn numbers
df["turn_index"] = (
    df.groupby("conversation_id").cumcount() + 1
)

# Keep useful columns
output_columns = [
    "conversation_id",
    "turn_index",
    "tweet_id",
    "author_id",
    "role",
    "clean_text",
    "created_at",
    "in_response_to_tweet_id",
]

reconstructed = df[output_columns].copy()

Path("data/processed").mkdir(
    parents=True,
    exist_ok=True,
)

reconstructed.to_csv(
    OUTPUT_PATH,
    index=False,
)

print("\n--- Reconstruction Complete ---")
print(f"Total messages: {len(reconstructed)}")
print(
    f"Total conversations: "
    f"{reconstructed['conversation_id'].nunique()}"
)

print("\n--- Role Distribution ---")
print(reconstructed["role"].value_counts())

print("\n--- Conversation Length Statistics ---")
conversation_lengths = (
    reconstructed.groupby("conversation_id")
    .size()
)

print(conversation_lengths.describe())

print("\n--- Sample Reconstructed Conversation ---")

sample_id = reconstructed["conversation_id"].iloc[0]

sample = reconstructed[
    reconstructed["conversation_id"] == sample_id
]

for _, row in sample.iterrows():
    print(
        f"\nTurn {row['turn_index']} "
        f"[{row['role'].upper()}]:"
    )
    print(row["clean_text"])

print(f"\nSaved to: {OUTPUT_PATH}")