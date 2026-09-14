import pandas as pd

DATA_PATH = "data/processed/amazonhelp_conversations.csv"

print("Loading AmazonHelp conversation data...")

df = pd.read_csv(DATA_PATH)

# Create a lookup table for tweets by tweet ID
tweet_lookup = df.set_index("tweet_id")["text"].to_dict()

# Select AmazonHelp replies
support_replies = df[
    (df["author_id"] == "AmazonHelp")
    & (df["in_response_to_tweet_id"].notna())
]

print(f"Total support replies with parent tweets: {len(support_replies)}")

print("\n--- Sample Customer-Support Conversations ---")

for _, row in support_replies.head(10).iterrows():
    parent_id = int(row["in_response_to_tweet_id"])
    customer_text = tweet_lookup.get(parent_id, "[Customer message not found]")

    print("\n" + "=" * 80)
    print("CUSTOMER:")
    print(customer_text)

    print("\nAMAZONHELP:")
    print(row["text"])

    print(f"\nCustomer tweet ID: {parent_id}")
    print(f"Support tweet ID: {row['tweet_id']}")