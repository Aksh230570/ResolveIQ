import pandas as pd

DATA_PATH = "data/processed/amazonhelp_conversations.csv"

print("Loading AmazonHelp data...")

df = pd.read_csv(DATA_PATH)

print("\n--- Basic Information ---")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Duplicate Tweet IDs ---")
print(f"Duplicate tweet IDs: {df['tweet_id'].duplicated().sum()}")

print("\n--- Inbound Distribution ---")
print(df["inbound"].value_counts())

print("\n--- Message Length Statistics ---")
df["text_length"] = df["text"].fillna("").str.len()

print(df["text_length"].describe())

print("\n--- Empty or Very Short Messages ---")
short_messages = df[df["text_length"] < 10]
print(f"Messages shorter than 10 characters: {len(short_messages)}")

print("\n--- Support Replies With Parent Tweets ---")
support_replies = df[
    (df["author_id"] == "AmazonHelp")
    & (df["in_response_to_tweet_id"].notna())
]

print(f"Support replies with parent IDs: {len(support_replies)}")

print("\n--- Customer Messages ---")
customer_messages = df[df["inbound"] == True]
print(f"Customer messages: {len(customer_messages)}")

print("\n--- Customer Messages With Text ---")
customer_with_text = customer_messages[
    customer_messages["text"].notna()
    & (customer_messages["text"].str.strip() != "")
]

print(f"Customer messages with usable text: {len(customer_with_text)}")

print("\n--- Top Message Authors ---")
print(df["author_id"].value_counts().head(15))

print("\nAnalysis complete.")