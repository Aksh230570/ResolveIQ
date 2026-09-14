import pandas as pd

DATA_PATH = "data/raw/twcs.csv"

print("Finding authors/brands...")

author_counts = {}

for chunk in pd.read_csv(DATA_PATH, chunksize=100_000):
    counts = chunk["author_id"].value_counts()

    for author, count in counts.items():
        author_counts[author] = author_counts.get(author, 0) + count

authors = (
    pd.Series(author_counts)
    .sort_values(ascending=False)
)

print("\n--- Most Frequent Authors ---")
print(authors.head(50))