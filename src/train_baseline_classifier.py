
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


INPUT_PATH = "data/golden/amazonhelp_golden_final.csv"
MODEL_PATH = "models/baseline_intent_classifier.joblib"


print("Loading golden dataset...")
df = pd.read_csv(INPUT_PATH)

# Keep only usable labeled examples
df["clean_text"] = df["clean_text"].fillna("").astype(str)
df["intent"] = df["intent"].fillna("").astype(str).str.strip()

df = df[
    (df["clean_text"].str.strip() != "")
    & (df["intent"] != "")
].copy()

print(f"Usable labeled examples: {len(df)}")
print("\nIntent distribution:")
print(df["intent"].value_counts())

X = df["clean_text"]
y = df["intent"]

# Stratification requires at least two examples in every class
class_counts = y.value_counts()
use_stratify = class_counts.min() >= 2

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y if use_stratify else None
)

print(f"\nTraining examples: {len(X_train)}")
print(f"Testing examples: {len(X_test)}")

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_features=10000
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])

print("\nTraining baseline classifier...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nBaseline accuracy: {accuracy:.4f}")
print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

Path("models").mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")