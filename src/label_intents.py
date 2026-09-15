import pandas as pd
from pathlib import Path


SAMPLE_PATH = Path("data/processed/intent_review_samples.csv")
GOLDEN_PATH = Path("data/golden/amazonhelp_golden.csv")


INTENTS = {
    "1": "delivery_and_shipping",
    "2": "order_tracking",
    "3": "wrong_or_damaged_item",
    "4": "returns_refunds_cancellations",
    "5": "payment_and_billing",
    "6": "account_and_membership",
    "7": "website_app_technical_issue",
    "8": "product_or_service_information",
    "9": "contact_support",
    "10": "service_complaint",
    "11": "other_or_unclear",
}


def display_intents():
    print("\n--- Available Intent Labels ---")

    for number, intent in INTENTS.items():
        print(f"{number}. {intent}")


# Use the saved golden file when resuming.
# Otherwise, start with the original sample file.
if GOLDEN_PATH.exists():
    INPUT_PATH = GOLDEN_PATH
    print("Resuming from previously saved labels...")
else:
    INPUT_PATH = SAMPLE_PATH
    print("Starting a new labeling session...")


print("Loading messages for manual labeling...")

df = pd.read_csv(INPUT_PATH)

# Add the intent column if it does not exist yet.
if "intent" not in df.columns:
    df["intent"] = ""


# Make sure the output folder exists.
GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)

print(f"Messages available: {len(df)}")
print("Type 'q' to stop labeling.\n")


for index, row in df.iterrows():

    # Skip messages that have already been labeled.
    # Skip only messages that already have a real label.
    if pd.notna(row["intent"]) and str(row["intent"]).strip() != "":
        continue

    print("\n" + "=" * 80)
    print(f"Message {index + 1} of {len(df)}")

    print("\nCUSTOMER MESSAGE:")
    print(row["clean_text"])

    display_intents()

    while True:
        choice = input("\nChoose an intent number: ").strip()

        # Save progress and stop.
        if choice.lower() == "q":
            df.to_csv(GOLDEN_PATH, index=False)
            print(f"\nProgress saved to: {GOLDEN_PATH}")
            raise SystemExit

        # Save the selected intent.
        if choice in INTENTS:
            df.at[index, "intent"] = INTENTS[choice]
            break

        print("Invalid choice. Please select a number from 1 to 11.")

    # Save progress after every label.
    df.to_csv(GOLDEN_PATH, index=False)


print("\nLabeling complete!")
print(f"Saved to: {GOLDEN_PATH}")