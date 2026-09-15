
import joblib

MODEL_PATH = "models/baseline_intent_classifier.joblib"

print("Loading intent classifier...")
model = joblib.load(MODEL_PATH)

print("\nResolveIQ Intent Classifier")
print("Type a customer message to classify.")
print("Type 'q' to exit.\n")

while True:
    message = input("Customer message: ").strip()

    if message.lower() == "q":
        print("Exiting classifier.")
        break

    if not message:
        print("Please enter a message.")
        continue

    predicted_intent = model.predict([message])[0]

    print(f"Predicted intent: {predicted_intent}\n")