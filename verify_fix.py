import torch
from src.model import SpamRNN
from src.preprocess import load_vocab, encode, clean

vocab = load_vocab()
model = SpamRNN(len(vocab)+1)
model.load_state_dict(torch.load("models/rnn_model.pt", weights_only=True))
model.eval()

tests = [
    ("Congratulations! You won a free iPhone. Click here to claim now!", "SPAM"),
    ("Hey, are we still meeting for lunch tomorrow?", "HAM"),
    ("WINNER!! You have been selected for a cash prize of 1000. Call now", "SPAM"),
    ("Can you send me the project report by Friday?", "HAM"),
    ("FREE entry to win a trip! Text WIN to 80888 now", "SPAM"),
    ("Reminder: Your dentist appointment is at 3pm tomorrow", "HAM"),
]

print("\n--- Spam Detection Test ---")
for text, expected in tests:
    seq = encode(clean(text), vocab)
    x = torch.tensor([seq], dtype=torch.long)
    with torch.no_grad():
        pred_val = model(x).item()
    predicted = "SPAM" if pred_val > 0.5 else "HAM"
    match = "OK" if predicted == expected else "FAIL"
    print(f"[{match:4s}] Expected: {expected:4s} | Predicted: {predicted:4s} (conf: {pred_val:.3f}) | {text[:55]}")
