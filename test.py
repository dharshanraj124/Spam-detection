import torch
from src.model import SpamRNN
from src.preprocess import load_vocab, encode, load_data

X_train, X_test, y_train, y_test = load_data("Data/spam.csv")
vocab = load_vocab()

model = SpamRNN(len(vocab)+1)
model.load_state_dict(torch.load("models/rnn_model.pt", weights_only=True))
model.eval()

correct = 0
for text,label in zip(X_test, y_test):
    x = torch.tensor([encode(text, vocab)])
    pred = model(x).item()
    pred = 1 if pred>0.5 else 0
    if pred == label:
        correct += 1

print(f"Accuracy: {correct/len(X_test)*100:.1f}%")
