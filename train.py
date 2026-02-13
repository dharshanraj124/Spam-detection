import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.preprocess import load_data, build_vocab, encode, save_vocab
from src.model import SpamRNN
import numpy as np

# Load
X_train, X_test, y_train, y_test = load_data("Data/spam.csv")

vocab = build_vocab(X_train)
save_vocab(vocab)  # Save vocab so app.py uses the SAME mapping

X_train_enc = [encode(t, vocab) for t in X_train]
X_test_enc  = [encode(t, vocab) for t in X_test]

train_ds = TensorDataset(torch.tensor(X_train_enc), torch.tensor(y_train.values, dtype=torch.float32))
loader = DataLoader(train_ds, batch_size=64, shuffle=True)

# Handle class imbalance with weighted loss
n_spam = y_train.sum()
n_ham = len(y_train) - n_spam
pos_weight = torch.tensor([n_ham / n_spam])
print(f"Dataset: {n_ham} ham, {n_spam} spam (ratio: {n_ham/n_spam:.1f}:1)")
print(f"Using pos_weight={pos_weight.item():.2f} to balance classes")

model = SpamRNN(len(vocab)+1)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Remove sigmoid from model output for BCEWithLogitsLoss - we need raw logits
# Actually, let's keep the model with sigmoid and use BCELoss with class weights via manual weighting
loss_fn = nn.BCELoss(reduction='none')  # per-sample loss for manual weighting
opt = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
EPOCHS = 15
weight_spam = n_ham / n_spam  # upweight spam samples

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in loader:
        opt.zero_grad()
        out = model(x).squeeze()
        raw_loss = loss_fn(out, y)
        # Apply class weights: upweight spam (minority) samples
        weights = torch.where(y == 1, weight_spam, 1.0)
        loss = (raw_loss * weights).mean()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    
    # Quick eval every epoch
    model.eval()
    with torch.no_grad():
        correct = 0
        tp = fp = tn = fn = 0
        for text, label in zip(X_test, y_test):
            x = torch.tensor([encode(text, vocab)])
            pred = model(x).item()
            p = 1 if pred > 0.5 else 0
            if p == label:
                correct += 1
            if p == 1 and label == 1: tp += 1
            elif p == 1 and label == 0: fp += 1
            elif p == 0 and label == 0: tn += 1
            elif p == 0 and label == 1: fn += 1
        acc = correct / len(X_test) * 100
        precision = tp / (tp+fp) if (tp+fp) > 0 else 0
        recall = tp / (tp+fn) if (tp+fn) > 0 else 0
        print(f"Epoch {epoch+1:2d}/{EPOCHS}  Loss: {total_loss:.4f}  Acc: {acc:.1f}%  Spam Precision: {precision:.2f}  Spam Recall: {recall:.2f}")

torch.save(model.state_dict(), "models/rnn_model.pt")
print("\nTraining complete. Model and vocab saved.")
