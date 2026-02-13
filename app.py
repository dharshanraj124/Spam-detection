from flask import Flask, render_template, request
import torch
from src.model import SpamRNN
from src.preprocess import encode, load_vocab, clean

app = Flask(__name__)

# Load the SAME vocab that was used during training
vocab = load_vocab()

# Load model
model = SpamRNN(len(vocab) + 1)
model.load_state_dict(torch.load("models/rnn_model.pt", weights_only=True))
model.eval()

def predict(text):
    cleaned_text = clean(text)
    seq = encode(cleaned_text, vocab)
    x = torch.tensor([seq], dtype=torch.long)
    with torch.no_grad():
        pred = model(x).item()
        result = 1 if pred > 0.5 else 0
    return "Spam" if result == 1 else "Not Spam"


@app.route("/", methods=["GET","POST"])
def home():
    result = ""
    if request.method == "POST":
        msg = request.form["message"]
        result = predict(msg)
    return render_template("ui.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
