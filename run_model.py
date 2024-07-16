from transformers import pipeline
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load a pre-trained model
model_name = "distilbert-base-uncased"
nlp = pipeline("sentiment-analysis", model=model_name)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    result = nlp(text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)