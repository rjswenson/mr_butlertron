#!/user/bin/env python3

import sys
# from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, AutoConfig
# from flask import Flask, request, jsonify

# app = Flask(__name__)

# Load a pre-trained model
model_name = 'google/flan-t5-xl'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = GenerationConfig(max_new_tokens=200)

# TODO: pipeline vs above model def?
# nlp = pipeline("sentiment-analysis", model=model_name)

# creates a pytorch tensor
for line in sys.stdin:
    tokens = tokenizer(line, return_tensors="pt")
    outputs = model.generate(**tokens, generation_config=config)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.json
#     text = data.get("text", "")
#     result = nlp(text)
#     return jsonify(result)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)
