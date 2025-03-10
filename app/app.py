from transformers import pipeline
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load Hugging Face Sentiment Analysis model
sentiment_model = pipeline("sentiment-analysis")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    result = sentiment_model(text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
