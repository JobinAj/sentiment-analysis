from transformers import pipeline
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load Hugging Face Sentiment Analysis model
sentiment_model = pipeline("sentiment-analysis")

# Home route to verify API is running
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Sentiment Analysis API is running!"})

# Sentiment analysis route
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data["text"]
        result = sentiment_model(text)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
