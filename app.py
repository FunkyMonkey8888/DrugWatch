from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

# Load pre-trained BERT model and tokenizer
MODEL_PATH = "E:/DrugWatch/bert_multilingual_uncased"  # Path to your BERT model folder
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Function to classify messages
def classify_message(message):
    inputs = tokenizer(message, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities).item()
    confidence = probabilities[0][predicted_class].item()
    return predicted_class, confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "Message is required"}), 400

    try:
        predicted_class, confidence = classify_message(message)
        result = {
            "message": message,
            "is_related_to_drugs": bool(predicted_class),  # Assuming 1 = related to drugs, 0 = not related
            "confidence": confidence
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/payload', methods=['GET', 'POST'])
def collect_data():
    # Collect metadata
    ip_address = request.remote_addr
    user_agent = request.headers.get('User-Agent')
    data = {
        "ip_address": ip_address,
        "user_agent": user_agent,
        "headers": dict(request.headers)
    }
    # Log data (for demo purposes, consider secure storage)
    print(f"Data collected: {data}")

    # Simulated response
    return jsonify({"message": "Thank you for visiting!"}), 200
import requests
from flask import Flask, request, jsonify

# app = Flask(__name__)

def get_geolocation(ip):
    try:
        response = requests.get(f"https://ipinfo.io/{ip}/json")
        if response.status_code == 200:
            return response.json()
        return {"error": "Geolocation service unavailable"}
    except Exception as e:
        return {"error": str(e)}

@app.route('/track', methods=['GET'])
def track():
    user_ip = request.remote_addr
    user_agent = request.headers.get('User-Agent')
    geolocation = get_geolocation(user_ip)
    
    data = {
        'IP Address': user_ip,
        'User Agent': user_agent,
        'Geolocation': geolocation
    }
    
    print(f"Data Received: {data}")
    return jsonify({'status': 'success', 'message': 'Tracking data received', 'data':data}), 200

from flask import Flask, request, jsonify

# app = Flask(__name__)


# @app.route('/track', methods=['POST'])
# def track():
#     data = request.json
#     print(f"Data Received: {data}")
#     return jsonify({'status': 'success', 'message': 'Tracking data received'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

