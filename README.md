# 🚨 Drug Trafficking Detection System using BERT & OSINT

This project presents a comprehensive pipeline for detecting drug-related content from Telegram using a multilingual BERT-based classification model. The system is designed to identify and analyze suspicious messages, classify them, and perform further OSINT profiling using geolocation tools like Seeker.

---

## 📌 Project Features

- 🔍 **Message Classification** using BERT (Multilingual, Uncased)
- 🧠 **Custom Dataset Training** with Torch and Transformers
- 🌐 **Telegram Data Extraction** via automated crawlers
- 🛰️ **OSINT Integration** using IP tracking (Seeker)
- 📊 **Admin Dashboard** for reviewing classified messages
- 🧩 Modular and extensible architecture for future enhancements

---

## 🛠️ Technologies Used

- **Python 3.10+**
- **PyTorch**
- **Transformers (HuggingFace)**
- **Pandas, Scikit-learn**
- **Flask** for backend services
- **MongoDB** for storing classified Telegram data
- **Ngrok + Seeker** for tracking IP Geolocation
- **React (optional)** for frontend dashboard

---

## 📁 Project Structure

DrugDetectionProject/ │ ├── dataset.xlsx # Telegram messages with labels ├── model_training.py # Training script for BERT classifier ├── inference.py # Predict function for new messages ├── seeker/ # IP location tracking script ├── frontend/ # Web interface (React/Bootstrap) ├── app.py # Flask backend API └── bert_multilingual_uncased/ # Saved BERT model and tokenizer

yaml
Copy
Edit

---

## 🚀 How to Run

1. **Clone the repo**
```bash
git clone https://github.com/yourusername/DrugDetectionProject.git
cd DrugDetectionProject
Install requirements

bash
Copy
Edit
pip install -r requirements.txt
Train BERT Model

bash
Copy
Edit
python model_training.py
Run Flask API

bash
Copy
Edit
python app.py
Run Seeker

bash
Copy
Edit
cd seeker
bash seeker.sh
🧪 Sample Inference
python
Copy
Edit
from inference import classify_message
result = classify_message("Buy MDMA pills online", model, tokenizer)
print("Prediction:", result)
📈 Results
Accuracy: ~94% on validation set

Model: BERT Multilingual Uncased (fine-tuned)

Test Case Example: "Buy LSD now" → Detected as drug-related (1)

👨‍🔬 Project Goals
This system aims to assist law enforcement and cybercrime cells in monitoring Telegram channels involved in illegal drug trafficking by using deep learning and OSINT triangulation techniques.
