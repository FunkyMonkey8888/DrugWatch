import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW
from transformers import get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel("dataset.xlsx")  # Replace with your file path

# Ensure `Content` and `Label` columns exist
df = df[['Content', 'Label']].dropna()

# Split into train and test sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['Content'].values, df['Label'].values, test_size=0.2, random_state=42
)
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")

# Tokenize datasets
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128, return_tensors="pt")
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128, return_tensors="pt")

train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)


class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# Create datasets
train_dataset = CustomDataset(train_encodings, train_labels)
val_dataset = CustomDataset(val_encodings, val_labels)

from transformers import BertForSequenceClassification

# Load BERT multilingual uncased model for binary classification
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-uncased", num_labels=2)

from torch.utils.data import DataLoader

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)

# Learning rate scheduler
num_training_steps = len(train_loader) * 3  # Assuming 3 epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

from tqdm import tqdm

epochs = 1

for epoch in range(epochs):
    # Training
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    print(f"Epoch {epoch + 1} | Train Loss: {train_loss / len(train_loader)}")
    
    # Validation
    model.eval()
    val_loss = 0
    preds, targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            val_loss += outputs.loss.item()
            
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            targets.extend(batch['labels'].cpu().numpy())

    print(f"Epoch {epoch + 1} | Validation Loss: {val_loss / len(val_loader)}")
    print(classification_report(targets, preds))

model.save_pretrained("./bert_multilingual_uncased")
tokenizer.save_pretrained("./bert_multilingual_uncased")
def predict(text, model, tokenizer):
    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    encoding = {k: v.to(device) for k, v in encoding.items()}
    output = model(**encoding)
    logits = output.logits
    prediction = torch.argmax(logits, dim=-1)
    return prediction.item()

# Test example
sample_text = "Buy DMT online now!"
print("Prediction:", predict(sample_text, model, tokenizer))

def classify_message(text, model, tokenizer):
    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    encoding = {k: v.to(device) for k, v in encoding.items()}
    output = model(**encoding)
    logits = output.logits
    prediction = torch.argmax(logits, dim=-1).item()
    return prediction  # 1: Drug-related, 0: Not drug-related

api_key = "wMD5llzSKclBzv2JiqVaHkWSdF7i83P0"


import re
import requests

# Regular expressions for IP, email, and phone numbers
ip_regex = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
email_regex = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
phone_regex = r'\b\d{10}\b'  # Example for 10-digit phone numbers

def extract_metadata(text):
    ip_addresses = re.findall(ip_regex, text)
    emails = re.findall(email_regex, text)
    phone_numbers = re.findall(phone_regex, text)
    return {"ip_addresses": ip_addresses, "emails": emails, "phone_numbers": phone_numbers}

# Enrich IP information using an API (e.g., ipinfo.io)
def enrich_ip_info(ip):
    response = requests.get(f"https://ipinfo.io/{ip}/json")
    if response.status_code == 200:
        return response.json()
    return {}

import whois

def get_domain_info(email):
    domain = email.split('@')[-1]
    try:
        domain_info = whois.whois(domain)
        return domain_info
    except Exception as e:
        return str(e)

import shodan

def shodan_lookup(ip, api_key):
    api = shodan.Shodan(api_key)
    try:
        result = api.host(ip)
        return result
    except Exception as e:
        return str(e)

def osint_detection_pipeline(text, model, tokenizer, shodan_api_key):
    # Step 1: Classify the message
    prediction = classify_message(text, model, tokenizer)
    
    if prediction == 1:  # Drug-related message
        print("Drug-related message detected.")
        
        # Step 2: Extract metadata
        metadata = extract_metadata(text)
        print("Extracted Metadata:", metadata)
        
        # Step 3: Enrich metadata
        enriched_data = {}
        for ip in metadata["ip_addresses"]:
            enriched_data[ip] = enrich_ip_info(ip)
            enriched_data[ip]["shodan"] = shodan_lookup(ip, shodan_api_key)
        
        for email in metadata["emails"]:
            enriched_data[email] = get_domain_info(email)
        
        return enriched_data
    else:
        print("Message is not drug-related.")
        return {}

# Example message
message = "Buy DMT online! Contact: user@example.com or +1234567890. IP: 192.168.1.1"

# Shodan API key
# shodan_api_key = "your_shodan_api_key"

# Run the pipeline
result = osint_detection_pipeline(message, model, tokenizer, api_key)
print("Detection Results:", result)

