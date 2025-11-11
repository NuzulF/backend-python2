# app/sentiment_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import pandas as pd

class BERT_CNN(nn.Module):
    def __init__(self, num_classes=2, bert_model_name='indobenchmark/indobert-base-p1'):
        super(BERT_CNN, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.conv = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=1)
        self.fc = None

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = bert_output.last_hidden_state
        embeddings = embeddings.permute(0, 2, 1)
        conv_output = self.conv(embeddings)
        conv_output = self.relu(conv_output)
        pooled_output = self.pool(conv_output)
        pooled_output = pooled_output.view(pooled_output.size(0), -1)
        if self.fc is None:
            self.fc = nn.Linear(pooled_output.size(1), 2).to(pooled_output.device)
        output = self.fc(pooled_output)
        return output

    @staticmethod
    def build(model_path='models/BERT-CNN_model.pth'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sentiment_model = BERT_CNN(num_classes=2).to(device)
        sentiment_model.fc = nn.Linear(32768, 2).to(device)
        state_dict = torch.load(model_path, map_location=device)
        sentiment_model.load_state_dict(state_dict)
        sentiment_model.eval()
        return sentiment_model

def predict_text_with_score(model, tokenizer, text, device, max_len=128):
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(output, 'logits'):
            output = output.logits
        probabilities = F.softmax(output, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()
        sentiment_score = probabilities.squeeze().tolist()
    return predicted_label, round(5 * sentiment_score[1], 2)

def get_label(score):
    if pd.isna(score):
        return "tidak ada"
    elif score >= 0.5:
        return "positif"
    else:
        return "negatif"

def proses_sentimen(df):
    sentiment_model = BERT_CNN.build()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
    texts = df["review"].tolist()
    scores = []
    for text in texts:
        if not pd.isna(text):
            _, sentiment_score = predict_text_with_score(sentiment_model, tokenizer, text, device)
            scores.append(sentiment_score)
        else:
            scores.append(5)
    df['sentimen'] = scores
    df['kategori_sentimen'] = df['sentimen'].apply(get_label)
    return df
