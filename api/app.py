import os
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)
account_sid = 'TWILIO ACCOUNT SID'
auth_token = 'TWILIO AUTH TOKEN'
client = Client(account_sid,auth_token)
tokenizer = AutoTokenizer.from_pretrained("hamzab/roberta-fake-news-classification")
model = AutoModelForSequenceClassification.from_pretrained("hamzab/roberta-fake-news-classification")
    
def predict_fake(title,text):
    input_str = "<title>" + title + "<content>" +  text + "<end>"
    input_ids = tokenizer.encode_plus(input_str, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    device =  'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    with torch.no_grad():
        output = model(input_ids["input_ids"].to(device), attention_mask=input_ids["attention_mask"].to(device))
    return dict(zip(["Fake","Real"], [x.item() for x in list(torch.nn.Softmax()(output.logits)[0])] ))

def respond(message):
    response = MessagingResponse()
    response.message(message)
    return str(response)

@app.route('/message', methods=['POST'])
def reply():
    message = request.form.get('Body').lower()
    if message:
        inp = message.split('^')
        print(inp)
        x = predict_fake(inp[0],inp[1])
        if x["Fake"]>x["Real"]:
            return respond('Fake news with score {}'.format(x["Fake"]))
        else:
            return respond('Real news with score {}'.format(x["Real"]))
                