import os
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from transformers import pipeline
from flask_pymongo import PyMongo
from newspaper import Article

app = Flask(__name__)
account_sid = 'KEY'
auth_token = 'KEY'
client = Client(account_sid,auth_token)
tokenizer = AutoTokenizer.from_pretrained("hamzab/roberta-fake-news-classification")
model = AutoModelForSequenceClassification.from_pretrained("hamzab/roberta-fake-news-classification")
pipe = pipeline("text-classification", model="ealvaradob/bert-finetuned-phishing")
app.secret_key = 'KEY/'
app.config["MONGO_URI"] = "mongodb://localhost:27017/FakeNews"
mongo = PyMongo(app)


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
    message = request.form.get('Body')
    if message:
        inp = message.split('^')
        if inp[0].lower()=='news':
            article_url = inp[1]
            article = Article(article_url)
            article.download()
            article.parse()
            x = predict_fake(article.title,article.text)
            if x["Fake"]>x["Real"]:
                if not mongo.db.NewsList.find_one({'Body':article.title}):
                    mongo.db.NewsList.insert_one({'Type':'News','Body':article.title,'Prediction':"Fake",'Score':round(x["Fake"]*100,4)})
                return respond('Fake news with score {}%'.format(round(x["Fake"]*100,3)))
            else:
                if not mongo.db.NewsList.find_one({'Body':article.title}):
                    mongo.db.NewsList.insert_one({'Type':'News','Body':article.title,'Prediction':"Real",'Score':round(x["Real"]*100,4)})
                return respond('Real news with score {}%'.format(round(x["Real"]*100,3)))
        else:
            t = pipe(inp[0])
            if not mongo.db.NewsList.find_one({'Body':inp[0]}):
                mongo.db.NewsList.insert_one({'Type':'Social Media','Body':inp[0],'Prediction':t[0]['label'],'Score':round(t[0]['score']*100,4)})
            return respond('Given snippet is ' + t[0]['label'] + ' with score ' + str(round(t[0]['score']*100,4)) +'%') 
            