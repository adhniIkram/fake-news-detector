import pickle
import os
import sys
from flask import Flask, request, jsonify, render_template
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from processText import processText

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

with open (os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'), 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

with open(os.path.join(MODEL_DIR, 'lr_model.pkl'), 'rb') as file:
    lr_model = pickle.load(file)

app = Flask(__name__)


#Homepage route
@app.route('/')
def home():
    return render_template('index.html')

#Analyzing the input data
@app.route('/analyze', methods=['POST'])
def analyze():

    intext = request.json['text']

    cleantext = processText(intext)
    vectorizedtext = tfidf_vectorizer.transform([cleantext])
    prediction = lr_model.predict(vectorizedtext)[0]

    prediction_proba = lr_model.predict_proba(vectorizedtext)[0]
    confidence = prediction_proba[prediction]

    result = None
    if prediction == 1:
        result = "REAL"
    else:
        result = "FAKE"

    return jsonify({
        'result' : result,
        'confidence' : f"{confidence*100:2f}"
    })

if __name__ == "__main__":
    app.run(debug= True)
