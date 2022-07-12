from flask import Flask, render_template, request
import joblib
import sys
from src import predict_model


# # to run locally, uncomment below
# sys.path.append(r'/Users/chekwei/Documents/Personal/AIAP/review-classification/src')
# # to run locally, uncomment above

sys.path.append('src')

model = joblib.load("src/models/tfidf_log_rep.joblib")
vectorizer = joblib.load("src/models/vectorizer.joblib")

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/home', methods=['POST'])
def home():
    inf_review_text = request.form['input_review']
    pred, proba = predict_model.run_predict(model, vectorizer, [inf_review_text])
    if pred == 0:
        proba = proba[0][0]
    else:
        proba = proba[0][1]
    proba = str(round(proba.item() * 100, 2))
    return render_template('home.html', pred=pred, proba=proba)

# if __name__ == "__main__":
#     app.run(debug=True)

# from flask import Flask
# app = Flask(__name__)

# @app.route('/')
# def hello_geek():
#     return '<h1>Hello from Flask & Docker</h2>'

if __name__ == "__main__":
    app.run(debug=True)