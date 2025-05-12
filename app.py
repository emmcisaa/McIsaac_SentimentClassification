import os
import pickle

from flask import Flask, render_template, request

from update import store_review
from vectorizer import get_vectorizer, preprocess_text, tokenizer

app = Flask(__name__)

# Load model + NLP tools
clf = pickle.load(open(os.path.join('pkl_objects', 'classifier.pkl'), 'rb'))
stop = pickle.load(open(os.path.join('pkl_objects', 'stopwords.pkl'), 'rb'))
porter = pickle.load(open(os.path.join('pkl_objects', 'porter.pkl'), 'rb'))

# Rebuild vectorizer and fit on full dataset
tfidf = get_vectorizer()

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction page
@app.route('/results', methods=['POST'])
def results():
    if request.method == 'POST':
        review = request.form['review']
        cleaned = preprocess_text(review)
        vect = tfidf.transform([cleaned])
        prediction = clf.predict(vect)[0]
        prob = clf.predict_proba(vect).max()
        return render_template('results.html',
                               prediction=prediction,
                               probability=round(prob * 100, 2),
                               review=review)

# Feedback collection page
@app.route('/thanks', methods=['POST'])
def thanks():
    review = request.form['review']
    predicted_sentiment = int(request.form['sentiment'])
    feedback = int(request.form['feedback'])

    # If feedback was "incorrect", flip the sentiment
    final_sentiment = predicted_sentiment if feedback == 1 else int(not predicted_sentiment)

    store_review('reviews.sqlite', review, final_sentiment)
    return render_template('thanks.html')

# Run server
if __name__ == '__main__':
    app.run(debug=True, port=5050)
