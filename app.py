from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer
try:
    model = pickle.load(open('model.pkl', 'rb'))
    tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
except Exception as e:
    print("Error loading model/vectorizer:", e)
    model = None
    tfidf_vectorizer = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    input_text = request.form['text']

    # Check if vectorizer and model are loaded
    if model is None or tfidf_vectorizer is None:
        return render_template('index.html', result="Error: Model or Vectorizer not loaded. Retrain the model.")

    # Ensure vectorizer is fitted
    if not hasattr(tfidf_vectorizer, "idf_"):
        return render_template('index.html', result="Error: Vectorizer not fitted. Retrain the model.")

    # Transform input text and predict
    vectorized_text = tfidf_vectorizer.transform([input_text])
    result = model.predict(vectorized_text)

    result = "Plagiarism Detected" if result[0] == 1 else "No Plagiarism"
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
