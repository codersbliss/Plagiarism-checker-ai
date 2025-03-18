import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")  # Ensure index.html is inside the 'templates' folder

@app.route("/detect", methods=["POST"])
def detect():
    text = request.form.get("text", "")

    if not text:
        return render_template("index.html", result="No text provided.")

    try:
        # Transform the input text using the fitted vectorizer
        text_vectorized = vectorizer.transform([text])

        # Make a prediction using the trained model
        prediction = model.predict(text_vectorized)[0]  # Assuming output is 1 (Plagiarism) or 0 (No Plagiarism)

        # Interpret the prediction result
        result = "Plagiarism Detected 🚨" if prediction == 1 else "No Plagiarism Found ✅"

        return render_template("index.html", result=result)

    except Exception as e:
        return render_template("index.html", result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
