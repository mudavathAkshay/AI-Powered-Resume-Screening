from flask import Flask, render_template, request
import joblib
import PyPDF2
import re
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
TFIDF_PATH = os.path.join(os.path.dirname(__file__), "tfidf.pkl")

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)


def extract_text(file):
    """Extract text from PDF or TXT file"""
    if file.filename.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        text = []
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text.append(content)
        return "\n".join(text)
    else:
        return ""


def extract_name(text):
    """Basic name extraction using regex"""
    match = re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", text)
    return match.group() if match else "Candidate"


@app.route("/", methods=["GET", "POST"])
def index():
    name = None
    score = None
    if request.method == "POST":
        file = request.files.get("resume")
        if file:
            text = extract_text(file)
            if text.strip():
                name = extract_name(text)
                vector = tfidf.transform([text])
                score = round(model.predict_proba(vector).max() * 10, 2)
    return render_template("index.html", name=name, score=score)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
