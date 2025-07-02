from flask import Flask, render_template, request
import joblib
import PyPDF2
import re

app = Flask(__name__)
model = joblib.load('model.pkl')
tfidf = joblib.load('tfidf.pkl')


def extract_text(file):
    if file.filename.endswith('.txt'):
        return file.read().decode('utf-8')
    elif file.filename.endswith('.pdf'):
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages])
    else:
        return ""

def extract_name(text):
    # Look for the first capitalized name (basic heuristic)
    match = re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", text)
    if match:
        return match.group()
    else:
        return "Candidate"

@app.route('/', methods=['GET', 'POST'])
def index():
    name = None
    score = None
    if request.method == 'POST':
        file = request.files['resume']
        if file:
            text = extract_text(file)
            name = extract_name(text)
            vector = tfidf.transform([text])
            score = model.predict_proba(vector).max() * 10  # Scale to 10
            score = round(score, 2)
    return render_template('index.html', name=name, score=score)

if __name__ == '__main__':
    app.run(debug=True)
