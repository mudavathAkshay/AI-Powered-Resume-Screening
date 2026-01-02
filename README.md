if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
# 🚀 AI-Powered Resume Screening



An intelligent web application that evaluates resumes using **Machine Learning (ML)** and **Natural Language Processing (NLP)** techniques.  
This project leverages **Flask**, **scikit-learn**, and **TF-IDF Vectorization** to analyze text and generate a professional resume score.

---

## 🧩 Project Overview

The **AI-Powered Resume Screening** app automates resume evaluation to help recruiters and candidates get instant feedback.  
Users can upload `.pdf` or `.txt` resumes, and the model returns a **score (0–10)** based on keyword density, structure, and overall quality.

<p align="center">
  <img src="static/demo-banner.png" alt="Banner" width="800"/>
</p>

---

## 🧠 Features

- 📂 Upload resumes in `.pdf` or `.txt` formats  
- 🧠 ML model built with **TF-IDF + Classification Algorithm**  
- 🧾 Text extraction from PDFs using **PyPDF2**  
- 🌌 Dark animated UI with falling stars ✨  
- 🐳 Containerized via Docker for consistent deployment  
- ⚡ Lightweight Flask backend (runs on any OS)

---

## 🧰 Tech Stack

| Layer | Tools / Frameworks |
|-------|--------------------|
| **Frontend** | HTML, CSS (Dark theme + animation) |
| **Backend** | Flask (Python Web Framework) |
| **ML Model** | scikit-learn, TF-IDF |
| **Data Handling** | NumPy, Pandas |
| **Packaging** | Joblib |
| **Deployment** | Docker |

---

## 🧾 Architecture Diagram

```
          ┌───────────────────────────┐
          │        Frontend UI        │
          │ (HTML, CSS - Animated UI) │
          └────────────┬──────────────┘
                       │
                       ▼
             ┌───────────────────┐
             │     Flask API     │
             │  (app.py server)  │
             └───────────────────┘
                       │
                       ▼
            ┌────────────────────┐
            │   ML Model (pkl)   │
            │ TF-IDF + Classifier│
            └────────────────────┘
                       │
                       ▼
           ┌──────────────────────────┐
           │  Score & Candidate Name  │
           └──────────────────────────┘
```

---

## 🧩 Folder Structure

```
AI-Powered-Resume-Screening/
│
├── app.py                # Main Flask app
├── model.pkl             # Trained ML model
├── tfidf.pkl             # TF-IDF vectorizer
├── requirements.txt      # Dependencies
├── Dockerfile            # Docker setup
├── .dockerignore         # Ignore rules for Docker
├── templates/
│   └── index.html        # Frontend UI
├── static/
│   ├── style.css         # Custom styling + animation
│   ├── demo1.png         # Upload screenshot
│   ├── demo2.png         # Result screenshot
│   └── demo-banner.png   # Header banner
└── README.md
```

---

## 💻 Run the Application Locally

### 🪶 Clone the Repository
```bash
git clone https://github.com/mudavathAkshay/AI-Powered-Resume-Screening.git
cd AI-Powered-Resume-Screening
```

### 🧱 Create and Activate Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate     # On Windows
source venv/bin/activate  # On Mac/Linux
```

### 📦 Install Dependencies
```bash
pip install -r requirements.txt
```

### ▶️ Run Flask App
```bash
python app.py
```

Then open 👉 [http://localhost:5000](http://localhost:5000)

---<img width="1920" height="920" alt="Screenshot (42)" src="https://github.com/user-attachments/assets/d3db7970-2613-4c82-b8b5-d8e32fbd4b1a" />


## 🐳 Running in Docker

### 🧩 Build Docker Image
```bash
docker build -t resume-app .
```

### 🚀 Run the Container
```bash
docker run -p 5000:5000 resume-app
```

Then open your browser at 👉 [http://localhost:5000](http://localhost:5000)

---

## 🧠 Model Overview

The ML model uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to extract text features from resumes and a **classifier** trained on labeled resume data.  
This enables prediction of resume quality, relevancy, and skill presence.

---

## 📊 Example Screens

| Upload Page | Result Page |
|--------------|-------------|
| ![Upload Page](static/demo1.png) | ![Result Page](static/demo2.png) |

---

## 🧪 Testing the Application

Once running, upload sample `.pdf` or `.txt` resumes to test:
- Try with different career domains (Data Science, Developer, etc.)
- Observe changes in the predicted score

---

## 📜 Requirements

| Package | Version |
|----------|----------|
| Flask | 3.0.0 |
| joblib | 1.3.2 |
| PyPDF2 | 3.0.1 |
| scikit-learn | 1.4.0 |
| numpy | 1.26.4 |
| pandas | 2.2.1 |
| gunicorn | 21.2.0 |

---

## 🔧 Build a Container Image (Advanced)

You can use the Dockerfile provided to create a standalone container image:

```bash
docker build -t resume-screening-app .
docker run -d -p 5000:5000 resume-screening-app
```

---

## 🧠 Troubleshooting

| Issue | Possible Fix |
|--------|---------------|
| `ModuleNotFoundError: No module named 'flask'` | Run `pip install flask` |
| `ModuleNotFoundError: No module named 'sklearn'` | Install `scikit-learn` (not `sklearn`) |
| Permission denied on venv | Run PowerShell as Administrator |
| Flask not starting | Check `app.run(host='0.0.0.0', port=5000)` in `app.py` |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to open a [Pull Request](https://github.com/mudavathAkshay/AI-Powered-Resume-Screening/pulls) or [Issue](https://github.com/mudavathAkshay/AI-Powered-Resume-Screening/issues).

**Contribution Steps:**
1. Fork the repository  
2. Create your feature branch  
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit changes  
   ```bash
   git commit -m "Add new feature"
   ```
4. Push and open a PR

---

## 📚 Documentation

- [Flask Documentation](https://flask.palletsprojects.com/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Docker Reference](https://docs.docker.com/)
- [PyPDF2 Docs](https://pypdf2.readthedocs.io/en/latest/)

---

## 🧑‍💻 Author

**✨ Project done by [Akshay](https://www.linkedin.com/in/mudavath-akshay/)**  
_Data Scientist | AI Developer | Python Enthusiast_  

📧 **Email:** yourname@example.com  
🔗 **GitHub:** [github.com/mudavathAkshay](https://github.com/mudavathAkshay)  
💼 **LinkedIn:** [linkedin.com/in/mudavath-akshay](https://www.linkedin.com/in/mudavath-akshay/)

---

## 🪄 Future Enhancements

- Add support for `.docx` file parsing  
- Deploy to cloud (Render / AWS / Heroku)  
- Integrate OpenAI for text-based scoring  
- Create HR dashboard for analytics  

---

## 🧾 License

This project is licensed under the [MIT License](LICENSE) — feel free to use and modify for your own learning and projects.

---

> “Built with ❤️, Flask, and Machine Learning by Akshay.”
