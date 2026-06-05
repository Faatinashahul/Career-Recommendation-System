# Careerly 
### AI-Powered Career Path Predictor for CS/IT Students

Careerly is a machine learning web app that predicts the best career path for CS/IT students based on their academic grades, skills, and interests — and generates a personalized roadmap to get there.

Built as a final year project using real student placement data.

---

## Features

- **Career Prediction** — ML model trained on real CS/IT student placement records predicts your best-fit role
- **AI-Generated Reports** — Personalized explanation of why a career fits you, powered by LLM
- **Skill Roadmap** — Step-by-step learning plan with specific courses and milestones
- **Company Matching** — Top companies hiring for your predicted role
- **Multi-Step Form** — Clean 3-step form covering academics, skills, interests, and certifications

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML Model | Scikit-learn (trained on IFP dataset) |
| AI Reports | Local GGUF LLM via llama-cpp-python |
| Frontend | HTML, CSS, Bootstrap 5 |
| Data | Pandas, Joblib |
| Visualization | Tableau (Careerly Dashboard) |

---

## Project Structure

```
careerly/
├── app.py                  # Flask app & prediction logic
├── career_model.pkl        # Trained ML classification model
├── company_mapping.pkl     # Role → companies mapping
├── requirements.txt
├── templates/
│   ├── home.html           # Landing page
│   ├── index.html          # Prediction form (3-step)
│   ├── result.html         # Career report output
│   └── base.html
└── static/
    └── background.png
```

---

## How to Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/careerly.git
cd careerly
```

**2. Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
python app.py
```

Open `http://localhost:5000` in your browser.

> **Note:** The AI report generation requires a local GGUF model (Mistral 7B). Without it the app still works — career prediction and company matching will function, but the AI explanation section will be skipped.

---

## The Dataset

The model was trained on academic performance data from CS/IT students (batches 2018–22 and 2019–23), covering 8 core subjects:

| Code | Subject |
|---|---|
| UCS1511 | Mathematics |
| UCS1512 | Programming Fundamentals |
| UCS1601 | Data Structures |
| UCS1602 | Database Management Systems |
| UCS1603 | Operating Systems |
| UCS1604 | Computer Networks |
| UCS1701 | Software Engineering |
| UCS1818 | AI & Machine Learning |

Grades are encoded O → A+ → A → B+ → B (1–5 scale). The model also considers certifications, interests, and skills.

---

## Career Roles Predicted

- Data and AI Analyst
- Junior Software Engineer
- Software Engineer Trainee
- Cyber Security Analyst
- Test Engineer
- Mobile Application Developer
- Programmer Analyst
- Graduate Analyst
- Production Manager
- Senior Associate

---


## Deployment

The app can be deployed on **Render** (free tier):

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New Web Service → Connect repo
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `python app.py`
5. Deploy

---

## Future Improvements

- [ ] Replace local LLM with Anthropic/OpenAI API for cloud deployment
- [ ] Resume upload and parsing for auto-filled predictions
- [ ] User accounts and saved career reports
- [ ] Expanded dataset with more batches and roles
- [ ] Mobile responsive redesign

---

## Team

Built with ❤️ as part of the IFP (Industry Focused Project) at **Sri Sivasubramaniya Nadar College of Engineering**.

---

## License

MIT License — free to use, modify, and distribute.
