# HybridAI Content Prediction System

HybridAI is a web-based machine learning project for analyzing digital content. It brings together sentiment analysis, fake engagement detection, popularity prediction, and live monitoring in one dashboard.

The goal of the project is simple: help users understand how content is performing and support better promotion decisions.

## What This Project Does

- Analyze text sentiment
- Detect suspicious or fake engagement
- Predict content popularity
- Show live prediction updates
- Provide simple decision support and what-if analysis

## Main Modules

### 1. Sentiment Analysis
This module analyzes text and identifies whether the sentiment is positive, negative, or neutral.

### 2. Fake Engagement Detection
This module checks engagement patterns and helps identify suspicious activity such as fake likes or unusual interaction behavior.

### 3. Popularity Prediction
This module predicts future content performance using features like views, likes, comments, engagement rate, sentiment score, and followers.

### 4. Live Prediction
This module provides real-time prediction support and helps users monitor changing content performance.

## Tech Stack

### Frontend
- React
- Material UI
- Recharts
- Axios

### Backend
- FastAPI
- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- TextBlob

## Project Structure

```text
hybrid-ai-content-popularity-prediction/
|-- backend/
|   |-- main_app.py
|   |-- models/
|   |-- requirements.txt
|
|-- frontend/
|   |-- src/
|   |   |-- components/
|   |   |-- App.js
|   |-- package.json
|
|-- datasets/
|-- notebooks/
|-- README.md
```

## How to Run

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main_app:app --reload --host 127.0.0.1 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm start
```

## App URLs

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

## Dataset

The full project dataset is available here:

- Google Drive: `https://drive.google.com/drive/folders/1G_va3pWg0hYAqFfevWvc34Brw5cVOc_T`

Download the dataset files from the shared folder and place them in the `datasets/` directory if you want to run the full training and preprocessing pipeline locally.

## Machine Learning Used

- Sentiment Analysis using NLP-based methods
- Fake Engagement Detection using anomaly detection logic
- Popularity Prediction using Random Forest Regressor
- Feature preprocessing using RobustScaler

## Why This Project Is Useful

This project is useful for content creators, marketers, and businesses who want to evaluate content before promoting it. Instead of only showing raw numbers, the system helps users understand performance, detect risks, and make better decisions.

## Future Improvements

- Add real social media API integration
- Support more platforms like YouTube, Instagram, and LinkedIn
- Add authentication and saved history
- Improve recommendations with more advanced AI models

## License

This project is licensed under the MIT License.
