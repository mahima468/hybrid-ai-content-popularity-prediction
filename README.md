# HybridAI Content Prediction System

HybridAI is a web-based machine learning project for analyzing digital content. It brings together sentiment analysis, fake engagement detection, popularity prediction, and live monitoring in one dashboard.

The goal of the project is simple: help users understand how content is performing and support better promotion decisions.

## What This Project Does

- Analyze text sentiment
- Detect suspicious or fake engagement
- Predict content popularity
- Show live prediction updates
- Provide simple decision support and what-if analysis
- **NEW**: User authentication with JWT tokens
- **NEW**: AI-powered chat bot for assistance
- Track user analysis history and statistics

## Main Modules

### 1. Sentiment Analysis
This module analyzes text and identifies whether the sentiment is positive, negative, or neutral.

### 2. Fake Engagement Detection
This module checks engagement patterns and helps identify suspicious activity such as fake likes or unusual interaction behavior.

### 3. Popularity Prediction
This module predicts future content performance using features like views, likes, comments, engagement rate, sentiment score, and followers.

### 4. Live Prediction
This module provides real-time prediction support and helps users monitor changing content performance.

### 5. User Authentication
This module provides secure user registration and login with JWT tokens. Users can create accounts, track their analysis history, and access personalized statistics.

### 6. AI Chat Bot
This module provides an intelligent assistant powered by HuggingFace's Qwen2.5-7B model. The chat bot helps users understand sentiment analysis results, get recommendations, and answer questions about content analytics.

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
- **NEW**: SQLite for user data storage
- **NEW**: JWT authentication with python-jose
- **NEW**: HuggingFace InferenceClient for AI chat
- **NEW**: Async support with concurrent.futures

## Project Structure

```text
hybrid-ai-content-popularity-prediction/
|-- backend/
|   |-- app.py
|   |-- routes/
|   |   |-- auth_routes.py      # NEW: User authentication
|   |   |-- chat_routes.py      # NEW: AI chat bot
|   |   |-- sentiment_routes.py
|   |   |-- engagement_routes.py
|   |   |-- prediction_routes.py
|   |-- models/
|   |-- utils/
|   |-- requirements.txt
|   |-- auth.db              # NEW: SQLite user database
|
|-- frontend/
|   |-- src/
|   |   |-- components/
|   |   |   |-- Dashboard.jsx
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
- **NEW**: Authentication endpoints: `http://localhost:8000/api/auth`
- **NEW**: Chat bot endpoint: `http://localhost:8000/api/chat`

## API Endpoints

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `GET /api/auth/me` - Get current user info
- `POST /api/auth/log-analysis` - Log analysis history
- `GET /api/auth/my-stats` - User statistics

### Chat Bot
- `POST /api/chat` - AI chat assistant

### Existing Endpoints
- `POST /api/sentiment/analyze` - Sentiment analysis
- `POST /api/engagement/detect` - Engagement detection
- `POST /api/prediction/predict` - Popularity prediction
- `GET /api/model-status` - Model status
- `GET /api/dashboard-analytics` - Dashboard analytics

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

## Recent Updates (v2.0)

### ✅ New Features Added
- **User Authentication System**: Complete JWT-based authentication with registration, login, and profile management
- **AI Chat Bot**: Intelligent assistant powered by HuggingFace Qwen2.5-7B model
- **User Analytics**: Personalized statistics and analysis history tracking
- **Enhanced Security**: Secure password hashing and token-based authentication
- **Database Integration**: SQLite database for user data persistence
- **Async Processing**: Improved performance with concurrent request handling

### 🔧 Technical Improvements
- Modular route structure with separate authentication and chat modules
- Enhanced error handling and fallback mechanisms
- Batch sentiment analysis support
- Improved engagement detection with proper model loading
- Better API organization and documentation

## Future Improvements

- Add real social media API integration
- Support more platforms like YouTube, Instagram, and LinkedIn
- Enhanced chat bot with more AI models
- Improve recommendations with advanced AI models
- Add file upload and processing capabilities
- Implement real-time notifications

## License

This project is licensed under the MIT License.
