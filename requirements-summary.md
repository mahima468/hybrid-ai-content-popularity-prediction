# Project Dependencies Summary

## Backend Python Dependencies (requirements.txt)

### Core Framework & Web Server
- **fastapi==0.104.1** - Modern, fast web framework for building APIs
- **uvicorn[standard]==0.24.0** - ASGI server for FastAPI with standard extras

### Data Processing & Analysis
- **pandas==2.1.4** - Data manipulation and analysis library
- **numpy==1.24.3** - Numerical computing library

### Machine Learning
- **scikit-learn==1.3.2** - Machine learning algorithms and utilities
- **joblib==1.3.2** - Model serialization and parallel processing
- **xgboost==1.7.6** - Gradient boosting framework
- **lightgbm==4.1.0** - Light gradient boosting machine

### Natural Language Processing
- **nltk==3.8.1** - Natural language toolkit for text processing
- **textblob==0.17.1** - Text processing and sentiment analysis

### Data Visualization
- **matplotlib==3.8.2** - Plotting and visualization library
- **seaborn==0.13.0** - Statistical data visualization
- **wordcloud==1.9.2** - Word cloud generation
- **plotly==5.17.0** - Interactive plotting library

### File Handling & Forms
- **python-multipart==0.0.6** - Multipart form data handling

### Data Validation & Serialization
- **pydantic==2.5.0** - Data validation using Python type annotations

### Authentication & Security
- **python-jose[cryptography]==3.3.0** - JWT token handling
- **passlib[bcrypt]==1.7.4** - Password hashing and authentication

### Environment Configuration
- **python-dotenv==1.0.0** - Environment variable management

### Statistical Analysis
- **scipy==1.11.4** - Scientific computing and statistical functions

### HTTP Client
- **requests==2.31.0** - HTTP library for Python
- **aiohttp==3.9.1** - Async HTTP client/server

### Database Support
- **sqlalchemy==2.0.23** - SQL toolkit and ORM
- **alembic==1.12.1** - Database migration tool

### Testing
- **pytest==7.4.3** - Testing framework
- **pytest-asyncio==0.21.1** - Async testing support
- **httpx==0.25.2** - Async HTTP client for testing

### Development Tools
- **black==23.11.0** - Code formatting
- **flake8==6.1.0** - Code linting
- **mypy==1.7.1** - Static type checking

### Monitoring & Logging
- **structlog==23.2.0** - Structured logging
- **sentry-sdk==1.38.0** - Error tracking and monitoring

---

## Frontend JavaScript Dependencies (package.json)

### Core React Framework
- **react==^18.2.0** - React library for building user interfaces
- **react-dom==^18.2.0** - React DOM renderer
- **react-router-dom==^6.8.0** - Routing for React applications
- **react-scripts==5.0.1** - Build and development scripts

### UI Components & Styling
- **@mui/material==^5.15.0** - Material-UI React components
- **@mui/icons-material==^5.15.0** - Material-UI icons
- **@emotion/react==^11.11.0** - CSS-in-JS styling library
- **@emotion/styled==^11.11.0** - Styled components utility
- **@fontsource/roboto==^5.0.0** - Roboto font for Material-UI

### Data Visualization
- **chart.js==^4.4.0** - Charting library for JavaScript
- **react-chartjs-2==^5.2.0** - React wrapper for Chart.js

### HTTP Client & Utilities
- **axios==^1.6.0** - HTTP client for API requests
- **react-dropzone==^14.2.3** - File upload component with drag & drop
- **prop-types==^15.8.1** - React prop type validation
- **date-fns==^2.30.0** - Date manipulation utilities
- **lodash==^4.17.21** - Utility library for JavaScript

### Development Tools
- **eslint==^8.55.0** - JavaScript linting
- **prettier==^3.1.0** - Code formatting
- **eslint-config-prettier==^9.0.0** - ESLint configuration for Prettier
- **eslint-plugin-prettier==^5.0.1** - Prettier plugin for ESLint

### Testing
- **@testing-library/jest-dom==^6.1.0** - Testing utilities for DOM
- **@testing-library/react==^13.4.0** - React testing utilities
- **@testing-library/user-event==^14.5.0** - User event simulation for testing

---

## Installation Instructions

### Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the React development server
npm start

# Additional scripts
npm run lint          # Run ESLint
npm run lint:fix      # Fix ESLint issues
npm run format        # Format code with Prettier
npm test              # Run tests
npm run build         # Build for production
```

---

## Additional Notes

### Streamlit
While Streamlit was mentioned in the requirements, it's not included in the current setup as we're using a React frontend with FastAPI backend. If you need Streamlit for rapid prototyping or additional interfaces, you can add:
```bash
pip install streamlit
```

### Chart.js Integration
The frontend uses Chart.js with React Chart.js 2 for data visualization. The linting errors related to Chart.js imports are due to case sensitivity in the module names and can be resolved by ensuring consistent import casing across components.

### Development Environment
- **Backend**: Python 3.8+ recommended
- **Frontend**: Node.js 14+ recommended
- **Database**: Optional (SQLAlchemy configured but not required for basic functionality)

### Production Considerations
For production deployment, consider:
- Using a production ASGI server (Gunicorn + Uvicorn)
- Setting up a reverse proxy (Nginx)
- Environment variable management
- Database setup and migrations
- SSL/TLS configuration
