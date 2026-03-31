# API Routes Summary

## Main Routes (with /api prefix)

### Core API Endpoints
- `GET /api/dashboard-analytics` - Dashboard analytics data
- `GET /api/model-status` - Model loading status
- `POST /api/detect-engagement` - Engagement detection
- `POST /api/virality-score` - Virality score calculation

### Router-based Endpoints
- `POST /api/sentiment/analyze` - Sentiment analysis
- `POST /api/sentiment/analyze/batch` - Batch sentiment analysis
- `POST /api/engagement/detect` - Engagement detection
- `POST /api/engagement/detect/file` - File-based engagement detection
- `POST /api/prediction/predict` - Popularity prediction
- `POST /api/prediction/predict/file` - File-based popularity prediction

### Direct Endpoints
- `POST /predict-popularity` - Direct popularity prediction (no /api prefix)

## Alias Routes (without /api prefix)

For backward compatibility and convenience:

- `GET /dashboard-analytics` → `GET /api/dashboard-analytics`
- `GET /model-status` → `GET /api/model-status`
- `POST /analyze-sentiment` → Sentiment analysis functionality
- `POST /detect-engagement` → `POST /api/detect-engagement`
- `POST /predict-popularity-alias` → `POST /predict-popularity`

## System Endpoints

- `GET /` - Root endpoint (API status)
- `GET /health` - Health check

## Route Structure

```
├── /api/ (main prefix)
│   ├── dashboard-analytics
│   ├── model-status
│   ├── detect-engagement
│   ├── virality-score
│   ├── sentiment/
│   │   ├── analyze
│   │   └── analyze/batch
│   ├── engagement/
│   │   ├── detect
│   │   └── detect/file
│   └── prediction/
│       ├── predict
│       └── predict/file
├── / (alias routes)
│   ├── dashboard-analytics
│   ├── model-status
│   ├── analyze-sentiment
│   ├── detect-engagement
│   └── predict-popularity-alias
├── / (system)
│   └── health
└── / (direct)
    └── predict-popularity
```

## Frontend Integration

Frontend should use these consistent endpoints:

### Recommended (with /api prefix):
- `http://127.0.0.1:8001/api/dashboard-analytics`
- `http://127.0.0.1:8001/api/model-status`
- `http://127.0.0.1:8001/analyze-sentiment` (alias)
- `http://127.0.0.1:8001/detect-engagement` (alias)
- `http://127.0.0.1:8001/predict-popularity` (direct)

### Alternative (all with /api prefix):
- `http://127.0.0.1:8001/api/dashboard-analytics`
- `http://127.0.0.1:8001/api/model-status`
- `http://127.0.0.1:8001/api/sentiment/analyze`
- `http://127.0.0.1:8001/api/engagement/detect`
- `http://127.0.0.1:8001/api/prediction/predict`

## Notes

1. All main routes use `/api` prefix for consistency
2. Alias routes provide backward compatibility
3. Direct route `/predict-popularity` maintained for existing frontend integration
4. Error handling ensures API never crashes
5. All routes return consistent JSON responses
