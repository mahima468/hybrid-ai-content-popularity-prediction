const API_BASE_URL = 'http://127.0.0.1:8000';

export const API_ENDPOINTS = {
  // System
  HEALTH: `${API_BASE_URL}/health`,
  DASHBOARD_ANALYTICS: `${API_BASE_URL}/dashboard-analytics`,
  MODEL_STATUS: `${API_BASE_URL}/api/model-status`,
  MY_STATS: `${API_BASE_URL}/api/auth/my-stats`,

  // Sentiment Analysis
  SENTIMENT_ANALYZE: `${API_BASE_URL}/analyze-sentiment`,
  SENTIMENT_BATCH: `${API_BASE_URL}/batch-analyze-sentiment`,

  // Engagement Detection
  ENGAGEMENT_DETECT: `${API_BASE_URL}/detect-engagement`,
  ENGAGEMENT_BATCH: `${API_BASE_URL}/api/engagement/detect`,

  // Popularity Prediction
  PREDICT_POPULARITY: `${API_BASE_URL}/api/predict-popularity`,
  PREDICT_POPULARITY_BATCH: `${API_BASE_URL}/api/prediction/predict`,

  // Model Training
  TRAIN_SENTIMENT: `${API_BASE_URL}/api/sentiment/train`,
  TRAIN_ENGAGEMENT: `${API_BASE_URL}/api/engagement/train`,
  TRAIN_POPULARITY: `${API_BASE_URL}/api/prediction/train`,
};

export const API_CONFIG = {
  headers: { 'Content-Type': 'application/json' },
  timeout: 60000,
};

export const apiCall = async (endpoint, options = {}) => {
  try {
    const token = localStorage.getItem('token');
    const config = {
      ...options,
      headers: {
        ...API_CONFIG.headers,
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...options.headers,
      },
    };
    const response = await fetch(endpoint, config);
    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      const detail = errData?.detail;
      const message = Array.isArray(detail)
        ? detail.map(d => d.msg || JSON.stringify(d)).join('; ')
        : detail || `API error: ${response.status}`;
      throw new Error(message);
    }
    return await response.json();
  } catch (error) {
    console.error('API call error:', error);
    throw error;
  }
};

export const logAnalysis = (type, label, confidence) => {
  apiCall(`${API_BASE_URL}/api/auth/log-analysis`, {
    method: 'POST',
    body: JSON.stringify({ type, label, confidence: Number(confidence) || 0 }),
  }).catch(() => {});  // fire-and-forget
};

export const uploadFile = async (endpoint, file, additionalData = {}) => {
  const formData = new FormData();
  formData.append('file', file);
  Object.keys(additionalData).forEach(key => {
    formData.append(key, additionalData[key]);
  });
  return apiCall(endpoint, { method: 'POST', body: formData, headers: {} });
};

export default { API_BASE_URL, API_ENDPOINTS, apiCall, uploadFile };
