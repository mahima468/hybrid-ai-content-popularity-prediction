/**
 * API Configuration
 * Centralized API endpoints and configuration
 */

const API_BASE_URL = 'http://127.0.0.1:8000';

export const API_ENDPOINTS = {
  // System
  HEALTH: `${API_BASE_URL}/health`,
  DASHBOARD_ANALYTICS: `${API_BASE_URL}/dashboard-analytics`,
  MODEL_STATUS: `${API_BASE_URL}/models/status`,

  // Sentiment Analysis
  SENTIMENT_ANALYZE: `${API_BASE_URL}/analyze-sentiment`,
  SENTIMENT_BATCH: `${API_BASE_URL}/batch-analyze-sentiment`,

  // Engagement Detection
  ENGAGEMENT_DETECT: `${API_BASE_URL}/detect-engagement`,
  ENGAGEMENT_BATCH: `${API_BASE_URL}/batch-detect-engagement`,

  // Popularity Prediction
  PREDICT_POPULARITY: `${API_BASE_URL}/predict-popularity`,
  PREDICT_POPULARITY_BATCH: `${API_BASE_URL}/batch-predict-popularity`,

  // Model Training
  TRAIN_SENTIMENT: `${API_BASE_URL}/models/train-sentiment`,
  TRAIN_ENGAGEMENT: `${API_BASE_URL}/models/train-engagement`,
  TRAIN_POPULARITY: `${API_BASE_URL}/models/train-popularity`,
};

export const API_CONFIG = {
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000,
};

export const apiCall = async (endpoint, options = {}) => {
  try {
    const config = {
      ...options,
      headers: {
        ...API_CONFIG.headers,
        ...options.headers,
      },
    };
    const response = await fetch(endpoint, config);
    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData?.detail || `API error: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('API call error:', error);
    throw error;
  }
};

export const uploadFile = async (endpoint, file, additionalData = {}) => {
  const formData = new FormData();
  formData.append('file', file);
  Object.keys(additionalData).forEach(key => {
    formData.append(key, additionalData[key]);
  });
  return apiCall(endpoint, {
    method: 'POST',
    body: formData,
    headers: {},
  });
};

export default { API_BASE_URL, API_ENDPOINTS, apiCall, uploadFile };
