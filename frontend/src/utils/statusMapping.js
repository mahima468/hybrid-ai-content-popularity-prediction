/**
 * Status Mapping Utilities
 * Centralized functions for mapping backend status to frontend display values
 */

/**
 * Map backend model status to frontend display status
 * @param {string} backendStatus - Backend status value
 * @returns {string} Frontend display status
 */
export const mapModelStatus = (backendStatus) => {
  switch (backendStatus) {
    case 'ready':
      return 'Trained';
    case 'not_ready':
      return 'Not Ready';
    case 'error':
      return 'Error';
    case 'loading':
      return 'Loading';
    default:
      return 'Not Ready';
  }
};

/**
 * Map backend model status to boolean trained status
 * @param {string} backendStatus - Backend status value
 * @returns {boolean} Whether model is trained
 */
export const isModelTrained = (backendStatus) => {
  return backendStatus === 'ready';
};

/**
 * Get appropriate accuracy display based on model status
 * @param {string} backendStatus - Backend status value
 * @param {string} defaultAccuracy - Default accuracy when trained
 * @returns {string} Accuracy display string
 */
export const getAccuracyDisplay = (backendStatus, defaultAccuracy = '90.0%') => {
  switch (backendStatus) {
    case 'ready':
      return defaultAccuracy;
    case 'error':
      return 'Error';
    case 'loading':
      return 'Loading...';
    case 'not_ready':
    default:
      return 'Not Ready';
  }
};

/**
 * Get appropriate last trained display based on model status
 * @param {string} backendStatus - Backend status value
 * @returns {string} Last trained display string
 */
export const getLastTrainedDisplay = (backendStatus) => {
  switch (backendStatus) {
    case 'ready':
      return 'Recently';
    case 'error':
      return 'Error';
    case 'loading':
      return 'Loading...';
    case 'not_ready':
    default:
      return 'Never';
  }
};

/**
 * Get status color for Material-UI Chip components
 * @param {string} status - Display status
 * @returns {string} Material-UI color name
 */
export const getStatusColor = (status) => {
  switch (status) {
    case 'Trained':
    case 'Ready':
    case 'Connected':
      return 'success';
    case 'Not Trained':
    case 'Not Ready':
    case 'Loading':
      return 'warning';
    case 'Error':
    case 'Disconnected':
      return 'error';
    default:
      return 'default';
  }
};

/**
 * Process model status response from backend
 * @param {Object} statusResponse - Backend status response
 * @returns {Object} Processed status object for frontend
 */
export const processModelStatus = (statusResponse) => {
  return {
    sentimentModel: mapModelStatus(statusResponse.sentiment_model),
    engagementModel: mapModelStatus(statusResponse.engagement_model),
    popularityModel: mapModelStatus(statusResponse.prediction_model),
    apiStatus: 'Connected'
  };
};

/**
 * Process single model status for components
 * @param {Object} statusResponse - Backend status response
 * @param {string} modelKey - Key for the specific model
 * @param {string} accuracy - Default accuracy for trained model
 * @returns {Object} Processed model status
 */
export const processSingleModelStatus = (statusResponse, modelKey, accuracy) => {
  const backendStatus = statusResponse[modelKey];
  
  return {
    is_trained: isModelTrained(backendStatus),
    accuracy: getAccuracyDisplay(backendStatus, accuracy),
    last_trained: getLastTrainedDisplay(backendStatus)
  };
};

export default {
  mapModelStatus,
  isModelTrained,
  getAccuracyDisplay,
  getLastTrainedDisplay,
  getStatusColor,
  processModelStatus,
  processSingleModelStatus
};
