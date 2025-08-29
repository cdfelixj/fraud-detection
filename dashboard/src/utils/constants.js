// API Endpoints
export const API_ENDPOINTS = {
  STATS: '/api/stats',
  HEALTH: '/api/health',
  TRANSACTIONS: '/api/transactions',
  ALERTS: '/api/alerts',
  MODELS: '/api/models',
};

// Alert Types
export const ALERT_TYPES = {
  HIGH: 'high',
  MEDIUM: 'medium', 
  LOW: 'low',
};

// Risk Levels
export const RISK_LEVELS = {
  HIGH: {
    threshold: 0.7,
    label: 'HIGH',
    color: 'error',
  },
  MEDIUM: {
    threshold: 0.4,
    label: 'MEDIUM', 
    color: 'warning',
  },
  LOW: {
    threshold: 0,
    label: 'LOW',
    color: 'success',
  },
};

// Theme Colors
export const COLORS = {
  PRIMARY: '#1976d2',
  SECONDARY: '#f50057',
  ERROR: '#f44336',
  WARNING: '#ff9800',
  SUCCESS: '#4caf50',
  INFO: '#2196f3',
};

// Chart Colors
export const CHART_COLORS = {
  FRAUD_PROBABILITY: 'rgb(244, 67, 54)',
  FRAUD_PROBABILITY_BG: 'rgba(244, 67, 54, 0.1)',
  TRANSACTION_VOLUME: 'rgb(33, 150, 243)',
  TRANSACTION_VOLUME_BG: 'rgba(33, 150, 243, 0.1)',
};

// Refresh Intervals (in milliseconds)
export const REFRESH_INTERVALS = {
  DASHBOARD: 30000, // 30 seconds
  ALERTS: 15000,    // 15 seconds
  HEALTH: 60000,    // 1 minute
};

// Local Storage Keys
export const STORAGE_KEYS = {
  THEME_PREFERENCE: 'fraud_detection_theme',
  USER_PREFERENCES: 'fraud_detection_preferences',
  ALERT_FILTERS: 'fraud_detection_alert_filters',
};

// Date Formats
export const DATE_FORMATS = {
  DISPLAY: 'YYYY-MM-DD HH:mm:ss',
  SHORT: 'MM/DD/YYYY',
  TIME_ONLY: 'HH:mm:ss',
  CHART_LABEL: 'HH:mm',
};

// Component Sizes
export const SIZES = {
  LOADING_SPINNER: {
    SMALL: 24,
    MEDIUM: 40,
    LARGE: 60,
  },
  CHART_HEIGHT: {
    SMALL: 200,
    MEDIUM: 300,
    LARGE: 400,
  },
};

// Default Values
export const DEFAULTS = {
  STATS: {
    totalTransactions: 0,
    fraudTransactions: 0,
    avgFraudProbability: 0,
    highRiskTransactions: 0,
  },
  HEALTH: {
    api: false,
    database: false,
    cache: false,
  },
  PAGINATION: {
    PAGE_SIZE: 10,
    MAX_PAGE_SIZE: 100,
  },
};

// Error Messages
export const ERROR_MESSAGES = {
  NETWORK_ERROR: 'Network error occurred. Please check your connection.',
  API_ERROR: 'Failed to fetch data from server.',
  VALIDATION_ERROR: 'Please check your input and try again.',
  GENERIC_ERROR: 'An unexpected error occurred.',
};

// Success Messages
export const SUCCESS_MESSAGES = {
  ALERT_ACKNOWLEDGED: 'Alert has been acknowledged successfully.',
  DATA_UPDATED: 'Data has been updated successfully.',
  SETTINGS_SAVED: 'Settings have been saved successfully.',
};

// Regular Expressions
export const REGEX = {
  EMAIL: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
  PHONE: /^\+?[\d\s\-\(\)]+$/,
  TRANSACTION_ID: /^TXN_\d{6}$/,
  ACCOUNT_ID: /^ACC_\d{6}$/,
};

// Feature Flags
export const FEATURE_FLAGS = {
  REAL_TIME_UPDATES: true,
  ADVANCED_ANALYTICS: true,
  EXPORT_FUNCTIONALITY: true,
  DARK_MODE: true,
};
