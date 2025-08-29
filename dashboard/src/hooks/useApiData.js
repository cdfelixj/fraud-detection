import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

/**
 * Custom hook for fetching data from API endpoints
 * @param {string} endpoint - API endpoint to fetch from
 * @param {any} defaultValue - Default value for data
 * @param {number} interval - Refresh interval in milliseconds
 * @returns {object} - { data, loading, error, refetch }
 */
export const useApiData = (endpoint, defaultValue, interval = 30000) => {
  const [data, setData] = useState(defaultValue);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = useCallback(async () => {
    try {
      setError(null);
      const response = await axios.get(endpoint);
      setData(response.data);
    } catch (err) {
      console.error(`Error fetching ${endpoint}:`, err);
      setError(err.message);
      
      // Provide mock data for demo purposes
      if (endpoint === '/api/stats') {
        setData({
          totalTransactions: 1247,
          fraudTransactions: 23,
          avgFraudProbability: 0.12,
          highRiskTransactions: 45,
        });
      } else if (endpoint === '/api/health') {
        setData({
          api: true,
          database: true,
          cache: false,
        });
      }
    } finally {
      setLoading(false);
    }
  }, [endpoint]);

  useEffect(() => {
    fetchData();
    
    if (interval > 0) {
      const intervalId = setInterval(fetchData, interval);
      return () => clearInterval(intervalId);
    }
  }, [fetchData, interval]);

  return { data, loading, error, refetch: fetchData };
};

/**
 * Custom hook for managing local storage
 * @param {string} key - Storage key
 * @param {any} initialValue - Initial value
 * @returns {array} - [value, setValue]
 */
export const useLocalStorage = (key, initialValue) => {
  const [storedValue, setStoredValue] = useState(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.error(`Error reading localStorage key "${key}":`, error);
      return initialValue;
    }
  });

  const setValue = useCallback((value) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      window.localStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.error(`Error setting localStorage key "${key}":`, error);
    }
  }, [key, storedValue]);

  return [storedValue, setValue];
};

/**
 * Custom hook for debouncing values
 * @param {any} value - Value to debounce
 * @param {number} delay - Delay in milliseconds
 * @returns {any} - Debounced value
 */
export const useDebounce = (value, delay) => {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
};

/**
 * Custom hook for managing async operations
 * @returns {object} - { loading, error, execute }
 */
export const useAsync = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const execute = useCallback(async (asyncFunction) => {
    try {
      setLoading(true);
      setError(null);
      const result = await asyncFunction();
      return result;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return { loading, error, execute };
};
