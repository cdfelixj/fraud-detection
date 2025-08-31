# Overview

This is a real-time credit card fraud detection system built with Flask and machine learning models. The system processes credit card transaction data using multiple ML algorithms (Isolation Forest and Logistic Regression) to detect fraudulent transactions. It features a web dashboard for monitoring fraud statistics, uploading transaction data, training models, and managing fraud alerts. The system is designed to handle highly imbalanced datasets typical in fraud detection scenarios.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Backend Framework
- **Flask Application**: Main web framework with SQLAlchemy ORM for database operations
- **Database Layer**: Uses SQLAlchemy with support for both SQLite (development) and PostgreSQL (production) through environment configuration
- **Session Management**: Configurable session secret with proxy fix for deployment behind reverse proxies

## Machine Learning Pipeline
- **Ensemble Approach**: Combines Isolation Forest (anomaly detection) and Logistic Regression (classification)
- **Data Processing**: Standardized feature scaling and preprocessing for 28 PCA-transformed features plus time and amount
- **Model Training**: Handles class imbalance using balanced class weights and contamination parameters
- **Performance Tracking**: Stores model performance metrics and predictions in database

## Database Schema
- **Transactions Table**: Stores transaction features (V1-V28, time, amount) and actual fraud labels
- **Predictions Table**: Stores model predictions, confidence scores, and ensemble results
- **Model Performance**: Tracks accuracy, precision, recall, F1-score over time
- **Fraud Alerts**: Manages real-time fraud detection alerts with acknowledgment system

## Web Interface
- **Dashboard**: Real-time statistics, fraud rate monitoring, and active alerts display
- **Data Upload**: CSV file processing with validation for required columns
- **Model Training**: Interface for training and evaluating ML models
- **Bootstrap UI**: Dark theme with responsive design and interactive charts

## Data Flow Architecture
- **Input**: CSV upload with transaction data validation
- **Processing**: Feature extraction, scaling, and preprocessing pipeline
- **Prediction**: Ensemble model scoring with confidence calculation
- **Output**: Real-time alerts, dashboard metrics, and historical analysis

# External Dependencies

## Core Framework Dependencies
- **Flask**: Web framework with SQLAlchemy extension for ORM
- **SQLAlchemy**: Database abstraction with DeclarativeBase for modern table definitions
- **Werkzeug**: WSGI utilities including ProxyFix for deployment

## Machine Learning Stack
- **scikit-learn**: Isolation Forest, Logistic Regression, preprocessing, and metrics
- **pandas**: Data manipulation and CSV processing
- **numpy**: Numerical computations and array operations
- **joblib**: Model serialization and persistence

## Frontend Technologies
- **Bootstrap**: Dark theme CSS framework via CDN
- **Font Awesome**: Icon library for UI components
- **Chart.js**: Interactive charts for analytics dashboard
- **Custom CSS/JS**: Enhanced styling and dashboard functionality

## Database Support
- **SQLite**: Default development database (file-based)
- **PostgreSQL**: Production database support through DATABASE_URL environment variable
- **Connection Pooling**: Configured with pool recycling and pre-ping for reliability

## File Processing
- **CSV Validation**: Automatic detection of required columns (Time, V1-V28, Amount, Class)
- **Error Handling**: Comprehensive logging and user feedback for data processing issues
- **File Upload**: Secure file handling with validation and processing pipeline