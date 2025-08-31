from flask import render_template, request, redirect, url_for, flash, jsonify
from app import app, db
from models import Transaction, Prediction, ModelPerformance, FraudAlert
from data_processor import DataProcessor
from ml_models import FraudDetectionModels
import logging
import os
from datetime import datetime, timedelta

# Initialize processors
data_processor = DataProcessor()
ml_models = FraudDetectionModels()

@app.route('/')
def index():
    """Main dashboard page"""
    try:
        # Get basic statistics
        total_transactions = Transaction.query.count()
        fraud_transactions = Transaction.query.filter_by(actual_class=1).count()
        normal_transactions = total_transactions - fraud_transactions
        
        # Get recent performance metrics
        latest_performance = ModelPerformance.query.order_by(ModelPerformance.evaluation_date.desc()).first()
        
        # Get active fraud alerts
        active_alerts = FraudAlert.query.filter_by(acknowledged=False).order_by(FraudAlert.created_at.desc()).limit(10).all()
        
        # Calculate fraud rate
        fraud_rate = (fraud_transactions / total_transactions * 100) if total_transactions > 0 else 0
        
        stats = {
            'total_transactions': total_transactions,
            'fraud_transactions': fraud_transactions,
            'normal_transactions': normal_transactions,
            'fraud_rate': fraud_rate,
            'model_trained': ml_models.is_trained,
            'latest_performance': latest_performance,
            'active_alerts_count': len(active_alerts)
        }
        
        return render_template('index.html', stats=stats, alerts=active_alerts)
        
    except Exception as e:
        logging.error(f"Error loading dashboard: {str(e)}")
        flash(f'Error loading dashboard: {str(e)}', 'danger')
        
        # Provide default stats when there's an error
        default_stats = {
            'total_transactions': 0,
            'fraud_transactions': 0,
            'normal_transactions': 0,
            'fraud_rate': 0.0,
            'model_trained': False,
            'latest_performance': None,
            'active_alerts_count': 0
        }
        return render_template('index.html', stats=default_stats, alerts=[])

@app.route('/upload', methods=['GET', 'POST'])
def upload_data():
    """Handle CSV file upload and processing"""
    if request.method == 'GET':
        return render_template('upload.html')
    
    try:
        # Check if file is provided
        if 'file' not in request.files:
            flash('No file provided', 'danger')
            return redirect(url_for('upload_data'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(url_for('upload_data'))
        
        if not file.filename or not file.filename.endswith('.csv'):
            flash('Please upload a CSV file', 'danger')
            return redirect(url_for('upload_data'))
        
        # Save uploaded file temporarily
        temp_path = 'temp_upload.csv'
        file.save(temp_path)
        
        # Process the data
        df = data_processor.load_csv_data(temp_path)
        if df is None:
            flash('Error loading CSV file. Please check the format.', 'danger')
            os.remove(temp_path)
            return redirect(url_for('upload_data'))
        
        # Preprocess data
        processed_df = data_processor.preprocess_data(df)
        if processed_df is None:
            flash('Error processing data. Please check the CSV format.', 'danger')
            os.remove(temp_path)
            return redirect(url_for('upload_data'))
        
        # Save to database
        if data_processor.save_to_database(processed_df):
            flash(f'Successfully uploaded and processed {len(processed_df)} transactions', 'success')
            logging.info(f"Uploaded {len(processed_df)} transactions")
        else:
            flash('Error saving data to database', 'danger')
        
        # Clean up
        os.remove(temp_path)
        
        return redirect(url_for('train_models'))
        
    except Exception as e:
        logging.error(f"Error uploading data: {str(e)}")
        flash(f'Error processing upload: {str(e)}', 'danger')
        return redirect(url_for('upload_data'))

@app.route('/train', methods=['GET', 'POST'])
def train_models():
    """Train machine learning models"""
    if request.method == 'GET':
        transaction_count = Transaction.query.count()
        return render_template('dashboard.html', 
                             page='train', 
                             transaction_count=transaction_count,
                             model_trained=ml_models.is_trained)
    
    try:
        # Check if we have data
        if Transaction.query.count() == 0:
            flash('No transaction data available. Please upload data first.', 'warning')
            return redirect(url_for('upload_data'))
        
        # Prepare training data
        X_train, X_test, y_train, y_test = data_processor.prepare_training_data()
        
        if X_train is None:
            flash('Error preparing training data', 'danger')
            return redirect(url_for('train_models'))
        
        # Train models
        if ml_models.train_models(X_train, X_test, y_train, y_test):
            flash('Models trained successfully!', 'success')
            logging.info("Model training completed successfully")
        else:
            flash('Error training models', 'danger')
            
        return redirect(url_for('dashboard'))
        
    except Exception as e:
        logging.error(f"Error training models: {str(e)}")
        flash(f'Error training models: {str(e)}', 'danger')
        return redirect(url_for('train_models'))

@app.route('/dashboard')
def dashboard():
    """Main analytics dashboard"""
    try:
        # Load models if not already loaded
        if not ml_models.is_trained:
            ml_models.load_models()
        
        # Get transaction statistics
        total_transactions = Transaction.query.count()
        fraud_count = Transaction.query.filter_by(actual_class=1).count()
        normal_count = total_transactions - fraud_count
        
        # Get recent transactions for display
        recent_transactions = Transaction.query.order_by(Transaction.id.desc()).limit(20).all()
        
        # Get model performance
        performance_metrics = ModelPerformance.query.order_by(ModelPerformance.evaluation_date.desc()).first()
        
        # Get predictions if available
        recent_predictions = Prediction.query.order_by(Prediction.prediction_time.desc()).limit(10).all()
        
        # Get fraud alerts
        recent_alerts = FraudAlert.query.order_by(FraudAlert.created_at.desc()).limit(10).all()
        
        # Prepare chart data
        chart_data = prepare_chart_data()
        
        context = {
            'total_transactions': total_transactions,
            'fraud_count': fraud_count,
            'normal_count': normal_count,
            'fraud_rate': (fraud_count / total_transactions * 100) if total_transactions > 0 else 0,
            'recent_transactions': recent_transactions,
            'performance_metrics': performance_metrics,
            'recent_predictions': recent_predictions,
            'recent_alerts': recent_alerts,
            'model_trained': ml_models.is_trained,
            'chart_data': chart_data
        }
        
        return render_template('dashboard.html', **context)
        
    except Exception as e:
        logging.error(f"Error loading dashboard: {str(e)}")
        flash(f'Error loading dashboard: {str(e)}', 'danger')
        return render_template('dashboard.html', model_trained=False)

@app.route('/predict')
def run_predictions():
    """Run predictions on existing transactions"""
    try:
        if not ml_models.is_trained:
            if not ml_models.load_models():
                flash('Models not trained. Please train models first.', 'warning')
                return redirect(url_for('train_models'))
        
        # Get transactions without predictions
        transactions = Transaction.query.outerjoin(Prediction).filter(Prediction.id.is_(None)).limit(100).all()
        
        if not transactions:
            flash('No new transactions to predict', 'info')
            return redirect(url_for('dashboard'))
        
        # Get feature matrix
        X, _ = data_processor.get_feature_matrix(transactions)
        if X is None:
            flash('Error preparing data for prediction', 'danger')
            return redirect(url_for('dashboard'))
        
        # Scale features
        X_scaled = data_processor.scaler.transform(X)
        
        # Get predictions
        results = ml_models.ensemble_predict(X_scaled)
        if results is None:
            flash('Error generating predictions', 'danger')
            return redirect(url_for('dashboard'))
        
        # Save predictions
        predictions_saved = 0
        for i, txn in enumerate(transactions):
            prediction = Prediction()
            prediction.transaction_id = txn.id
            prediction.isolation_forest_score = float(results['isolation_scores'][i])
            prediction.ensemble_prediction = float(results['ensemble_scores'][i])
            prediction.final_prediction = int(results['final_predictions'][i])
            prediction.confidence_score = float(results['confidence_scores'][i])
            prediction.model_version = 'v1.0'
            db.session.add(prediction)
            predictions_saved += 1
        
        db.session.commit()
        
        # Create fraud alerts for positive predictions
        fraud_transactions = [txn for i, txn in enumerate(transactions) if results['final_predictions'][i] == 1]
        if fraud_transactions:
            ml_models.create_fraud_alerts(fraud_transactions, results)
        
        flash(f'Generated predictions for {predictions_saved} transactions', 'success')
        logging.info(f"Generated {predictions_saved} predictions")
        
        return redirect(url_for('dashboard'))
        
    except Exception as e:
        logging.error(f"Error running predictions: {str(e)}")
        flash(f'Error generating predictions: {str(e)}', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/api/chart-data')
def api_chart_data():
    """API endpoint for chart data"""
    try:
        data = prepare_chart_data()
        return jsonify(data)
    except Exception as e:
        logging.error(f"Error getting chart data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/acknowledge-alert/<int:alert_id>')
def acknowledge_alert(alert_id):
    """Mark fraud alert as acknowledged"""
    try:
        alert = FraudAlert.query.get_or_404(alert_id)
        alert.acknowledged = True
        db.session.commit()
        flash('Alert acknowledged', 'success')
    except Exception as e:
        logging.error(f"Error acknowledging alert: {str(e)}")
        flash('Error acknowledging alert', 'danger')
    
    return redirect(url_for('dashboard'))

def prepare_chart_data():
    """Prepare data for charts"""
    try:
        # Transaction amount distribution
        transactions = Transaction.query.all()
        amounts = [t.amount for t in transactions]
        classes = [t.actual_class for t in transactions]
        
        # Amount distribution by class
        normal_amounts = [a for a, c in zip(amounts, classes) if c == 0]
        fraud_amounts = [a for a, c in zip(amounts, classes) if c == 1]
        
        # Time series data (simplified - using transaction order as time)
        time_series = []
        for i, txn in enumerate(transactions[:100]):  # Last 100 transactions
            time_series.append({
                'x': i,
                'y': txn.amount,
                'class': txn.actual_class
            })
        
        # Performance metrics over time
        performance_history = ModelPerformance.query.order_by(ModelPerformance.evaluation_date).all()
        performance_data = []
        for perf in performance_history:
            performance_data.append({
                'date': perf.evaluation_date.strftime('%Y-%m-%d'),
                'precision': perf.precision_score,
                'recall': perf.recall_score,
                'f1': perf.f1_score,
                'accuracy': perf.accuracy_score
            })
        
        return {
            'amounts': amounts,
            'classes': classes,
            'normal_amounts': normal_amounts,
            'fraud_amounts': fraud_amounts,
            'time_series': time_series,
            'performance_history': performance_data
        }
        
    except Exception as e:
        logging.error(f"Error preparing chart data: {str(e)}")
        return {}

# Load initial data when app starts
def load_initial_data():
    """Load initial data if CSV file exists"""
    try:
        csv_path = 'attached_assets/creditcard - Copy_1756648231869.csv'
        if os.path.exists(csv_path) and Transaction.query.count() == 0:
            logging.info("Loading initial dataset...")
            
            df = data_processor.load_csv_data(csv_path)
            if df is not None:
                processed_df = data_processor.preprocess_data(df)
                if processed_df is not None:
                    if data_processor.save_to_database(processed_df):
                        logging.info("Initial dataset loaded successfully")
                    else:
                        logging.error("Failed to save initial dataset")
    except Exception as e:
        logging.error(f"Error loading initial data: {str(e)}")

# Call the function during module import
with app.app_context():
    load_initial_data()
