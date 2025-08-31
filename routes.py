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
    """Train machine learning models with enhanced parameters"""
    if request.method == 'GET':
        transaction_count = Transaction.query.count()
        latest_performance = ModelPerformance.query.order_by(ModelPerformance.evaluation_date.desc()).first()
        return render_template('train_enhanced.html', 
                             transaction_count=transaction_count,
                             model_trained=ml_models.is_trained,
                             latest_performance=latest_performance)
    
    try:
        # Check if we have data
        if Transaction.query.count() == 0:
            flash('No transaction data available. Please upload data first.', 'warning')
            return redirect(url_for('upload_data'))
        
        action = request.form.get('action', 'quick_train')
        
        if action == 'custom_train':
            # Get custom parameters from form
            custom_params = {
                'contamination': float(request.form.get('contamination', 0.002)),
                'n_estimators_iso': int(request.form.get('n_estimators_iso', 100)),
                'max_samples': request.form.get('max_samples', 'auto'),
                'max_iter': int(request.form.get('max_iter', 1000)),
                'solver': request.form.get('solver', 'liblinear'),
                'penalty': request.form.get('penalty', 'l2'),
                'iso_weight': float(request.form.get('iso_weight', 0.3)),
                'log_weight': float(request.form.get('log_weight', 0.7)),
                'test_size': float(request.form.get('test_size', 0.2)),
                'random_state': int(request.form.get('random_state', 42)),
                'cross_validation': request.form.get('cross_validation') == 'true'
            }
            
            # Update ML models with custom parameters
            ml_models.update_parameters(custom_params)
        
        # Prepare training data
        X_train, X_test, y_train, y_test = data_processor.prepare_training_data()
        
        if X_train is None:
            flash('Error preparing training data', 'danger')
            return redirect(url_for('train_models'))
        
        # Train models
        if ml_models.train_models(X_train, X_test, y_train, y_test):
            flash('Models trained successfully with custom parameters!', 'success')
            logging.info("Model training completed successfully")
        else:
            flash('Error training models', 'danger')
            
        return redirect(url_for('train_models'))
        
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
        
        # Load scaler if not already fitted
        if not hasattr(data_processor.scaler, 'scale_'):
            data_processor.load_scaler()
        
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
        # Return empty data structure instead of error to prevent JS errors
        return jsonify({
            'amounts': [],
            'classes': [],
            'normal_amounts': [],
            'fraud_amounts': [],
            'time_series': [],
            'performance_history': []
        })

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

@app.route('/api/simulation-data')
def simulation_data():
    """Generate simulated real-time transaction data"""
    try:
        import random
        import datetime
        
        # Generate random transaction data
        amount = random.uniform(0.01, 5000.0)
        
        # Simple fraud scoring based on amount and random factors
        fraud_score = 0.0
        
        # Higher amounts are more likely to be fraud
        if amount > 1000:
            fraud_score += 0.3
        if amount > 3000:
            fraud_score += 0.3
            
        # Add some randomness
        fraud_score += random.uniform(0, 0.4)
        fraud_score = min(fraud_score, 1.0)
        
        # Add timestamp
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        
        return jsonify({
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'fraud_score': round(fraud_score, 3)
        })
        
    except Exception as e:
        logging.error(f"Error generating simulation data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/transactions')
def view_transactions():
    """View all transactions with filtering"""
    try:
        # Get filter parameters
        class_filter = request.args.get('class_filter', '')
        min_amount = request.args.get('min_amount', type=float)
        max_amount = request.args.get('max_amount', type=float)
        limit = request.args.get('limit', 100, type=int)
        
        # Build query
        query = Transaction.query
        
        # Apply filters
        if class_filter in ['0', '1']:
            query = query.filter(Transaction.actual_class == int(class_filter))
        
        if min_amount is not None:
            query = query.filter(Transaction.amount >= min_amount)
            
        if max_amount is not None:
            query = query.filter(Transaction.amount <= max_amount)
        
        # Order by ID and limit
        transactions = query.order_by(Transaction.id.desc()).limit(limit).all()
        
        # Calculate statistics
        total_count = Transaction.query.count()
        normal_count = Transaction.query.filter_by(actual_class=0).count()
        fraud_count = Transaction.query.filter_by(actual_class=1).count()
        
        # Average amount
        avg_amount_result = db.session.query(db.func.avg(Transaction.amount)).scalar()
        avg_amount = round(avg_amount_result, 2) if avg_amount_result else 0
        
        return render_template('transactions.html',
                             transactions=transactions,
                             total_count=total_count,
                             normal_count=normal_count,
                             fraud_count=fraud_count,
                             avg_amount=avg_amount,
                             limit=limit)
        
    except Exception as e:
        logging.error(f"Error viewing transactions: {str(e)}")
        flash(f'Error loading transactions: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/api/transaction/<int:transaction_id>')
def get_transaction_details(transaction_id):
    """Get detailed information about a specific transaction"""
    try:
        transaction = Transaction.query.get_or_404(transaction_id)
        
        return jsonify({
            'id': transaction.id,
            'time_feature': transaction.time_feature,
            'amount': transaction.amount,
            'actual_class': transaction.actual_class,
            'v1': transaction.v1,
            'v2': transaction.v2,
            'v3': transaction.v3,
            'v4': transaction.v4,
            'v5': transaction.v5,
            'v6': transaction.v6,
            'v7': transaction.v7,
            'v8': transaction.v8,
            'v9': transaction.v9,
            'v10': transaction.v10,
            'v11': transaction.v11,
            'v12': transaction.v12,
            'v13': transaction.v13,
            'v14': transaction.v14,
            'v15': transaction.v15,
            'v16': transaction.v16,
            'v17': transaction.v17,
            'v18': transaction.v18,
            'v19': transaction.v19,
            'v20': transaction.v20,
            'v21': transaction.v21,
            'v22': transaction.v22,
            'v23': transaction.v23,
            'v24': transaction.v24,
            'v25': transaction.v25,
            'v26': transaction.v26,
            'v27': transaction.v27,
            'v28': transaction.v28
        })
        
    except Exception as e:
        logging.error(f"Error getting transaction details: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict-single/<int:transaction_id>', methods=['POST'])
def predict_single_transaction(transaction_id):
    """Run prediction on a single transaction"""
    try:
        transaction = Transaction.query.get_or_404(transaction_id)
        
        if not ml_models.is_trained:
            if not ml_models.load_models():
                return jsonify({'error': 'Models not trained'}), 400
        
        # Get feature matrix for this transaction
        X, _ = data_processor.get_feature_matrix([transaction])
        if X is None:
            return jsonify({'error': 'Error preparing transaction data'}), 500
        
        # Load scaler if not already fitted
        if not hasattr(data_processor.scaler, 'scale_'):
            data_processor.load_scaler()
        
        # Scale features
        X_scaled = data_processor.scaler.transform(X)
        
        # Get predictions
        results = ml_models.ensemble_predict(X_scaled)
        if results is None:
            return jsonify({'error': 'Error generating prediction'}), 500
        
        return jsonify({
            'transaction_id': transaction_id,
            'prediction': int(results['final_predictions'][0]),
            'confidence': float(results['ensemble_scores'][0])
        })
        
    except Exception as e:
        logging.error(f"Error predicting single transaction: {str(e)}")
        return jsonify({'error': str(e)}), 500

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
        # Return empty but valid structure
        return {
            'amounts': [],
            'classes': [],
            'normal_amounts': [],
            'fraud_amounts': [],
            'time_series': [],
            'performance_history': []
        }

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


@app.route("/predict-manual")
def manual_prediction():
    """Manual transaction prediction page"""
    try:
        latest_performance = ModelPerformance.query.order_by(ModelPerformance.evaluation_date.desc()).first()
        return render_template("predict_manual.html",
                             model_trained=ml_models.is_trained,
                             latest_performance=latest_performance)
        
    except Exception as e:
        logging.error(f"Error loading manual prediction page: {str(e)}")
        flash(f"Error loading page: {str(e)}", "danger")
        return redirect(url_for("index"))

@app.route("/api/predict-manual", methods=["POST"])
def api_predict_manual():
    """API endpoint for manual transaction prediction"""
    try:
        if not ml_models.is_trained:
            if not ml_models.load_models():
                return jsonify({"error": "Models not trained"}), 400
        
        # Get transaction data from request
        data = request.json
        
        # Validate required fields
        required_fields = ["time_feature", "amount"] + [f"v{i}" for i in range(1, 29)]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400
        
        # Prepare feature array
        features = [
            data["time_feature"],
            *[data[f"v{i}"] for i in range(1, 29)],
            data["amount"]
        ]
        
        X = np.array([features])
        
        # Load scaler if not already fitted
        if not hasattr(data_processor.scaler, "scale_"):
            data_processor.load_scaler()
        
        # Scale features
        X_scaled = data_processor.scaler.transform(X)
        
        # Get predictions
        results = ml_models.ensemble_predict(X_scaled)
        if results is None:
            return jsonify({"error": "Error generating prediction"}), 500
        
        # Get individual model scores for detailed analysis
        iso_scores, iso_preds = ml_models.predict_isolation_forest(X_scaled)
        log_probs, log_preds = ml_models.predict_logistic(X_scaled)
        
        return jsonify({
            "prediction": int(results["final_predictions"][0]),
            "confidence": float(results["ensemble_scores"][0]),
            "isolation_score": float(iso_scores[0]) if iso_scores is not None else None,
            "logistic_score": float(log_probs[0]) if log_probs is not None else None
        })
        
    except Exception as e:
        logging.error(f"Error in manual prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500
