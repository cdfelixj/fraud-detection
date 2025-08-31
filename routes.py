from flask import render_template, request, redirect, url_for, flash, jsonify
from app import app, db
from models import Transaction, Prediction, ModelPerformance, FraudAlert, PredictionFeedback
from data_processor import DataProcessor
from ml_models import FraudDetectionModels
import logging
import os
import numpy as np
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
        
        # Get prediction statistics
        total_predictions = Prediction.query.count()
        predictions_with_ground_truth = db.session.query(Prediction, Transaction).join(
            Transaction, Prediction.transaction_id == Transaction.id
        ).filter(Transaction.actual_class.in_([0, 1])).count()
        
        # Get feedback statistics
        total_feedback = PredictionFeedback.query.count()
        recent_feedback = PredictionFeedback.query.order_by(PredictionFeedback.created_at.desc()).limit(5).all()
        
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
            'active_alerts_count': len(active_alerts),
            'total_predictions': total_predictions,
            'predictions_with_ground_truth': predictions_with_ground_truth,
            'total_feedback': total_feedback,
            'recent_feedback': recent_feedback
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
            'active_alerts_count': 0,
            'total_predictions': 0,
            'predictions_with_ground_truth': 0,
            'total_feedback': 0,
            'recent_feedback': []
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
            
            # Handle potential NaN values
            ensemble_score = float(results['ensemble_scores'][i])
            if np.isnan(ensemble_score):
                ensemble_score = 0.5
            
            confidence_score = float(results['confidence_scores'][i])
            if np.isnan(confidence_score):
                confidence_score = 0.5
            
            prediction.ensemble_prediction = ensemble_score
            prediction.final_prediction = int(results['final_predictions'][i])
            prediction.confidence_score = confidence_score
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
    """Stream real transaction data from database"""
    try:
        import random
        import datetime
        
        # Check if we have real transaction data in database
        total_transactions = Transaction.query.count()
        
        if total_transactions == 0:
            return jsonify({
                'error': 'No transaction data available in database. Please upload data first.'
            }), 404
        
        # Stream real data from database
        # Get a random transaction to simulate real-time processing
        random_offset = random.randint(0, total_transactions - 1)
        transaction = Transaction.query.offset(random_offset).first()
        
        if not transaction:
            return jsonify({
                'error': 'Could not retrieve transaction from database.'
            }), 500
        
        # Use real transaction data
        fraud_score = transaction.actual_class  # 0 or 1
        
        # If we have trained models, get actual prediction
        if ml_models.is_trained:
            try:
                # Prepare features for prediction
                features = np.array([[
                    transaction.time_feature,
                    transaction.v1, transaction.v2, transaction.v3, transaction.v4, transaction.v5,
                    transaction.v6, transaction.v7, transaction.v8, transaction.v9, transaction.v10,
                    transaction.v11, transaction.v12, transaction.v13, transaction.v14, transaction.v15,
                    transaction.v16, transaction.v17, transaction.v18, transaction.v19, transaction.v20,
                    transaction.v21, transaction.v22, transaction.v23, transaction.v24, transaction.v25,
                    transaction.v26, transaction.v27, transaction.v28, transaction.amount
                ]])
                
                # Get model prediction using ensemble method
                prediction_result = ml_models.ensemble_predict(features)
                if prediction_result and 'ensemble_scores' in prediction_result:
                    fraud_score = float(prediction_result['ensemble_scores'][0])
                
            except Exception as e:
                logging.warning(f"Could not get model prediction for simulation: {e}")
                # Fall back to actual class as score
                pass
        
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        
        return jsonify({
            'timestamp': timestamp,
            'amount': round(transaction.amount, 2),
            'fraud_score': round(fraud_score, 3),
            'transaction_id': transaction.id,
            'actual_class': transaction.actual_class,
            'data_source': 'database'
        })
        
    except Exception as e:
        logging.error(f"Error streaming simulation data: {str(e)}")
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
        
        # Handle potential NaN values
        confidence_score = float(results['ensemble_scores'][0])
        if np.isnan(confidence_score):
            logging.warning(f"NaN confidence score for transaction {transaction_id}, using default")
            confidence_score = 0.5  # Default neutral confidence
        
        # Ensure confidence is between 0 and 1
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        return jsonify({
            'transaction_id': transaction_id,
            'prediction': int(results['final_predictions'][0]),
            'confidence': confidence_score
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
        csv_path = 'attached_assets/creditcard.csv'
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
        
        # Create a transaction record for tracking (with unknown actual_class initially)
        transaction = Transaction()
        transaction.time_feature = float(data["time_feature"])
        transaction.amount = float(data["amount"])
        transaction.actual_class = -1  # Mark as manual prediction (unknown ground truth)
        
        # Set all V features
        for i in range(1, 29):
            setattr(transaction, f'v{i}', float(data[f"v{i}"]))
        
        db.session.add(transaction)
        db.session.commit()
        
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
        iso_result = ml_models.predict_isolation_forest(X_scaled)
        log_result = ml_models.predict_logistic(X_scaled)
        
        iso_scores = iso_result[0] if iso_result is not None else None
        log_probs = log_result[0] if log_result is not None else None
        
        # Save prediction to database for tracking
        prediction_id = ml_models.save_prediction(
            transaction_id=transaction.id,
            prediction_results=results,
            iso_scores=iso_scores,
            log_probs=log_probs,
            model_version="1.0"
        )
        
        # Handle potential NaN values
        confidence_score = float(results["ensemble_scores"][0])
        if np.isnan(confidence_score):
            logging.warning("NaN confidence score in manual prediction, using default")
            confidence_score = 0.5
        
        # Ensure confidence is between 0 and 1
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        return jsonify({
            "prediction": int(results["final_predictions"][0]),
            "confidence": confidence_score,
            "isolation_score": float(iso_scores[0]) if iso_scores is not None and not np.isnan(iso_scores[0]) else 0.5,
            "logistic_score": float(log_probs[0]) if log_probs is not None and not np.isnan(log_probs[0]) else 0.5,
            "transaction_id": transaction.id,
            "prediction_id": prediction_id
        })
        
    except Exception as e:
        logging.error(f"Error in manual prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/batch-predict", methods=["POST"])
def api_batch_predict():
    """Generate predictions for existing transactions"""
    try:
        if not ml_models.is_trained:
            if not ml_models.load_models():
                return jsonify({"error": "Models not trained"}), 400
        
        # Get parameters from request
        data = request.json or {}
        limit = data.get('limit', 100)  # Default to 100 transactions
        skip_existing = data.get('skip_existing', True)  # Skip transactions that already have predictions
        
        # Build query for transactions
        query = Transaction.query
        
        if skip_existing:
            # Only get transactions without existing predictions
            predicted_transaction_ids = db.session.query(Prediction.transaction_id).distinct().subquery()
            query = query.filter(~Transaction.id.in_(predicted_transaction_ids))
        
        # Get transactions with known ground truth (actual_class != -1)
        transactions = query.filter(Transaction.actual_class != -1).limit(limit).all()
        
        if not transactions:
            return jsonify({"message": "No transactions found for prediction", "predictions_saved": 0})
        
        # Generate and save predictions
        success = ml_models.predict_and_save_batch(transactions)
        
        if success:
            return jsonify({
                "message": f"Successfully generated predictions for {len(transactions)} transactions",
                "predictions_saved": len(transactions)
            })
        else:
            return jsonify({"error": "Failed to generate batch predictions"}), 500
            
    except Exception as e:
        logging.error(f"Error in batch prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/validation")
def prediction_validation():
    """View prediction validation and accuracy metrics"""
    try:
        # Get predictions with known ground truth
        predictions_with_truth = db.session.query(Prediction, Transaction).join(
            Transaction, Prediction.transaction_id == Transaction.id
        ).filter(Transaction.actual_class.in_([0, 1])).all()
        
        if not predictions_with_truth:
            flash("No predictions with ground truth available. Generate predictions first.", "info")
            return render_template('validation.html', 
                                 validation_stats=None,
                                 recent_predictions=[])
        
        # Calculate validation metrics
        y_true = []
        y_pred = []
        y_scores = []
        validation_details = []
        
        for pred, txn in predictions_with_truth:
            y_true.append(txn.actual_class)
            y_pred.append(pred.final_prediction)
            y_scores.append(pred.ensemble_prediction)
            
            validation_details.append({
                'transaction_id': txn.id,
                'actual_class': txn.actual_class,
                'predicted_class': pred.final_prediction,
                'confidence': pred.confidence_score,
                'ensemble_score': pred.ensemble_prediction,
                'is_correct': txn.actual_class == pred.final_prediction,
                'prediction_time': pred.prediction_time,
                'amount': txn.amount
            })
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        validation_stats = {
            'total_predictions': len(y_true),
            'correct_predictions': sum(1 for t, p in zip(y_true, y_pred) if t == p),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'fraud_detection_rate': recall,  # Same as recall
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0
        }
        
        # Get recent predictions for display (last 20)
        recent_predictions = validation_details[-20:] if len(validation_details) > 20 else validation_details
        recent_predictions.reverse()  # Show most recent first
        
        return render_template('validation.html', 
                             validation_stats=validation_stats,
                             recent_predictions=recent_predictions)
        
    except Exception as e:
        logging.error(f"Error in validation view: {str(e)}")
        flash(f"Error loading validation data: {str(e)}", "danger")
        return redirect(url_for("index"))


@app.route("/api/validate-prediction", methods=["POST"])
def api_validate_prediction():
    """Update ground truth for a transaction to validate prediction"""
    try:
        data = request.json
        transaction_id = data.get('transaction_id')
        actual_class = data.get('actual_class')  # 0 for normal, 1 for fraud
        
        if transaction_id is None or actual_class is None:
            return jsonify({"error": "Missing transaction_id or actual_class"}), 400
        
        if actual_class not in [0, 1]:
            return jsonify({"error": "actual_class must be 0 (normal) or 1 (fraud)"}), 400
        
        # Update transaction with ground truth
        transaction = Transaction.query.get(transaction_id)
        if not transaction:
            return jsonify({"error": "Transaction not found"}), 404
        
        transaction.actual_class = actual_class
        db.session.commit()
        
        # Check if prediction exists and calculate if it was correct
        prediction = Prediction.query.filter_by(transaction_id=transaction_id).first()
        is_correct = None
        if prediction:
            is_correct = prediction.final_prediction == actual_class
        
        return jsonify({
            "message": "Ground truth updated successfully",
            "transaction_id": transaction_id,
            "actual_class": actual_class,
            "prediction_correct": is_correct
        })
        
    except Exception as e:
        logging.error(f"Error validating prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/feedback", methods=["POST"])
def api_submit_feedback():
    """Submit user feedback on a prediction"""
    try:
        data = request.json
        prediction_id = data.get('prediction_id')
        feedback = data.get('feedback')  # 'correct', 'incorrect', 'uncertain'
        reason = data.get('reason', '')
        confidence_rating = data.get('confidence_rating', 3)
        actual_outcome = data.get('actual_outcome')  # 0 or 1 if known
        user_id = data.get('user_id', 'anonymous')
        
        # Validate input
        if not prediction_id or feedback not in ['correct', 'incorrect', 'uncertain']:
            return jsonify({"error": "Missing or invalid prediction_id or feedback"}), 400
        
        if confidence_rating not in range(1, 6):
            confidence_rating = 3  # Default to neutral
        
        # Check if prediction exists
        prediction = Prediction.query.get(prediction_id)
        if not prediction:
            return jsonify({"error": "Prediction not found"}), 404
        
        # Check if feedback already exists
        existing_feedback = PredictionFeedback.query.filter_by(
            prediction_id=prediction_id,
            created_by=user_id
        ).first()
        
        if existing_feedback:
            return jsonify({"error": "Feedback already provided by this user"}), 409
        
        # Save feedback using ML models method
        feedback_data = {
            'feedback': feedback,
            'reason': reason,
            'confidence_rating': confidence_rating,
            'actual_outcome': actual_outcome,
            'user_id': user_id
        }
        
        if ml_models.save_prediction_feedback(prediction_id, feedback_data):
            return jsonify({
                "message": "Feedback submitted successfully",
                "prediction_id": prediction_id,
                "feedback": feedback
            })
        else:
            return jsonify({"error": "Failed to save feedback"}), 500
            
    except Exception as e:
        logging.error(f"Error submitting feedback: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/feedback/<int:prediction_id>", methods=["GET"])
def api_get_feedback(prediction_id):
    """Get feedback for a specific prediction"""
    try:
        feedback_list = PredictionFeedback.query.filter_by(prediction_id=prediction_id).all()
        
        feedback_data = []
        for feedback in feedback_list:
            feedback_data.append({
                'id': feedback.id,
                'feedback': feedback.user_feedback,
                'reason': feedback.feedback_reason,
                'confidence_rating': feedback.confidence_rating,
                'actual_outcome': feedback.actual_outcome,
                'created_at': feedback.created_at.isoformat(),
                'created_by': feedback.created_by
            })
        
        return jsonify({
            "prediction_id": prediction_id,
            "feedback": feedback_data,
            "total_feedback": len(feedback_data)
        })
        
    except Exception as e:
        logging.error(f"Error getting feedback: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/feedback")
def feedback_dashboard():
    """Feedback management dashboard"""
    try:
        # Get feedback statistics
        feedback_stats = ml_models.get_feedback_statistics()
        
        # Get recent feedback
        recent_feedback = db.session.query(
            PredictionFeedback, Prediction, Transaction
        ).join(
            Prediction, PredictionFeedback.prediction_id == Prediction.id
        ).join(
            Transaction, PredictionFeedback.transaction_id == Transaction.id
        ).order_by(
            PredictionFeedback.created_at.desc()
        ).limit(20).all()
        
        # Get problematic predictions
        problematic_predictions = ml_models.get_problematic_predictions(limit=10)
        
        # Format recent feedback for display
        formatted_feedback = []
        for feedback, prediction, transaction in recent_feedback:
            formatted_feedback.append({
                'id': feedback.id,
                'prediction_id': prediction.id,
                'transaction_id': transaction.id,
                'user_feedback': feedback.user_feedback,
                'predicted_class': prediction.final_prediction,
                'actual_class': transaction.actual_class,
                'confidence': prediction.confidence_score,
                'amount': transaction.amount,
                'feedback_reason': feedback.feedback_reason,
                'created_at': feedback.created_at,
                'created_by': feedback.created_by
            })
        
        return render_template('feedback.html',
                             feedback_stats=feedback_stats,
                             recent_feedback=formatted_feedback,
                             problematic_predictions=problematic_predictions)
        
    except Exception as e:
        logging.error(f"Error loading feedback dashboard: {str(e)}")
        flash(f"Error loading feedback dashboard: {str(e)}", "danger")
        return redirect(url_for("index"))


@app.route("/api/retrain-with-feedback", methods=["POST"])
def api_retrain_with_feedback():
    """Retrain models incorporating user feedback"""
    try:
        data = request.json or {}
        use_feedback = data.get('use_feedback', True)
        
        if not ml_models.is_trained:
            return jsonify({"error": "No existing models to retrain"}), 400
        
        # Check if we have enough feedback data
        feedback_count = PredictionFeedback.query.filter(
            PredictionFeedback.actual_outcome.isnot(None)
        ).count()
        
        if use_feedback and feedback_count < 5:
            return jsonify({
                "error": f"Not enough feedback data for retraining. Need at least 5, have {feedback_count}"
            }), 400
        
        # Start retraining
        success = ml_models.retrain_with_feedback(use_feedback=use_feedback)
        
        if success:
            return jsonify({
                "message": "Models successfully retrained with feedback",
                "feedback_used": feedback_count if use_feedback else 0
            })
        else:
            return jsonify({"error": "Failed to retrain models"}), 500
            
    except Exception as e:
        logging.error(f"Error retraining with feedback: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/transaction/<int:transaction_id>/prediction", methods=["GET"])
def api_get_transaction_prediction(transaction_id):
    """Get prediction details for a specific transaction"""
    try:
        # Get the most recent prediction for this transaction
        prediction = Prediction.query.filter_by(transaction_id=transaction_id).order_by(
            Prediction.prediction_time.desc()
        ).first()
        
        if not prediction:
            return jsonify({"error": "No prediction found for this transaction"}), 404
        
        # Get transaction details
        transaction = Transaction.query.get(transaction_id)
        if not transaction:
            return jsonify({"error": "Transaction not found"}), 404
        
        return jsonify({
            "prediction_id": prediction.id,
            "transaction_id": transaction_id,
            "amount": transaction.amount,
            "predicted_class": prediction.final_prediction,
            "confidence": prediction.confidence_score,
            "ensemble_score": prediction.ensemble_prediction,
            "actual_class": transaction.actual_class,
            "prediction_time": prediction.prediction_time.isoformat()
        })
        
    except Exception as e:
        logging.error(f"Error getting transaction prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/transaction/<int:transaction_id>/details", methods=["GET"])
def api_get_transaction_details(transaction_id):
    """Get detailed information about a transaction and its predictions"""
    try:
        # Get transaction
        transaction = Transaction.query.get(transaction_id)
        if not transaction:
            return jsonify({"error": "Transaction not found"}), 404
        
        # Get all predictions for this transaction
        predictions = Prediction.query.filter_by(transaction_id=transaction_id).order_by(
            Prediction.prediction_time.desc()
        ).all()
        
        # Get feedback for this transaction
        feedback = PredictionFeedback.query.filter_by(transaction_id=transaction_id).order_by(
            PredictionFeedback.created_at.desc()
        ).all()
        
        prediction_data = []
        for pred in predictions:
            prediction_data.append({
                'id': pred.id,
                'final_prediction': pred.final_prediction,
                'confidence_score': pred.confidence_score,
                'ensemble_prediction': pred.ensemble_prediction,
                'isolation_forest_score': pred.isolation_forest_score,
                'prediction_time': pred.prediction_time.isoformat(),
                'model_version': pred.model_version
            })
        
        feedback_data = []
        for fb in feedback:
            feedback_data.append({
                'id': fb.id,
                'user_feedback': fb.user_feedback,
                'actual_outcome': fb.actual_outcome,
                'feedback_reason': fb.feedback_reason,
                'confidence_rating': fb.confidence_rating,
                'created_at': fb.created_at.isoformat(),
                'created_by': fb.created_by
            })
        
        return jsonify({
            "transaction_id": transaction_id,
            "amount": transaction.amount,
            "actual_class": transaction.actual_class,
            "time_feature": transaction.time_feature,
            "predictions": prediction_data,
            "feedback": feedback_data,
            "total_predictions": len(predictions),
            "total_feedback": len(feedback)
        })
        
    except Exception as e:
        logging.error(f"Error getting transaction details: {str(e)}")
        return jsonify({"error": str(e)}), 500