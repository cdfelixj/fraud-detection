-- Initialize the fraud detection database

-- Create transactions table
CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(255) UNIQUE NOT NULL,
    amount DECIMAL(10, 2) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    merchant_id VARCHAR(255),
    user_id VARCHAR(255),
    merchant_category VARCHAR(100),
    transaction_type VARCHAR(50),
    location VARCHAR(255),
    is_fraud BOOLEAN DEFAULT FALSE,
    fraud_probability DECIMAL(5, 4),
    model_prediction VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(255) REFERENCES transactions(transaction_id),
    model_name VARCHAR(100) NOT NULL,
    prediction_value DECIMAL(5, 4),
    confidence_score DECIMAL(5, 4),
    risk_level VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create alerts table
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(255) REFERENCES transactions(transaction_id),
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) DEFAULT 'medium',
    message TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'active',
    acknowledged_at TIMESTAMP NULL,
    acknowledged_by VARCHAR(255) NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_transactions_fraud ON transactions(is_fraud);
CREATE INDEX IF NOT EXISTS idx_transactions_user ON transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_transaction ON predictions(transaction_id);
CREATE INDEX IF NOT EXISTS idx_alerts_transaction ON alerts(transaction_id);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);

-- Insert some sample data for testing
INSERT INTO transactions (transaction_id, amount, merchant_id, user_id, merchant_category, transaction_type, is_fraud, fraud_probability, model_prediction) VALUES
('T001', 150.00, 'M001', 'U001', 'retail', 'debit', false, 0.15, 'LOW'),
('T002', 2500.00, 'M002', 'U002', 'online', 'credit', true, 0.89, 'HIGH'),
('T003', 45.20, 'M003', 'U001', 'restaurant', 'debit', false, 0.12, 'LOW'),
('T004', 1200.00, 'M004', 'U003', 'retail', 'credit', true, 0.78, 'HIGH'),
('T005', 89.99, 'M005', 'U002', 'gas', 'debit', false, 0.23, 'LOW')
ON CONFLICT (transaction_id) DO NOTHING;

-- Insert corresponding predictions
INSERT INTO predictions (transaction_id, model_name, prediction_value, confidence_score, risk_level) VALUES
('T001', 'isolation_forest', 0.15, 0.85, 'LOW'),
('T002', 'ensemble', 0.89, 0.95, 'HIGH'),
('T003', 'isolation_forest', 0.12, 0.88, 'LOW'),
('T004', 'lstm', 0.78, 0.92, 'HIGH'),
('T005', 'ensemble', 0.23, 0.77, 'LOW')
ON CONFLICT DO NOTHING;

-- Insert sample alerts
INSERT INTO alerts (transaction_id, alert_type, severity, message) VALUES
('T002', 'high_fraud_probability', 'high', 'Transaction flagged with 89% fraud probability'),
('T004', 'unusual_amount', 'medium', 'High-value transaction detected for user')
ON CONFLICT DO NOTHING;
