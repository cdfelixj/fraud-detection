from app import db
from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, Text
from sqlalchemy.orm import Mapped, mapped_column

class Transaction(db.Model):
    __tablename__ = 'transactions'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    time_feature: Mapped[float] = mapped_column(Float, nullable=False)
    v1: Mapped[float] = mapped_column(Float, nullable=False)
    v2: Mapped[float] = mapped_column(Float, nullable=False)
    v3: Mapped[float] = mapped_column(Float, nullable=False)
    v4: Mapped[float] = mapped_column(Float, nullable=False)
    v5: Mapped[float] = mapped_column(Float, nullable=False)
    v6: Mapped[float] = mapped_column(Float, nullable=False)
    v7: Mapped[float] = mapped_column(Float, nullable=False)
    v8: Mapped[float] = mapped_column(Float, nullable=False)
    v9: Mapped[float] = mapped_column(Float, nullable=False)
    v10: Mapped[float] = mapped_column(Float, nullable=False)
    v11: Mapped[float] = mapped_column(Float, nullable=False)
    v12: Mapped[float] = mapped_column(Float, nullable=False)
    v13: Mapped[float] = mapped_column(Float, nullable=False)
    v14: Mapped[float] = mapped_column(Float, nullable=False)
    v15: Mapped[float] = mapped_column(Float, nullable=False)
    v16: Mapped[float] = mapped_column(Float, nullable=False)
    v17: Mapped[float] = mapped_column(Float, nullable=False)
    v18: Mapped[float] = mapped_column(Float, nullable=False)
    v19: Mapped[float] = mapped_column(Float, nullable=False)
    v20: Mapped[float] = mapped_column(Float, nullable=False)
    v21: Mapped[float] = mapped_column(Float, nullable=False)
    v22: Mapped[float] = mapped_column(Float, nullable=False)
    v23: Mapped[float] = mapped_column(Float, nullable=False)
    v24: Mapped[float] = mapped_column(Float, nullable=False)
    v25: Mapped[float] = mapped_column(Float, nullable=False)
    v26: Mapped[float] = mapped_column(Float, nullable=False)
    v27: Mapped[float] = mapped_column(Float, nullable=False)
    v28: Mapped[float] = mapped_column(Float, nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    actual_class: Mapped[int] = mapped_column(Integer, nullable=False)  # 0=normal, 1=fraud
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class Prediction(db.Model):
    __tablename__ = 'predictions'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[int] = mapped_column(Integer, db.ForeignKey('transactions.id'), nullable=False)
    isolation_forest_score: Mapped[float] = mapped_column(Float)
    ensemble_prediction: Mapped[float] = mapped_column(Float)
    final_prediction: Mapped[int] = mapped_column(Integer)  # 0=normal, 1=fraud
    confidence_score: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(50))
    prediction_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    transaction = db.relationship('Transaction', back_populates='predictions')

# Add relationships to Transaction
Transaction.predictions = db.relationship('Prediction', back_populates='transaction')
Transaction.alerts = db.relationship('FraudAlert', back_populates='transaction')

class ModelPerformance(db.Model):
    __tablename__ = 'model_performance'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    precision_score: Mapped[float] = mapped_column(Float)
    recall_score: Mapped[float] = mapped_column(Float)
    f1_score: Mapped[float] = mapped_column(Float)
    auc_score: Mapped[float] = mapped_column(Float)
    accuracy_score: Mapped[float] = mapped_column(Float)
    evaluation_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class FraudAlert(db.Model):
    __tablename__ = 'fraud_alerts'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[int] = mapped_column(Integer, db.ForeignKey('transactions.id'), nullable=False)
    alert_level: Mapped[str] = mapped_column(String(20), nullable=False)  # HIGH, MEDIUM, LOW
    alert_reason: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    
    transaction = db.relationship('Transaction', back_populates='alerts')

class PredictionFeedback(db.Model):
    __tablename__ = 'prediction_feedback'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(Integer, db.ForeignKey('predictions.id'), nullable=False)
    transaction_id: Mapped[int] = mapped_column(Integer, db.ForeignKey('transactions.id'), nullable=False)
    user_feedback: Mapped[str] = mapped_column(String(20), nullable=False)  # 'correct', 'incorrect', 'uncertain'
    actual_outcome: Mapped[int] = mapped_column(Integer, nullable=False, default=0)  # 0=normal, 1=fraud (required)
    feedback_reason: Mapped[str] = mapped_column(Text)  # Why user thinks it's correct/incorrect
    confidence_rating: Mapped[int] = mapped_column(Integer)  # 1-5 scale of user confidence
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_by: Mapped[str] = mapped_column(String(50))  # User identifier
    
    prediction = db.relationship('Prediction', back_populates='feedback')
    transaction = db.relationship('Transaction', back_populates='feedback')

# Add relationships to existing models
Transaction.feedback = db.relationship('PredictionFeedback', back_populates='transaction')
Prediction.feedback = db.relationship('PredictionFeedback', back_populates='prediction')

