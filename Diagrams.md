# System Diagrams

This document contains comprehensive diagrams for the Fraud Detection System architecture, data flow, and machine learning pipeline.

## System Architecture Diagram

```mermaid
graph TB
    subgraph "Client Layer"
        UI[Web Dashboard]
        API[REST API Client]
        CSV[CSV Upload]
    end
    
    subgraph "Application Layer"
        Flask[Flask Application<br/>Port 5000]
        Routes[Routes Controller]
        ML[ML Models Engine]
        DP[Data Processor]
    end
    
    subgraph "Streaming Layer"
        Producer[Kafka Producer]
        Broker[Kafka Broker<br/>Port 9092]
        Consumer[Kafka Consumer]
        ZK[Zookeeper<br/>Port 2181]
    end
    
    subgraph "Data Layer"
        PG[(PostgreSQL<br/>Port 5432)]
        Redis[(Redis Cache<br/>Port 6379)]
        Models[Saved Models<br/>isolation_forest.pkl<br/>logistic_model.pkl]
        Scaler[Scaler<br/>scaler.pkl]
    end
    
    subgraph "Docker Infrastructure"
        DC[Docker Compose]
        Network[Docker Network]
    end
    
    %% Client connections
    UI --> Flask
    API --> Flask
    CSV --> Flask
    
    %% Application connections
    Flask --> Routes
    Routes --> ML
    Routes --> DP
    Routes --> Producer
    
    %% ML connections
    ML --> Models
    ML --> Scaler
    ML --> PG
    
    %% Data processing
    DP --> PG
    DP --> Scaler
    
    %% Kafka flow
    Producer --> Broker
    Broker --> Consumer
    Consumer --> ML
    Consumer --> PG
    ZK --> Broker
    
    %% Data storage
    Flask --> PG
    Flask --> Redis
    
    %% Docker orchestration
    DC --> Flask
    DC --> Broker
    DC --> ZK
    DC --> PG
    DC --> Redis
    
    %% Styling
    classDef client fill:#e1f5fe
    classDef app fill:#f3e5f5
    classDef stream fill:#fff3e0
    classDef data fill:#e8f5e8
    classDef docker fill:#fce4ec
    
    class UI,API,CSV client
    class Flask,Routes,ML,DP app
    class Producer,Broker,Consumer,ZK stream
    class PG,Redis,Models,Scaler data
    class DC,Network docker
```

## Data Flow Diagram

```mermaid
flowchart TD
    subgraph "Data Input Sources"
        WebForm[Web Form Input]
        CSVFile[CSV File Upload]
        KafkaStream[Kafka Stream]
        ManualEntry[Manual Transaction Entry]
    end
    
    subgraph "Data Processing Pipeline"
        Validation[Data Validation]
        Preprocessing[Data Preprocessing<br/>- Feature Engineering<br/>- Scaling<br/>- Normalization]
        FeatureStore[Feature Store]
    end
    
    subgraph "ML Processing Engine"
        Ensemble[Ensemble Models]
        IsoForest[Isolation Forest<br/>Anomaly Detection]
        LogReg[Logistic Regression<br/>Classification]
        Scoring[Ensemble Scoring<br/>Weighted Combination]
    end
    
    subgraph "Decision Engine"
        Threshold[Threshold Application]
        Confidence[Confidence Scoring]
        Alert[Alert Generation]
    end
    
    subgraph "Data Storage"
        TransDB[(Transactions Table)]
        PredDB[(Predictions Table)]
        AlertDB[(Fraud Alerts Table)]
        FeedDB[(Feedback Table)]
        PerfDB[(Model Performance)]
    end
    
    subgraph "Output & Monitoring"
        Dashboard[Real-time Dashboard]
        Alerts[Fraud Alerts]
        Reports[Performance Reports]
        Feedback[User Feedback Loop]
    end
    
    %% Input flow
    WebForm --> Validation
    CSVFile --> Validation
    KafkaStream --> Validation
    ManualEntry --> Validation
    
    %% Processing flow
    Validation --> Preprocessing
    Preprocessing --> FeatureStore
    FeatureStore --> Ensemble
    
    %% ML flow
    Ensemble --> IsoForest
    Ensemble --> LogReg
    IsoForest --> Scoring
    LogReg --> Scoring
    
    %% Decision flow
    Scoring --> Threshold
    Threshold --> Confidence
    Confidence --> Alert
    
    %% Storage flow
    Validation --> TransDB
    Scoring --> PredDB
    Alert --> AlertDB
    Feedback --> FeedDB
    Ensemble --> PerfDB
    
    %% Output flow
    PredDB --> Dashboard
    AlertDB --> Alerts
    PerfDB --> Reports
    Dashboard --> Feedback
    
    %% Feedback loop
    Feedback --> Ensemble
    
    %% Data flow styling
    classDef input fill:#e3f2fd
    classDef process fill:#f1f8e9
    classDef ml fill:#fff3e0
    classDef decision fill:#fce4ec
    classDef storage fill:#f3e5f5
    classDef output fill:#e0f2f1
    
    class WebForm,CSVFile,KafkaStream,ManualEntry input
    class Validation,Preprocessing,FeatureStore process
    class Ensemble,IsoForest,LogReg,Scoring ml
    class Threshold,Confidence,Alert decision
    class TransDB,PredDB,AlertDB,FeedDB,PerfDB storage
    class Dashboard,Alerts,Reports,Feedback output
```

## ML Pipeline Diagram

```mermaid
flowchart TD
    subgraph "Data Ingestion"
        RawData[Raw Transaction Data<br/>29 PCA Features + Amount + Time]
        DataValidation[Data Validation<br/>- Missing value check<br/>- Schema validation<br/>- Range validation]
    end
    
    subgraph "Feature Engineering"
        TimeFeature[Time Feature Engineering<br/>- Temporal patterns<br/>- Cyclic encoding]
        AmountScaling[Amount Scaling<br/>- StandardScaler<br/>- Outlier handling]
        FeatureSelection[Feature Selection<br/>- PCA features V1-V28<br/>- Amount normalization]
    end
    
    subgraph "Model Training Pipeline"
        DataSplit[Train/Test Split<br/>80/20 ratio]
        
        subgraph "Isolation Forest Training"
            IsoTrain[Isolation Forest<br/>- contamination=0.002<br/>- n_estimators=200<br/>- max_samples=auto]
            IsoValidation[Anomaly Score<br/>Validation]
        end
        
        subgraph "Logistic Regression Training"
            ClassWeight[Class Weight<br/>Calculation<br/>Handle imbalance]
            LogTrain[Logistic Regression<br/>- max_iter=1000<br/>- solver=liblinear<br/>- L2 penalty]
            LogValidation[Classification<br/>Validation]
        end
        
        EnsembleWeights[Ensemble Weight<br/>Optimization<br/>ISO: 0.4, LOG: 0.6]
    end
    
    subgraph "Model Evaluation"
        MetricsCalc[Performance Metrics<br/>- Precision/Recall<br/>- F1-Score<br/>- ROC-AUC<br/>- Accuracy]
        CrossVal[Cross Validation<br/>Optional]
        ModelSave[Model Persistence<br/>- isolation_forest.pkl<br/>- logistic_model.pkl<br/>- scaler.pkl]
    end
    
    subgraph "Inference Pipeline"
        NewTransaction[New Transaction]
        FeaturePrep[Feature Preparation<br/>- Apply scaler<br/>- Feature alignment]
        
        subgraph "Ensemble Prediction"
            IsoPredict[Isolation Forest<br/>Anomaly Score]
            LogPredict[Logistic Regression<br/>Probability Score]
            WeightedScore[Weighted Ensemble<br/>Final Score Calculation]
        end
        
        ThresholdApply[Threshold Application<br/>Binary Classification]
        ConfidenceCalc[Confidence Score<br/>Calculation]
        AlertGen[Alert Generation<br/>Based on risk level]
    end
    
    subgraph "Feedback Loop"
        UserFeedback[User Feedback<br/>- Correct/Incorrect<br/>- Confidence rating<br/>- Reasoning]
        ModelUpdate[Model Retraining<br/>- Incorporate feedback<br/>- Performance monitoring]
        ContinuousLearning[Continuous Learning<br/>- Model drift detection<br/>- Auto-retraining triggers]
    end
    
    %% Data flow
    RawData --> DataValidation
    DataValidation --> TimeFeature
    DataValidation --> AmountScaling
    DataValidation --> FeatureSelection
    
    %% Training flow
    TimeFeature --> DataSplit
    AmountScaling --> DataSplit
    FeatureSelection --> DataSplit
    
    DataSplit --> IsoTrain
    DataSplit --> ClassWeight
    ClassWeight --> LogTrain
    
    IsoTrain --> IsoValidation
    LogTrain --> LogValidation
    
    IsoValidation --> EnsembleWeights
    LogValidation --> EnsembleWeights
    
    EnsembleWeights --> MetricsCalc
    MetricsCalc --> CrossVal
    CrossVal --> ModelSave
    
    %% Inference flow
    NewTransaction --> FeaturePrep
    FeaturePrep --> IsoPredict
    FeaturePrep --> LogPredict
    
    IsoPredict --> WeightedScore
    LogPredict --> WeightedScore
    WeightedScore --> ThresholdApply
    ThresholdApply --> ConfidenceCalc
    ConfidenceCalc --> AlertGen
    
    %% Feedback flow
    AlertGen --> UserFeedback
    UserFeedback --> ModelUpdate
    ModelUpdate --> ContinuousLearning
    ContinuousLearning --> ModelSave
    
    %% Model loading for inference
    ModelSave -.-> IsoPredict
    ModelSave -.-> LogPredict
    
    %% Styling
    classDef ingestion fill:#e3f2fd
    classDef feature fill:#f1f8e9
    classDef training fill:#fff3e0
    classDef evaluation fill:#fce4ec
    classDef inference fill:#f3e5f5
    classDef feedback fill:#e0f2f1
    
    class RawData,DataValidation ingestion
    class TimeFeature,AmountScaling,FeatureSelection feature
    class DataSplit,IsoTrain,LogTrain,ClassWeight,EnsembleWeights training
    class MetricsCalc,CrossVal,ModelSave evaluation
    class NewTransaction,FeaturePrep,IsoPredict,LogPredict,WeightedScore,ThresholdApply,ConfidenceCalc,AlertGen inference
    class UserFeedback,ModelUpdate,ContinuousLearning feedback
```

## Real-time Processing Flow

```mermaid
sequenceDiagram
    participant Client
    participant Flask as Flask App
    participant Kafka as Kafka Broker
    participant Consumer as Kafka Consumer
    participant ML as ML Engine
    participant DB as PostgreSQL
    participant Redis as Redis Cache
    
    Note over Client,Redis: Real-time Transaction Processing
    
    Client->>Flask: Submit Transaction
    Flask->>DB: Store Transaction
    Flask->>Kafka: Publish to transactions topic
    Flask-->>Client: Transaction ID
    
    Kafka->>Consumer: Transaction Message
    Consumer->>ML: Process Transaction
    
    Note over ML: Ensemble Processing
    ML->>ML: Isolation Forest Scoring
    ML->>ML: Logistic Regression Scoring
    ML->>ML: Weighted Ensemble Calculation
    
    ML->>DB: Store Prediction
    
    alt High Risk Transaction
        ML->>DB: Create Fraud Alert
        ML->>Kafka: Publish Alert
        Consumer->>Redis: Cache Alert
        Consumer-->>Flask: Alert Notification
        Flask-->>Client: Real-time Alert
    end
    
    Consumer->>DB: Update Processing Status
    
    Note over Client,Redis: Dashboard Updates
    Client->>Flask: Request Dashboard Data
    Flask->>Redis: Check Cache
    alt Cache Hit
        Redis-->>Flask: Cached Data
    else Cache Miss
        Flask->>DB: Query Fresh Data
        DB-->>Flask: Data
        Flask->>Redis: Update Cache
    end
    Flask-->>Client: Dashboard Update
```

## Model Training Flow

```mermaid
flowchart TD
    Start([Training Request]) --> LoadData[Load Training Data<br/>CSV/Database]
    
    LoadData --> ValidateData{Data Validation<br/>Check schema & quality}
    ValidateData -->|Invalid| DataError[Data Error<br/>Return validation errors]
    ValidateData -->|Valid| PreprocessData[Data Preprocessing<br/>- Clean missing values<br/>- Feature scaling<br/>- Train/test split]
    
    PreprocessData --> CheckBalance{Check Class Balance<br/>Fraud vs Normal ratio}
    CheckBalance --> ClassWeights[Calculate Class Weights<br/>Handle imbalanced data]
    
    ClassWeights --> ParallelTrain{Parallel Model Training}
    
    ParallelTrain --> IsoForestTrain[Isolation Forest Training<br/>- Unsupervised anomaly detection<br/>- contamination=0.002<br/>- n_estimators=200]
    ParallelTrain --> LogRegTrain[Logistic Regression Training<br/>- Supervised classification<br/>- Class weight balancing<br/>- L2 regularization]
    
    IsoForestTrain --> IsoEval[Isolation Forest Evaluation<br/>- Anomaly scores<br/>- Threshold optimization]
    LogRegTrain --> LogEval[Logistic Regression Evaluation<br/>- Probability scores<br/>- ROC-AUC analysis]
    
    IsoEval --> EnsembleOpt[Ensemble Optimization<br/>- Weight tuning<br/>- Performance validation<br/>- Cross-validation]
    LogEval --> EnsembleOpt
    
    EnsembleOpt --> FinalEval[Final Model Evaluation<br/>- Precision: Fraud detection accuracy<br/>- Recall: Fraud coverage<br/>- F1-Score: Harmonic mean<br/>- ROC-AUC: Overall performance]
    
    FinalEval --> SaveModels[Save Trained Models<br/>- isolation_forest.pkl<br/>- logistic_model.pkl<br/>- scaler.pkl]
    
    SaveModels --> UpdateDB[Update Model Performance<br/>Store metrics in database]
    UpdateDB --> NotifyComplete[Training Complete<br/>Models ready for inference]
    
    DataError --> End([End])
    NotifyComplete --> End
    
    %% Styling
    classDef process fill:#e3f2fd
    classDef decision fill:#fff3e0
    classDef ml fill:#f3e5f5
    classDef storage fill:#e8f5e8
    classDef error fill:#ffebee
    classDef start fill:#e0f2f1
    
    class Start,End start
    class LoadData,PreprocessData,ClassWeights,SaveModels,UpdateDB,NotifyComplete process
    class ValidateData,CheckBalance,ParallelTrain decision
    class IsoForestTrain,LogRegTrain,IsoEval,LogEval,EnsembleOpt,FinalEval ml
    class DataError error
```

## Database Schema Diagram

```mermaid
erDiagram
    TRANSACTIONS {
        int id PK
        float time_feature
        float v1_to_v28
        float amount
        int actual_class
        datetime created_at
    }
    
    PREDICTIONS {
        int id PK
        int transaction_id FK
        float isolation_forest_score
        float ensemble_prediction
        int final_prediction
        float confidence_score
        string model_version
        datetime prediction_time
    }
    
    FRAUD_ALERTS {
        int id PK
        int transaction_id FK
        string alert_level
        text alert_reason
        datetime created_at
        boolean acknowledged
    }
    
    PREDICTION_FEEDBACK {
        int id PK
        int prediction_id FK
        int transaction_id FK
        string user_feedback
        int actual_outcome
        text feedback_reason
        int confidence_rating
        datetime created_at
        string created_by
    }
    
    MODEL_PERFORMANCE {
        int id PK
        string model_name
        float precision_score
        float recall_score
        float f1_score
        float auc_score
        float accuracy_score
        datetime evaluation_date
    }
    
    TRANSACTIONS ||--o{ PREDICTIONS : "has"
    TRANSACTIONS ||--o{ FRAUD_ALERTS : "triggers"
    TRANSACTIONS ||--o{ PREDICTION_FEEDBACK : "receives"
    PREDICTIONS ||--o{ PREDICTION_FEEDBACK : "evaluated_by"
```

## Kafka Event Flow

```mermaid
flowchart LR
    subgraph "Kafka Topics"
        TransTopic[fraud-detection-transactions<br/>Raw transaction data]
        PredTopic[fraud-detection-predictions<br/>ML prediction results]
        AlertTopic[fraud-detection-alerts<br/>High-risk alerts]
        FeedTopic[fraud-detection-feedback<br/>User feedback events]
    end
    
    subgraph "Producers"
        FlaskProd[Flask Application<br/>Transaction Producer]
        MLProd[ML Engine<br/>Prediction Producer]
        AlertProd[Alert System<br/>Alert Producer]
        WebProd[Web Interface<br/>Feedback Producer]
    end
    
    subgraph "Consumers"
        MLConsumer[ML Consumer<br/>Real-time Processing]
        DashConsumer[Dashboard Consumer<br/>UI Updates]
        AlertConsumer[Alert Consumer<br/>Notification System]
        AnalyticsConsumer[Analytics Consumer<br/>Performance Monitoring]
    end
    
    %% Producer to Topic flow
    FlaskProd --> TransTopic
    MLProd --> PredTopic
    AlertProd --> AlertTopic
    WebProd --> FeedTopic
    
    %% Topic to Consumer flow
    TransTopic --> MLConsumer
    PredTopic --> DashConsumer
    AlertTopic --> AlertConsumer
    FeedTopic --> AnalyticsConsumer
    
    %% Cross-topic consumption
    TransTopic --> DashConsumer
    AlertTopic --> DashConsumer
    PredTopic --> AnalyticsConsumer
    
    classDef topic fill:#e1f5fe
    classDef producer fill:#f3e5f5
    classDef consumer fill:#e8f5e8
    
    class TransTopic,PredTopic,AlertTopic,FeedTopic topic
    class FlaskProd,MLProd,AlertProd,WebProd producer
    class MLConsumer,DashConsumer,AlertConsumer,AnalyticsConsumer consumer
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Docker Compose Environment"
        subgraph "Application Services"
            WebApp[Flask Web App<br/>fraud-detection-app<br/>Port: 5000]
            KafkaConsumerSvc[Kafka Consumer Service<br/>Background processing]
        end
        
        subgraph "Message Streaming"
            Zookeeper[Apache Zookeeper<br/>Port: 2181<br/>Coordination service]
            KafkaBroker[Apache Kafka<br/>Port: 9092<br/>Message broker]
        end
        
        subgraph "Data Storage"
            PostgresDB[PostgreSQL Database<br/>Port: 5432<br/>Primary data store]
            RedisCache[Redis Cache<br/>Port: 6379<br/>Session & cache store]
        end
        
        subgraph "Persistent Storage"
            PostgresVol[postgres_data volume]
            ModelFiles[Model Files<br/>- isolation_forest.pkl<br/>- logistic_model.pkl<br/>- scaler.pkl]
        end
    end
    
    subgraph "External Access"
        Browser[Web Browser<br/>localhost:5000]
        APIClient[API Client<br/>REST endpoints]
        KafkaClient[External Kafka Client<br/>localhost:9092]
    end
    
    %% Service dependencies
    KafkaBroker --> Zookeeper
    WebApp --> PostgresDB
    WebApp --> RedisCache
    WebApp --> KafkaBroker
    KafkaConsumerSvc --> KafkaBroker
    KafkaConsumerSvc --> PostgresDB
    
    %% Storage connections
    PostgresDB --> PostgresVol
    WebApp --> ModelFiles
    KafkaConsumerSvc --> ModelFiles
    
    %% External connections
    Browser --> WebApp
    APIClient --> WebApp
    KafkaClient --> KafkaBroker
    
    %% Health checks and monitoring
    WebApp -.->|Health Check| PostgresDB
    WebApp -.->|Health Check| RedisCache
    KafkaConsumerSvc -.->|Health Check| KafkaBroker
    
    classDef webapp fill:#e3f2fd
    classDef messaging fill:#fff3e0
    classDef database fill:#e8f5e8
    classDef storage fill:#f3e5f5
    classDef external fill:#fce4ec
    
    class WebApp,KafkaConsumerSvc webapp
    class Zookeeper,KafkaBroker messaging
    class PostgresDB,RedisCache database
    class PostgresVol,ModelFiles storage
    class Browser,APIClient,KafkaClient external
```

## API Flow Diagram

```mermaid
sequenceDiagram
    participant Client
    participant Routes as Flask Routes
    participant DataProc as Data Processor
    participant ML as ML Models
    participant DB as Database
    participant Kafka as Kafka Producer
    
    Note over Client,Kafka: Manual Prediction Flow
    
    Client->>Routes: POST /predict_manual
    Routes->>DataProc: Validate input data
    DataProc-->>Routes: Validation result
    
    alt Valid Data
        Routes->>ML: Load trained models
        ML-->>Routes: Models loaded
        Routes->>ML: Predict fraud
        ML->>ML: Isolation Forest scoring
        ML->>ML: Logistic Regression scoring
        ML->>ML: Ensemble calculation
        ML-->>Routes: Prediction result
        
        Routes->>DB: Store transaction
        Routes->>DB: Store prediction
        Routes->>Kafka: Send to stream (optional)
        Routes-->>Client: Prediction response
    else Invalid Data
        Routes-->>Client: Validation error
    end
    
    Note over Client,Kafka: CSV Upload Flow
    
    Client->>Routes: POST /upload_csv
    Routes->>DataProc: Process CSV file
    DataProc->>DataProc: Validate schema
    DataProc->>DataProc: Preprocess data
    DataProc->>DB: Batch insert transactions
    
    loop For each transaction
        Routes->>ML: Predict fraud
        ML-->>Routes: Prediction
        Routes->>DB: Store prediction
    end
    
    Routes-->>Client: Upload complete
    
    Note over Client,Kafka: Model Training Flow
    
    Client->>Routes: POST /train_models
    Routes->>DB: Load training data
    DB-->>Routes: Training dataset
    Routes->>ML: Train ensemble models
    
    ML->>ML: Train Isolation Forest
    ML->>ML: Train Logistic Regression
    ML->>ML: Optimize ensemble weights
    ML->>ML: Evaluate performance
    ML->>ML: Save trained models
    
    ML-->>Routes: Training complete
    Routes->>DB: Store performance metrics
    Routes-->>Client: Training result
```

## System Monitoring Flow

```mermaid
flowchart TD
    subgraph "Data Collection"
        AppMetrics[Application Metrics<br/>- Request count<br/>- Response time<br/>- Error rate]
        MLMetrics[ML Model Metrics<br/>- Prediction accuracy<br/>- Model drift<br/>- Feature importance]
        KafkaMetrics[Kafka Metrics<br/>- Message throughput<br/>- Consumer lag<br/>- Topic size]
        DBMetrics[Database Metrics<br/>- Query performance<br/>- Connection pool<br/>- Storage usage]
    end
    
    subgraph "Performance Monitoring"
        Dashboard[Real-time Dashboard<br/>- Transaction volume<br/>- Fraud rate trends<br/>- Model performance]
        Alerts[Alert System<br/>- High fraud rate<br/>- Model degradation<br/>- System errors]
        Reports[Performance Reports<br/>- Daily summaries<br/>- Model evaluation<br/>- Trend analysis]
    end
    
    subgraph "Model Quality Monitoring"
        DriftDetection[Model Drift Detection<br/>- Feature distribution changes<br/>- Performance degradation]
        FeedbackAnalysis[Feedback Analysis<br/>- User corrections<br/>- Prediction accuracy]
        RetrainingTrigger[Retraining Trigger<br/>- Performance threshold<br/>- Data volume threshold]
    end
    
    AppMetrics --> Dashboard
    MLMetrics --> Dashboard
    KafkaMetrics --> Dashboard
    DBMetrics --> Dashboard
    
    Dashboard --> Alerts
    Dashboard --> Reports
    
    MLMetrics --> DriftDetection
    Reports --> FeedbackAnalysis
    
    DriftDetection --> RetrainingTrigger
    FeedbackAnalysis --> RetrainingTrigger
    
    classDef collection fill:#e3f2fd
    classDef monitoring fill:#f1f8e9
    classDef quality fill:#fff3e0
    
    class AppMetrics,MLMetrics,KafkaMetrics,DBMetrics collection
    class Dashboard,Alerts,Reports monitoring
    class DriftDetection,FeedbackAnalysis,RetrainingTrigger quality
```
