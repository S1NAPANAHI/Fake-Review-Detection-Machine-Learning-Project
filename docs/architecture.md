# System Architecture

This document describes the architecture of the Fake Review Detection System, including system components, data flow, and technical design decisions.

## Overview

The Fake Review Detection System is a microservices-based architecture designed for scalability, maintainability, and high performance. It combines multiple machine learning approaches to detect fraudulent reviews using NLP, behavioral analysis, and network-based features.

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Client Layer                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Web UI  │  Mobile App  │  REST API  │  SDK Libraries  │  CLI Tools        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Load Balancer/API Gateway                         │
├─────────────────────────────────────────────────────────────────────────────┤
│          Nginx/HAProxy          │         Rate Limiting                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FastAPI Service                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  Authentication  │  Request Validation  │  Response Formatting              │
│  Rate Limiting   │  Error Handling      │  Metrics Collection               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Core ML Pipeline                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐   │
│  │  Text Features  │  │ Behavioral      │  │    Network Features         │   │
│  │                 │  │ Features        │  │                             │   │
│  │ • TF-IDF        │  │ • Rating Stats  │  │ • User Similarity           │   │
│  │ • Word2Vec      │  │ • Review Length │  │ • Review Similarity         │   │
│  │ • Transformers  │  │ • Temporal      │  │ • Graph Features            │   │
│  │ • Sentiment     │  │ • User Activity │  │ • Community Detection       │   │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘   │
│                                      │                                     │
│                                      ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Feature Engineering                             │   │
│  │                                                                     │   │
│  │ • Feature Scaling    • Feature Selection   • Dimensionality        │   │
│  │ • Feature Encoding   • Feature Validation  • Feature Caching       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                     │
│                                      ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Model Ensemble                                 │   │
│  │                                                                     │   │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │   │
│  │ │   Random    │ │   XGBoost   │ │   Neural    │ │   SVM       │     │   │
│  │ │   Forest    │ │             │ │   Network   │ │             │     │   │
│  │ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘     │   │
│  │                                      │                               │   │
│  │                                      ▼                               │   │
│  │ ┌─────────────────────────────────────────────────────────────────┐ │   │
│  │ │              Ensemble Voting/Averaging                         │ │   │
│  │ └─────────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Data Layer                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐     │
│  │    Redis    │  │ PostgreSQL  │  │    S3/      │  │    MLflow       │     │
│  │   (Cache)   │  │(Metadata)   │  │ File Store  │  │  (Experiments)  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Monitoring & Observability                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐     │
│  │ Prometheus  │  │   Grafana   │  │    ELK      │  │   Jaeger        │     │
│  │ (Metrics)   │  │(Dashboard)  │  │(Logging)    │  │  (Tracing)      │     │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. API Layer

#### FastAPI Service
- **Purpose**: REST API endpoints for model predictions
- **Key Features**:
  - High-performance async request handling
  - Automatic OpenAPI documentation
  - Built-in request validation with Pydantic
  - CORS support for web applications
  - Rate limiting and throttling
- **Scalability**: Horizontal scaling with multiple workers
- **Technology Stack**: FastAPI, Uvicorn, Pydantic

#### Authentication & Authorization
- **Current**: No authentication (development/demo)
- **Production Options**:
  - API Key authentication
  - OAuth2 with JWT tokens
  - Role-based access control (RBAC)

### 2. ML Pipeline

#### Feature Extraction Pipeline

##### Text Features
```
Input Text → Preprocessing → Feature Extraction → Vector Representation
                │                    │
                ▼                    ▼
    • HTML removal           • TF-IDF vectors
    • Text normalization     • Word2Vec embeddings
    • Tokenization          • Transformer embeddings
    • Stop word removal     • N-gram features
    • Lemmatization         • Sentiment scores
```

##### Behavioral Features
```
User/Review Metadata → Feature Engineering → Behavioral Vectors
                              │
                              ▼
                    • Rating statistics
                    • Review length analysis
                    • Temporal patterns
                    • User activity metrics
                    • Verification status
```

##### Network Features
```
User/Review Graph → Graph Analysis → Network Features
                         │
                         ▼
                • User similarity scores
                • Review similarity metrics
                • Graph centrality measures
                • Community detection
                • Anomaly detection
```

#### Model Ensemble
- **Architecture**: Weighted voting ensemble
- **Models**:
  - Random Forest (primary)
  - XGBoost (boosting)
  - Neural Network (deep learning)
  - SVM (support vector)
- **Combination Strategy**: Soft voting with learned weights

### 3. Data Layer

#### Data Storage
```
┌─────────────────────────────────────────────────┐
│                  Data Pipeline                  │
├─────────────────────────────────────────────────┤
│                                                 │
│  Raw Data → Preprocessing → Feature Store       │
│      │            │             │               │
│      ▼            ▼             ▼               │
│  ┌─────────┐  ┌─────────┐  ┌─────────────┐     │
│  │   S3    │  │ Pandas  │  │   Redis     │     │
│  │ Storage │  │Pipeline │  │   Cache     │     │
│  └─────────┘  └─────────┘  └─────────────┘     │
│                                                 │
│  Model Artifacts ← Training ← Processed Data    │
│       │               │            │           │
│       ▼               ▼            ▼           │
│  ┌─────────┐    ┌─────────┐  ┌─────────────┐   │
│  │ MLflow  │    │ Jupyter │  │ PostgreSQL  │   │
│  │Registry │    │Notebooks│  │  Metadata   │   │
│  └─────────┘    └─────────┘  └─────────────┘   │
└─────────────────────────────────────────────────┘
```

#### Caching Strategy
- **L1 Cache**: In-memory model cache (per worker)
- **L2 Cache**: Redis distributed cache (features)
- **L3 Cache**: File system cache (static artifacts)
- **Cache TTL**: 
  - Model predictions: 1 hour
  - Feature vectors: 24 hours
  - Static data: 7 days

### 4. Deployment Architecture

#### Container Orchestration
```
┌─────────────────────────────────────────────────┐
│                Docker Swarm/K8s                 │
├─────────────────────────────────────────────────┤
│                                                 │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│ │   API   │ │   API   │ │   API   │ │   API   │ │
│ │Service 1│ │Service 2│ │Service 3│ │Service N│ │
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘ │
│                                                 │
│ ┌─────────────────────────────────────────────┐ │
│ │            Load Balancer                    │ │
│ │         (Nginx/HAProxy)                     │ │
│ └─────────────────────────────────────────────┘ │
│                                                 │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐             │
│ │  Redis  │ │Database │ │ Storage │             │
│ │ Cluster │ │Cluster  │ │Cluster  │             │
│ └─────────┘ └─────────┘ └─────────┘             │
└─────────────────────────────────────────────────┘
```

#### Health Checks & Monitoring
```
Application → Health Endpoint → Load Balancer
     │              │               │
     ▼              ▼               ▼
Prometheus ← Metrics Endpoint → Grafana
     │
     ▼
Alert Manager → Notification Services
```

## Data Flow

### 1. Training Pipeline

```
Raw Data → Data Validation → Preprocessing → Feature Engineering → Model Training
    │              │              │                │                    │
    ▼              ▼              ▼                ▼                    ▼
Data/raw/    Quality Checks   Data/processed   artifacts/features  artifacts/models
reviews.csv      │           reviews_clean.csv      │                model.pkl
                 ▼                                   ▼                    │
            Data/interim                      Feature Store              ▼
            validation.json                   (Redis Cache)         MLflow Registry
```

### 2. Inference Pipeline

```
API Request → Input Validation → Feature Extraction → Model Prediction → Response
     │              │                    │                   │              │
     ▼              ▼                    ▼                   ▼              ▼
Request Body   Pydantic Model      Feature Pipeline    Ensemble Model  JSON Response
{"text": ...}      │              (Cached Features)    (Cached Model)      │
                   ▼                    │                   │              ▼
               Validated Input          ▼                   ▼         Client Response
                                Feature Vector        Probability Score
```

### 3. Batch Processing Pipeline

```
Batch Request → Request Splitting → Parallel Processing → Result Aggregation
     │                  │                   │                    │
     ▼                  ▼                   ▼                    ▼
Multiple Reviews   Individual Chunks   Worker Pool Results   Combined Response
[review1,...]      [chunk1, chunk2]    [result1, result2]    {"predictions": [...]}
```

## Technology Stack

### Backend Services
- **Web Framework**: FastAPI 0.104+
- **ASGI Server**: Uvicorn
- **Task Queue**: Celery (optional, for batch processing)
- **Message Broker**: Redis

### Machine Learning
- **Core ML**: scikit-learn, XGBoost, LightGBM
- **Deep Learning**: PyTorch/TensorFlow (optional)
- **NLP**: spaCy, NLTK, Transformers
- **Feature Store**: Redis, Apache Parquet
- **Experiment Tracking**: MLflow

### Data Processing
- **Data Manipulation**: Pandas, NumPy
- **Data Validation**: Pydantic, Pandera
- **Pipeline**: Apache Airflow (optional)

### Infrastructure
- **Containerization**: Docker, Docker Compose
- **Orchestration**: Kubernetes/Docker Swarm
- **Load Balancing**: Nginx, HAProxy
- **Service Discovery**: Consul (optional)

### Monitoring & Observability
- **Metrics**: Prometheus
- **Visualization**: Grafana
- **Logging**: ELK Stack (Elasticsearch, Logstash, Kibana)
- **Tracing**: Jaeger (optional)
- **Alerting**: AlertManager, PagerDuty

### Storage
- **Cache**: Redis Cluster
- **Database**: PostgreSQL
- **Object Storage**: AWS S3, MinIO
- **File System**: NFS (optional)

## Design Patterns

### 1. Microservices Pattern
- **Service Separation**: API, ML Pipeline, Data Processing
- **Independent Deployment**: Each service can be deployed independently
- **Technology Diversity**: Different services can use different tech stacks

### 2. CQRS (Command Query Responsibility Segregation)
- **Commands**: Model training, data ingestion
- **Queries**: Prediction requests, model info
- **Separation**: Different optimizations for read/write operations

### 3. Circuit Breaker Pattern
- **Fault Tolerance**: Prevent cascade failures
- **Graceful Degradation**: Fallback responses when services are down
- **Auto-Recovery**: Automatic service restoration

### 4. Repository Pattern
- **Data Access**: Abstracted data access layer
- **Testability**: Easy to mock for unit tests
- **Flexibility**: Can switch between different data sources

## Performance Considerations

### 1. Caching Strategy
```
Request → Cache Check → Cache Hit? → Return Cached Result
    │           │            │
    ▼           ▼            ▼ (No)
Processing  Cache Miss   Compute Result → Update Cache
```

### 2. Load Balancing
- **Algorithm**: Round-robin with health checks
- **Session Affinity**: Stateless design (no sticky sessions)
- **Auto-scaling**: Based on CPU/memory metrics

### 3. Database Optimization
- **Indexing**: Proper indexing on frequently queried fields
- **Connection Pooling**: Efficient database connection management
- **Query Optimization**: Analyze and optimize slow queries

### 4. Model Serving Optimization
- **Model Caching**: Keep models in memory
- **Batch Inference**: Process multiple requests together
- **Feature Caching**: Cache computed features
- **Model Quantization**: Reduce model size for faster inference

## Security Architecture

### 1. Network Security
```
Internet → Firewall → Load Balancer → API Gateway → Services
              │            │             │            │
              ▼            ▼             ▼            ▼
         Block IPs    SSL Termination  Rate Limiting  Internal Network
```

### 2. Data Security
- **Encryption at Rest**: Database and file storage encryption
- **Encryption in Transit**: TLS/SSL for all communications
- **Data Masking**: Sensitive data obfuscation in logs
- **Access Control**: Role-based access to data and services

### 3. Application Security
- **Input Validation**: Strict validation of all inputs
- **Output Encoding**: Prevent injection attacks
- **Authentication**: API keys, OAuth2 tokens
- **Authorization**: Role-based access control

## Scalability Considerations

### 1. Horizontal Scaling
- **API Services**: Scale based on request volume
- **ML Workers**: Scale based on processing queue length
- **Database**: Read replicas and sharding strategies

### 2. Vertical Scaling
- **Resource Allocation**: CPU and memory optimization
- **Model Optimization**: Efficient model architectures
- **Caching**: Reduce computational overhead

### 3. Auto-scaling Policies
```yaml
# Example Kubernetes HPA configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-service
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Disaster Recovery

### 1. Backup Strategy
- **Database Backups**: Daily automated backups with point-in-time recovery
- **Model Artifacts**: Version-controlled storage with backup replicas
- **Configuration**: Infrastructure as code with version control

### 2. High Availability
- **Multi-zone Deployment**: Services deployed across multiple availability zones
- **Database Replication**: Master-slave replication with automatic failover
- **Load Balancer Redundancy**: Multiple load balancers with health checks

### 3. Recovery Procedures
- **RTO (Recovery Time Objective)**: 15 minutes
- **RPO (Recovery Point Objective)**: 1 hour
- **Automated Failover**: Automatic switching to backup systems
- **Manual Override**: Procedures for manual intervention when needed

---

This architecture is designed to be scalable, maintainable, and production-ready while providing high performance and reliability for fake review detection tasks.
