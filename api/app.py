"""
FastAPI application for Fake Review Detection System.

This module provides REST API endpoints for the fake review detection service,
including prediction, health check, and metrics endpoints with CORS support,
rate limiting, and Prometheus monitoring.
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, validator
import numpy as np

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    from prometheus_client import CollectorRegistry, multiprocess, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    RATE_LIMITING_AVAILABLE = True
except ImportError:
    RATE_LIMITING_AVAILABLE = False

# Import our modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from deployment import get_detector, initialize_detector, FakeReviewDetector
from src import utils

# Initialize logging
logger = utils.setup_logging(__name__)

# Initialize rate limiter if available
if RATE_LIMITING_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)
else:
    limiter = None

# Initialize Prometheus metrics if available
if PROMETHEUS_AVAILABLE:
    # Custom metrics registry for multiprocessing
    registry = CollectorRegistry()
    
    # Define metrics
    REQUEST_COUNT = Counter(
        'http_requests_total', 
        'Total HTTP requests',
        ['method', 'endpoint', 'status_code'],
        registry=registry
    )
    REQUEST_DURATION = Histogram(
        'http_request_duration_seconds',
        'HTTP request duration in seconds',
        ['method', 'endpoint'],
        registry=registry
    )
    PREDICTION_COUNT = Counter(
        'predictions_total',
        'Total predictions made',
        ['status'],
        registry=registry
    )
    PREDICTION_DURATION = Histogram(
        'prediction_duration_seconds',
        'Prediction processing time in seconds',
        registry=registry
    )
    PREDICTION_PROBABILITY = Histogram(
        'prediction_probability',
        'Distribution of prediction probabilities',
        buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        registry=registry
    )
    MODEL_HEALTH = Gauge(
        'model_health_status',
        'Model health status (1=healthy, 0=unhealthy)',
        registry=registry
    )
    ACTIVE_REQUESTS = Gauge(
        'active_requests',
        'Number of active requests being processed',
        registry=registry
    )

# Pydantic models
class ReviewRequest(BaseModel):
    """Request model for review prediction."""
    text: str = Field(..., min_length=1, max_length=10000, description="Review text to analyze")
    user_id: Optional[str] = Field(None, max_length=100, description="User ID (optional)")
    timestamp: Optional[str] = Field(None, description="Review timestamp (ISO format, optional)")
    rating: Optional[float] = Field(None, ge=1, le=5, description="Review rating (1-5, optional)")
    additional_features: Optional[Dict[str, Any]] = Field(None, description="Additional features")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        if v is not None:
            try:
                datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError('Invalid timestamp format. Use ISO format.')
        return v

class BatchReviewRequest(BaseModel):
    """Request model for batch review prediction."""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of review texts")
    user_ids: Optional[List[str]] = Field(None, description="List of user IDs (optional)")
    timestamps: Optional[List[str]] = Field(None, description="List of timestamps (optional)")
    ratings: Optional[List[float]] = Field(None, description="List of ratings (optional)")
    additional_features: Optional[List[Dict[str, Any]]] = Field(None, description="List of additional features")
    
    @validator('texts')
    def validate_texts(cls, v):
        if not all(text.strip() for text in v):
            raise ValueError('All texts must be non-empty')
        return [text.strip() for text in v]

class PredictionResponse(BaseModel):
    """Response model for prediction."""
    probability: float = Field(..., description="Probability that the review is fake (0-1)")
    prediction: str = Field(..., description="Binary prediction (fake/legitimate)")
    confidence: str = Field(..., description="Confidence level (low/medium/high)")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_version: str = Field(..., description="Model version/type")

class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction."""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_processed: int = Field(..., description="Total number of reviews processed")
    processing_time: float = Field(..., description="Total processing time in seconds")

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service health status")
    timestamp: str = Field(..., description="Health check timestamp")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_health: str = Field(..., description="Model health status")
    uptime: float = Field(..., description="Service uptime in seconds")
    memory_usage: Optional[Dict[str, float]] = Field(None, description="Memory usage statistics")

class MetricsResponse(BaseModel):
    """Response model for metrics."""
    requests_total: int = Field(..., description="Total number of requests")
    predictions_total: int = Field(..., description="Total number of predictions")
    average_response_time: float = Field(..., description="Average response time")
    model_health: float = Field(..., description="Model health score")
    uptime: float = Field(..., description="Service uptime in seconds")

# Initialize FastAPI app
app = FastAPI(
    title="Fake Review Detection API",
    description="REST API for detecting fake reviews using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=utils.get_setting('api.cors.origins', ["*"]),
    allow_credentials=utils.get_setting('api.cors.allow_credentials', True),
    allow_methods=utils.get_setting('api.cors.allow_methods', ["*"]),
    allow_headers=utils.get_setting('api.cors.allow_headers', ["*"]),
)

# Add rate limiting if available
if RATE_LIMITING_AVAILABLE:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Store startup time for uptime calculation
startup_time = time.time()

# Middleware for metrics collection
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Collect HTTP metrics for all requests."""
    start_time = time.time()
    
    if PROMETHEUS_AVAILABLE:
        ACTIVE_REQUESTS.inc()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    if PROMETHEUS_AVAILABLE:
        # Record metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code
        ).inc()
        
        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(process_time)
        
        ACTIVE_REQUESTS.dec()
    
    # Add timing header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

def get_confidence_level(probability: float) -> str:
    """Determine confidence level based on probability."""
    if probability < 0.3 or probability > 0.7:
        return "high"
    elif probability < 0.4 or probability > 0.6:
        return "medium"
    else:
        return "low"

def get_prediction_label(probability: float) -> str:
    """Get binary prediction label."""
    threshold = utils.get_setting('api.prediction_threshold', 0.5)
    return "fake" if probability >= threshold else "legitimate"

async def get_detector_dependency() -> FakeReviewDetector:
    """Dependency to get detector instance."""
    detector = get_detector()
    if not detector.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return detector

# API Endpoints

@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Fake Review Detection API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict/batch", 
            "health": "/health",
            "metrics": "/metrics"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_review(
    request: ReviewRequest,
    background_tasks: BackgroundTasks,
    detector: FakeReviewDetector = Depends(get_detector_dependency)
):
    """Predict if a review is fake."""
    if RATE_LIMITING_AVAILABLE:
        # Apply rate limiting
        limiter.limit("60/minute")(predict_review)
    
    start_time = time.time()
    
    try:
        # Prepare additional features
        additional_features = request.additional_features or {}
        if request.rating is not None:
            additional_features['rating'] = request.rating
        
        # Make prediction
        if PROMETHEUS_AVAILABLE:
            with PREDICTION_DURATION.time():
                probability = detector.predict(
                    text=request.text,
                    user_id=request.user_id,
                    timestamp=request.timestamp,
                    additional_features=additional_features
                )
        else:
            probability = detector.predict(
                text=request.text,
                user_id=request.user_id,
                timestamp=request.timestamp,
                additional_features=additional_features
            )
        
        processing_time = time.time() - start_time
        
        # Create response
        response = PredictionResponse(
            probability=probability,
            prediction=get_prediction_label(probability),
            confidence=get_confidence_level(probability),
            processing_time=processing_time,
            model_version=detector.model_metadata.get('model_type', 'unknown')
        )
        
        # Record metrics
        if PROMETHEUS_AVAILABLE:
            PREDICTION_COUNT.labels(status='success').inc()
            PREDICTION_PROBABILITY.observe(probability)
        
        # Log prediction (in background to avoid blocking)
        background_tasks.add_task(
            logger.info,
            f"Prediction made: probability={probability:.3f}, processing_time={processing_time:.3f}s"
        )
        
        return response
        
    except Exception as e:
        if PROMETHEUS_AVAILABLE:
            PREDICTION_COUNT.labels(status='error').inc()
        
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchReviewRequest,
    background_tasks: BackgroundTasks,
    detector: FakeReviewDetector = Depends(get_detector_dependency)
):
    """Predict if multiple reviews are fake."""
    if RATE_LIMITING_AVAILABLE:
        # Apply stricter rate limiting for batch requests
        limiter.limit("10/minute")(predict_batch)
    
    start_time = time.time()
    
    try:
        # Prepare batch data
        texts = request.texts
        user_ids = request.user_ids
        timestamps = request.timestamps
        
        # Prepare additional features
        additional_features_list = None
        if request.additional_features or request.ratings:
            additional_features_list = []
            for i in range(len(texts)):
                features = {}
                if request.additional_features and i < len(request.additional_features):
                    features.update(request.additional_features[i])
                if request.ratings and i < len(request.ratings):
                    features['rating'] = request.ratings[i]
                additional_features_list.append(features if features else None)
        
        # Make batch prediction
        probabilities = detector.predict_batch(
            texts=texts,
            user_ids=user_ids,
            timestamps=timestamps,
            additional_features_list=additional_features_list
        )
        
        processing_time = time.time() - start_time
        
        # Create individual responses
        predictions = []
        for prob in probabilities:
            predictions.append(PredictionResponse(
                probability=prob,
                prediction=get_prediction_label(prob),
                confidence=get_confidence_level(prob),
                processing_time=processing_time / len(texts),  # Average time per prediction
                model_version=detector.model_metadata.get('model_type', 'unknown')
            ))
        
        response = BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(texts),
            processing_time=processing_time
        )
        
        # Record metrics
        if PROMETHEUS_AVAILABLE:
            PREDICTION_COUNT.labels(status='success').inc(len(texts))
            for prob in probabilities:
                PREDICTION_PROBABILITY.observe(prob)
        
        # Log batch prediction
        background_tasks.add_task(
            logger.info,
            f"Batch prediction completed: {len(texts)} reviews, processing_time={processing_time:.3f}s"
        )
        
        return response
        
    except Exception as e:
        if PROMETHEUS_AVAILABLE:
            PREDICTION_COUNT.labels(status='error').inc()
        
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Perform health check on the service and model."""
    try:
        detector = get_detector()
        health_data = detector.health_check()
        
        uptime = time.time() - startup_time
        memory_info = utils.memory_usage()
        
        # Update model health metric
        if PROMETHEUS_AVAILABLE:
            health_status = 1.0 if health_data["status"] == "healthy" else 0.0
            MODEL_HEALTH.set(health_status)
        
        response = HealthResponse(
            status=health_data["status"],
            timestamp=datetime.now().isoformat(),
            model_loaded=health_data["model_loaded"],
            model_health=health_data.get("prediction_test", "unknown"),
            uptime=uptime,
            memory_usage=memory_info if memory_info["rss_mb"] > 0 else None
        )
        
        # Return appropriate HTTP status
        status_code = 200 if health_data["status"] == "healthy" else 503
        return JSONResponse(content=response.dict(), status_code=status_code)
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        
        if PROMETHEUS_AVAILABLE:
            MODEL_HEALTH.set(0.0)
        
        error_response = HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            model_loaded=False,
            model_health="failed",
            uptime=time.time() - startup_time,
            memory_usage=None
        )
        
        return JSONResponse(content=error_response.dict(), status_code=503)

@app.get("/metrics")
async def get_metrics():
    """Get Prometheus metrics."""
    if not PROMETHEUS_AVAILABLE:
        return JSONResponse(
            content={"error": "Prometheus metrics not available"},
            status_code=501
        )
    
    try:
        # Generate Prometheus format metrics
        return Response(
            content=generate_latest(registry),
            media_type=CONTENT_TYPE_LATEST
        )
    except Exception as e:
        logger.error(f"Failed to generate metrics: {str(e)}")
        
        # Fallback to JSON metrics
        uptime = time.time() - startup_time
        
        metrics_response = MetricsResponse(
            requests_total=0,  # Would be populated from actual metrics
            predictions_total=0,
            average_response_time=0.0,
            model_health=1.0,
            uptime=uptime
        )
        
        return metrics_response

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    try:
        detector = get_detector()
        info = detector.get_model_info()
        return info
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

# Startup and shutdown events

@app.on_event("startup")
async def startup_event():
    """Initialize the service on startup."""
    logger.info("Starting Fake Review Detection API")
    
    try:
        # Initialize detector with model from settings
        model_path = utils.get_setting('deployment.model_path')
        preprocessor_path = utils.get_setting('deployment.preprocessor_path')
        
        detector = initialize_detector(model_path, preprocessor_path)
        
        logger.info("API startup completed successfully")
        
        if PROMETHEUS_AVAILABLE:
            MODEL_HEALTH.set(1.0 if detector.is_loaded else 0.0)
            
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        # Don't raise exception - allow API to start even if model loading fails
        if PROMETHEUS_AVAILABLE:
            MODEL_HEALTH.set(0.0)

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down Fake Review Detection API")
    
    if PROMETHEUS_AVAILABLE:
        MODEL_HEALTH.set(0.0)

# Custom exception handlers

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError exceptions."""
    logger.warning(f"ValueError in request {request.url}: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={"detail": f"Invalid input: {str(exc)}"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception in request {request.url}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Development server
if __name__ == "__main__":
    # Get configuration from settings
    host = utils.get_setting('api.host', '0.0.0.0')
    port = utils.get_setting('api.port', 8000)
    debug = utils.get_setting('api.debug', False)
    workers = utils.get_setting('api.workers', 1)
    
    logger.info(f"Starting development server on {host}:{port}")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        debug=debug,
        reload=debug,
        workers=workers if not debug else 1,
        access_log=True
    )
