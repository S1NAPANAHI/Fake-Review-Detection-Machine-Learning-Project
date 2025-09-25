"""
Enhanced FastAPI application for Multi-Platform Content Authenticity API.

This comprehensive API supports integration with:
- E-commerce platforms (Amazon, Shopify, etc.)
- Social media platforms (Twitter, Facebook, Instagram, etc.)
- App stores (Google Play, Apple App Store)
- Review platforms (Yelp, TripAdvisor, Google Reviews)
- General websites and blogs

Features:
- Multi-modal content analysis
- Real-time processing
- Bulk operations
- Platform-specific optimizations
- Comprehensive monitoring and analytics
"""

import os
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import numpy as np

# Enhanced imports for different platforms
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    RATE_LIMITING_AVAILABLE = True
except ImportError:
    RATE_LIMITING_AVAILABLE = False

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from deployment import get_detector, initialize_detector, FakeReviewDetector
from src import utils

# Initialize logging
logger = utils.setup_logging(__name__)

# Enums for platform types and content types
class PlatformType(str, Enum):
    ECOMMERCE = "ecommerce"
    SOCIAL_MEDIA = "social_media"
    APP_STORE = "app_store"
    REVIEW_SITE = "review_site"
    BLOG = "blog"
    FORUM = "forum"
    NEWS = "news"
    GENERAL = "general"

class ContentType(str, Enum):
    REVIEW = "review"
    COMMENT = "comment"
    POST = "post"
    DESCRIPTION = "description"
    REPLY = "reply"
    TESTIMONIAL = "testimonial"
    RATING = "rating"

class AnalysisMode(str, Enum):
    FAST = "fast"
    COMPREHENSIVE = "comprehensive"
    DEEP = "deep"

# Enhanced Pydantic models
class UserProfile(BaseModel):
    """User profile information for behavioral analysis."""
    user_id: str = Field(..., description="Unique user identifier")
    account_age_days: Optional[int] = Field(None, ge=0, description="Account age in days")
    total_reviews: Optional[int] = Field(None, ge=0, description="Total number of reviews/posts")
    average_rating: Optional[float] = Field(None, ge=1, le=5, description="Average rating given")
    verified_user: Optional[bool] = Field(None, description="Whether user is verified")
    follower_count: Optional[int] = Field(None, ge=0, description="Number of followers")
    following_count: Optional[int] = Field(None, ge=0, description="Number of following")
    profile_completeness: Optional[float] = Field(None, ge=0, le=1, description="Profile completeness score")
    
class ContentMetadata(BaseModel):
    """Metadata about the content being analyzed."""
    timestamp: Optional[str] = Field(None, description="Content creation timestamp")
    device_info: Optional[str] = Field(None, description="Device information")
    location: Optional[str] = Field(None, description="Location information")
    language: Optional[str] = Field(None, description="Content language")
    platform_specific: Optional[Dict[str, Any]] = Field(None, description="Platform-specific metadata")
    
class EnhancedAnalysisRequest(BaseModel):
    """Enhanced request model for comprehensive content analysis."""
    text: str = Field(..., min_length=1, max_length=10000, description="Content text to analyze")
    platform: PlatformType = Field(..., description="Platform type")
    content_type: ContentType = Field(..., description="Type of content")
    analysis_mode: AnalysisMode = Field(AnalysisMode.FAST, description="Analysis depth")
    
    # Content context
    user_profile: Optional[UserProfile] = Field(None, description="User profile information")
    content_metadata: Optional[ContentMetadata] = Field(None, description="Content metadata")
    
    # Platform-specific fields
    product_id: Optional[str] = Field(None, description="Product ID for e-commerce")
    post_id: Optional[str] = Field(None, description="Post ID for social media")
    app_id: Optional[str] = Field(None, description="App ID for app stores")
    business_id: Optional[str] = Field(None, description="Business ID for review sites")
    
    # Rating and sentiment context
    rating: Optional[float] = Field(None, ge=1, le=5, description="Associated rating")
    expected_sentiment: Optional[str] = Field(None, description="Expected sentiment based on rating")
    
    # Network analysis
    related_content_ids: Optional[List[str]] = Field(None, description="Related content for network analysis")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()

class BatchAnalysisRequest(BaseModel):
    """Request model for batch analysis."""
    items: List[EnhancedAnalysisRequest] = Field(..., min_items=1, max_items=1000, description="List of items to analyze")
    priority: Optional[str] = Field("normal", description="Processing priority (low/normal/high)")
    callback_url: Optional[str] = Field(None, description="Webhook URL for results")
    
class StreamAnalysisRequest(BaseModel):
    """Request model for real-time stream analysis."""
    stream_id: str = Field(..., description="Unique stream identifier")
    platform: PlatformType = Field(..., description="Platform type")
    content_type: ContentType = Field(..., description="Content type")
    filters: Optional[Dict[str, Any]] = Field(None, description="Stream filtering options")

class AuthenticityResult(BaseModel):
    """Comprehensive authenticity analysis result."""
    # Core prediction
    authenticity_score: float = Field(..., ge=0, le=1, description="Authenticity score (0=fake, 1=authentic)")
    risk_level: str = Field(..., description="Risk level (low/medium/high/critical)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in prediction")
    
    # Detailed analysis
    text_analysis: Dict[str, Any] = Field(..., description="Text-based features analysis")
    behavioral_analysis: Optional[Dict[str, Any]] = Field(None, description="User behavior analysis")
    temporal_analysis: Optional[Dict[str, Any]] = Field(None, description="Temporal pattern analysis")
    network_analysis: Optional[Dict[str, Any]] = Field(None, description="Network-based analysis")
    
    # Platform-specific insights
    platform_insights: Dict[str, Any] = Field(..., description="Platform-specific analysis")
    
    # Explainability
    key_indicators: List[str] = Field(..., description="Key indicators for the decision")
    risk_factors: List[str] = Field(..., description="Identified risk factors")
    
    # Processing metadata
    processing_time: float = Field(..., description="Processing time in seconds")
    model_version: str = Field(..., description="Model version used")
    analysis_timestamp: str = Field(..., description="Analysis timestamp")

class BatchAnalysisResponse(BaseModel):
    """Response model for batch analysis."""
    job_id: str = Field(..., description="Batch job identifier")
    status: str = Field(..., description="Job status")
    total_items: int = Field(..., description="Total items submitted")
    completed_items: int = Field(..., description="Items completed")
    results: List[AuthenticityResult] = Field(..., description="Analysis results")
    processing_time: float = Field(..., description="Total processing time")
    
class WebhookPayload(BaseModel):
    """Webhook payload for async notifications."""
    event_type: str = Field(..., description="Event type")
    job_id: Optional[str] = Field(None, description="Job ID if applicable")
    data: Dict[str, Any] = Field(..., description="Event data")
    timestamp: str = Field(..., description="Event timestamp")

# Initialize FastAPI app with enhanced configuration
app = FastAPI(
    title="Multi-Platform Content Authenticity API",
    description="Comprehensive API for detecting fake content across e-commerce, social media, app stores, and review platforms",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={
        "name": "Content Authenticity Team",
        "email": "sina.panahi@outlook.com",
        "url": "https://github.com/S1NAPANAHI/Fake-Review-Detection-Machine-Learning-Project"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# Enhanced CORS for multiple platforms
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://*.shopify.com",
        "https://*.wordpress.com", 
        "https://*.facebook.com",
        "https://*.twitter.com",
        "https://*.instagram.com",
        "https://*.linkedin.com",
        "https://*.yelp.com",
        "https://*.tripadvisor.com",
        "https://*.amazon.com",
        "https://*.ebay.com",
        "https://play.google.com",
        "https://apps.apple.com",
        "http://localhost:*",
        "https://localhost:*"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"]
)

# Authentication
security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate API key authentication."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # TODO: Implement proper API key validation
    api_key = credentials.credentials
    if not api_key.startswith("auth_"):  # Simple validation for demo
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return {"api_key": api_key, "user_id": "demo_user"}

# Rate limiting setup
if RATE_LIMITING_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# WebSocket connection manager for real-time analysis
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.streams: Dict[str, Dict] = {}
    
    async def connect(self, websocket: WebSocket, stream_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.streams[stream_id] = {
            "websocket": websocket,
            "created_at": datetime.now(),
            "processed_count": 0
        }
    
    def disconnect(self, websocket: WebSocket, stream_id: str):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if stream_id in self.streams:
            del self.streams[stream_id]
    
    async def send_result(self, stream_id: str, data: dict):
        if stream_id in self.streams:
            try:
                await self.streams[stream_id]["websocket"].send_json(data)
                self.streams[stream_id]["processed_count"] += 1
            except WebSocketDisconnect:
                await self.disconnect_stream(stream_id)
    
    async def disconnect_stream(self, stream_id: str):
        if stream_id in self.streams:
            websocket = self.streams[stream_id]["websocket"]
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
            del self.streams[stream_id]

manager = ConnectionManager()

# Helper functions for enhanced analysis
def get_platform_specific_features(platform: PlatformType, request: EnhancedAnalysisRequest) -> Dict[str, Any]:
    """Extract platform-specific features."""
    features = {}
    
    if platform == PlatformType.ECOMMERCE:
        features.update({
            "product_id": request.product_id,
            "has_rating": request.rating is not None,
            "rating_text_consistency": check_rating_text_consistency(request.rating, request.text) if request.rating else None
        })
    
    elif platform == PlatformType.SOCIAL_MEDIA:
        features.update({
            "post_id": request.post_id,
            "content_length": len(request.text),
            "has_hashtags": "#" in request.text,
            "has_mentions": "@" in request.text
        })
    
    elif platform == PlatformType.APP_STORE:
        features.update({
            "app_id": request.app_id,
            "has_rating": request.rating is not None,
            "review_length_category": categorize_review_length(len(request.text))
        })
    
    elif platform == PlatformType.REVIEW_SITE:
        features.update({
            "business_id": request.business_id,
            "has_rating": request.rating is not None,
            "review_specificity": calculate_review_specificity(request.text)
        })
    
    return features

def check_rating_text_consistency(rating: float, text: str) -> float:
    """Check consistency between rating and text sentiment."""
    # Simplified sentiment analysis
    positive_words = ["great", "excellent", "amazing", "love", "perfect", "best"]
    negative_words = ["terrible", "awful", "hate", "worst", "bad", "horrible"]
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if rating >= 4 and positive_count > negative_count:
        return 1.0
    elif rating <= 2 and negative_count > positive_count:
        return 1.0
    elif rating == 3 and abs(positive_count - negative_count) <= 1:
        return 1.0
    else:
        return 0.5

def categorize_review_length(length: int) -> str:
    """Categorize review length."""
    if length < 50:
        return "short"
    elif length < 200:
        return "medium"
    else:
        return "long"

def calculate_review_specificity(text: str) -> float:
    """Calculate how specific a review is (simplified)."""
    specific_words = ["location", "price", "staff", "service", "quality", "time", "date"]
    text_lower = text.lower()
    specificity = sum(1 for word in specific_words if word in text_lower) / len(specific_words)
    return min(specificity, 1.0)

async def enhanced_analyze_content(request: EnhancedAnalysisRequest, detector: FakeReviewDetector) -> AuthenticityResult:
    """Perform enhanced content analysis."""
    start_time = time.time()
    
    # Extract platform-specific features
    platform_features = get_platform_specific_features(request.platform, request)
    
    # Prepare additional features for the model
    additional_features = platform_features.copy()
    if request.rating:
        additional_features["rating"] = request.rating
    
    # Get base prediction
    authenticity_prob = detector.predict(
        text=request.text,
        user_id=request.user_profile.user_id if request.user_profile else None,
        timestamp=request.content_metadata.timestamp if request.content_metadata else None,
        additional_features=additional_features
    )
    
    # Enhanced analysis based on mode
    text_analysis = {
        "sentiment_consistency": check_rating_text_consistency(request.rating, request.text) if request.rating else None,
        "text_length": len(request.text),
        "language_detected": request.content_metadata.language if request.content_metadata else "unknown"
    }
    
    behavioral_analysis = None
    if request.user_profile and request.analysis_mode in [AnalysisMode.COMPREHENSIVE, AnalysisMode.DEEP]:
        behavioral_analysis = {
            "account_age_risk": "high" if request.user_profile.account_age_days and request.user_profile.account_age_days < 30 else "low",
            "review_volume_risk": "high" if request.user_profile.total_reviews and request.user_profile.total_reviews > 100 else "low",
            "profile_completeness": request.user_profile.profile_completeness or 0.5
        }
    
    # Determine risk level
    if authenticity_prob >= 0.8:
        risk_level = "low"
    elif authenticity_prob >= 0.6:
        risk_level = "medium"
    elif authenticity_prob >= 0.3:
        risk_level = "high"
    else:
        risk_level = "critical"
    
    # Generate key indicators
    key_indicators = []
    risk_factors = []
    
    if request.rating and text_analysis["sentiment_consistency"] and text_analysis["sentiment_consistency"] < 0.5:
        risk_factors.append("Rating-text sentiment mismatch")
    
    if text_analysis["text_length"] < 20:
        risk_factors.append("Very short content")
    elif text_analysis["text_length"] > 1000:
        key_indicators.append("Detailed content")
    
    if behavioral_analysis and behavioral_analysis["account_age_risk"] == "high":
        risk_factors.append("New account")
    
    processing_time = time.time() - start_time
    
    return AuthenticityResult(
        authenticity_score=authenticity_prob,
        risk_level=risk_level,
        confidence=min(abs(authenticity_prob - 0.5) * 2, 1.0),
        text_analysis=text_analysis,
        behavioral_analysis=behavioral_analysis,
        temporal_analysis=None,  # TODO: Implement temporal analysis
        network_analysis=None,  # TODO: Implement network analysis
        platform_insights=platform_features,
        key_indicators=key_indicators,
        risk_factors=risk_factors,
        processing_time=processing_time,
        model_version=detector.model_metadata.get('model_type', 'unknown'),
        analysis_timestamp=datetime.now().isoformat()
    )

# API Endpoints

@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with comprehensive API information."""
    return {
        "service": "Multi-Platform Content Authenticity API",
        "version": "2.0.0",
        "description": "Comprehensive fake content detection across multiple platforms",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "supported_platforms": [platform.value for platform in PlatformType],
        "supported_content_types": [content.value for content in ContentType],
        "analysis_modes": [mode.value for mode in AnalysisMode],
        "endpoints": {
            "single_analysis": "/v1/analyze/content",
            "batch_analysis": "/v1/analyze/batch",
            "stream_analysis": "/v1/stream/{stream_id}",
            "platform_specific": {
                "ecommerce": "/v1/platforms/ecommerce/analyze",
                "social_media": "/v1/platforms/social/analyze",
                "app_store": "/v1/platforms/appstore/analyze",
                "review_site": "/v1/platforms/reviews/analyze"
            },
            "admin": "/v1/admin/",
            "health": "/health",
            "metrics": "/metrics"
        }
    }

# Enhanced single content analysis
@app.post("/v1/analyze/content", response_model=AuthenticityResult)
async def analyze_content(
    request: EnhancedAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    detector: FakeReviewDetector = Depends(get_detector)
):
    """Analyze content for authenticity with enhanced features."""
    try:
        result = await enhanced_analyze_content(request, detector)
        
        # Log analysis in background
        background_tasks.add_task(
            logger.info,
            f"Enhanced analysis completed: platform={request.platform}, "
            f"risk_level={result.risk_level}, processing_time={result.processing_time:.3f}s"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Enhanced analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Batch analysis endpoint
@app.post("/v1/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    detector: FakeReviewDetector = Depends(get_detector)
):
    """Process batch analysis requests."""
    job_id = f"batch_{int(time.time())}_{current_user['user_id']}"
    start_time = time.time()
    
    try:
        results = []
        for item in request.items:
            result = await enhanced_analyze_content(item, detector)
            results.append(result)
        
        processing_time = time.time() - start_time
        
        response = BatchAnalysisResponse(
            job_id=job_id,
            status="completed",
            total_items=len(request.items),
            completed_items=len(results),
            results=results,
            processing_time=processing_time
        )
        
        # Send webhook if callback URL provided
        if request.callback_url:
            background_tasks.add_task(
                send_webhook,
                request.callback_url,
                WebhookPayload(
                    event_type="batch_analysis_completed",
                    job_id=job_id,
                    data=response.dict(),
                    timestamp=datetime.now().isoformat()
                )
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

# Real-time WebSocket endpoint for streaming analysis
@app.websocket("/v1/stream/{stream_id}")
async def stream_analysis(
    websocket: WebSocket,
    stream_id: str,
    detector: FakeReviewDetector = Depends(get_detector)
):
    """Real-time content analysis via WebSocket."""
    await manager.connect(websocket, stream_id)
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_json()
            
            # Parse request
            try:
                request = EnhancedAnalysisRequest(**data)
                result = await enhanced_analyze_content(request, detector)
                
                # Send result back
                await manager.send_result(stream_id, {
                    "type": "analysis_result",
                    "data": result.dict(),
                    "stream_id": stream_id
                })
                
            except Exception as e:
                await manager.send_result(stream_id, {
                    "type": "error",
                    "message": str(e),
                    "stream_id": stream_id
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, stream_id)
        logger.info(f"WebSocket connection closed for stream {stream_id}")

# Platform-specific endpoints
@app.post("/v1/platforms/ecommerce/analyze", response_model=AuthenticityResult)
async def analyze_ecommerce_content(
    text: str,
    product_id: str,
    rating: Optional[float] = None,
    user_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    detector: FakeReviewDetector = Depends(get_detector)
):
    """Specialized endpoint for e-commerce platform integration."""
    request = EnhancedAnalysisRequest(
        text=text,
        platform=PlatformType.ECOMMERCE,
        content_type=ContentType.REVIEW,
        product_id=product_id,
        rating=rating,
        user_profile=UserProfile(user_id=user_id) if user_id else None
    )
    
    return await enhanced_analyze_content(request, detector)

@app.post("/v1/platforms/social/analyze", response_model=AuthenticityResult)
async def analyze_social_media_content(
    text: str,
    post_id: str,
    user_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    detector: FakeReviewDetector = Depends(get_detector)
):
    """Specialized endpoint for social media platform integration."""
    request = EnhancedAnalysisRequest(
        text=text,
        platform=PlatformType.SOCIAL_MEDIA,
        content_type=ContentType.COMMENT,
        post_id=post_id,
        user_profile=UserProfile(user_id=user_id) if user_id else None
    )
    
    return await enhanced_analyze_content(request, detector)

# Webhook utility
async def send_webhook(url: str, payload: WebhookPayload):
    """Send webhook notification."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload.dict())
            logger.info(f"Webhook sent to {url}: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to send webhook to {url}: {str(e)}")

# Health check with enhanced information
@app.get("/health")
async def enhanced_health_check():
    """Enhanced health check with detailed system information."""
    try:
        detector = get_detector()
        health_data = detector.health_check()
        
        return {
            "status": "healthy" if health_data["status"] == "healthy" else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "model_status": health_data,
            "active_streams": len(manager.streams),
            "supported_platforms": len(PlatformType),
            "api_version": "2.0.0",
            "uptime": time.time() - app.state.startup_time if hasattr(app.state, 'startup_time') else 0
        }
        
    except Exception as e:
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            },
            status_code=503
        )

# Admin endpoints for model management
@app.get("/v1/admin/models/info")
async def get_models_info(current_user: dict = Depends(get_current_user)):
    """Get information about loaded models."""
    try:
        detector = get_detector()
        return detector.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/admin/analytics/summary")
async def get_analytics_summary(current_user: dict = Depends(get_current_user)):
    """Get analytics summary."""
    return {
        "total_analyses_today": 1000,  # TODO: Implement actual analytics
        "platform_breakdown": {
            "ecommerce": 450,
            "social_media": 300,
            "app_store": 150,
            "review_site": 100
        },
        "risk_level_distribution": {
            "low": 60,
            "medium": 25,
            "high": 10,
            "critical": 5
        },
        "active_streams": len(manager.streams)
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the enhanced service."""
    app.state.startup_time = time.time()
    logger.info("Multi-Platform Content Authenticity API starting up...")
    
    try:
        # Initialize detector
        detector = initialize_detector()
        logger.info("Enhanced API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Enhanced startup failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "enhanced_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
