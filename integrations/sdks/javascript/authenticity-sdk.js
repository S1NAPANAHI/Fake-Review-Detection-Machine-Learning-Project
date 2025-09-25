/**
 * Content Authenticity SDK for JavaScript/TypeScript
 * 
 * Easy integration library for websites, e-commerce platforms,
 * social media sites, and other web applications.
 * 
 * Usage Examples:
 * - E-commerce: Validate product reviews in real-time
 * - Social Media: Check comment authenticity
 * - Blogs: Verify user-generated content
 * - Forums: Detect fake posts and spam
 */

class ContentAuthenticitySDK {
    constructor(config = {}) {
        this.apiKey = config.apiKey;
        this.baseUrl = config.baseUrl || 'https://api.contentauthenticity.com';
        this.timeout = config.timeout || 30000;
        this.retries = config.retries || 3;
        this.debug = config.debug || false;
        
        if (!this.apiKey) {
            throw new Error('API key is required. Get one at: https://contentauthenticity.com/api-keys');
        }
        
        // Platform-specific configurations
        this.platformConfigs = {
            shopify: { webhookSecret: config.shopifyWebhookSecret },
            wordpress: { pluginVersion: '1.0.0' },
            social: { rateLimits: config.socialRateLimits || {} }
        };
        
        this.log('SDK initialized', { baseUrl: this.baseUrl });
    }
    
    log(message, data = {}) {
        if (this.debug) {
            console.log(`[ContentAuthenticitySDK] ${message}`, data);
        }
    }
    
    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const config = {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.apiKey}`,
                'Content-Type': 'application/json',
                'User-Agent': 'ContentAuthenticity-JS-SDK/1.0.0'
            },
            timeout: this.timeout,
            ...options
        };
        
        let lastError;
        
        for (let attempt = 1; attempt <= this.retries; attempt++) {
            try {
                this.log(`Making request (attempt ${attempt})`, { url, method: config.method });
                
                const response = await fetch(url, config);
                
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    throw new Error(`API Error ${response.status}: ${errorData.detail || response.statusText}`);
                }
                
                const data = await response.json();
                this.log('Request successful', { status: response.status });
                return data;
                
            } catch (error) {
                lastError = error;
                this.log(`Request failed (attempt ${attempt})`, { error: error.message });
                
                if (attempt < this.retries && this.shouldRetry(error)) {
                    const delay = Math.pow(2, attempt) * 1000; // Exponential backoff
                    await new Promise(resolve => setTimeout(resolve, delay));
                    continue;
                }
                break;
            }
        }
        
        throw lastError;
    }
    
    shouldRetry(error) {
        return error.message.includes('timeout') || 
               error.message.includes('503') || 
               error.message.includes('502');
    }
    
    /**
     * Analyze a single piece of content for authenticity
     * @param {Object} content - Content to analyze
     * @param {string} content.text - The text content
     * @param {string} content.platform - Platform type (ecommerce, social_media, etc.)
     * @param {string} content.contentType - Content type (review, comment, post)
     * @param {Object} options - Additional options
     * @returns {Promise<Object>} Analysis result
     */
    async analyzeContent(content, options = {}) {
        const payload = {
            text: content.text,
            platform: content.platform || 'general',
            content_type: content.contentType || 'review',
            analysis_mode: options.mode || 'fast',
            ...this.buildContentContext(content, options)
        };
        
        return await this.request('/v1/analyze/content', {
            body: JSON.stringify(payload)
        });
    }
    
    /**
     * Analyze multiple pieces of content in batch
     * @param {Array} items - Array of content items
     * @param {Object} options - Batch options
     * @returns {Promise<Object>} Batch analysis result
     */
    async analyzeBatch(items, options = {}) {
        const payload = {
            items: items.map(item => ({
                text: item.text,
                platform: item.platform || 'general',
                content_type: item.contentType || 'review',
                analysis_mode: options.mode || 'fast',
                ...this.buildContentContext(item, options)
            })),
            priority: options.priority || 'normal',
            callback_url: options.webhookUrl
        };
        
        return await this.request('/v1/analyze/batch', {
            body: JSON.stringify(payload)
        });
    }
    
    buildContentContext(content, options) {
        const context = {};
        
        // User profile information
        if (content.user) {
            context.user_profile = {
                user_id: content.user.id,
                account_age_days: content.user.accountAgeDays,
                total_reviews: content.user.totalReviews,
                verified_user: content.user.verified,
                follower_count: content.user.followers,
                following_count: content.user.following
            };
        }
        
        // Content metadata
        if (content.metadata) {
            context.content_metadata = {
                timestamp: content.metadata.timestamp,
                location: content.metadata.location,
                device_info: content.metadata.device,
                language: content.metadata.language
            };
        }
        
        // Platform-specific fields
        if (content.productId) context.product_id = content.productId;
        if (content.postId) context.post_id = content.postId;
        if (content.appId) context.app_id = content.appId;
        if (content.businessId) context.business_id = content.businessId;
        if (content.rating) context.rating = content.rating;
        
        return context;
    }
    
    /**
     * E-commerce specific analysis
     * @param {Object} review - Product review data
     * @returns {Promise<Object>} Analysis result
     */
    async analyzeProductReview(review) {
        return await this.request('/v1/platforms/ecommerce/analyze', {
            body: JSON.stringify({
                text: review.text,
                product_id: review.productId,
                rating: review.rating,
                user_id: review.userId
            })
        });
    }
    
    /**
     * Social media specific analysis
     * @param {Object} comment - Social media comment data
     * @returns {Promise<Object>} Analysis result
     */
    async analyzeSocialComment(comment) {
        return await this.request('/v1/platforms/social/analyze', {
            body: JSON.stringify({
                text: comment.text,
                post_id: comment.postId,
                user_id: comment.userId
            })
        });
    }
    
    /**
     * Create a real-time analysis stream
     * @param {Object} config - Stream configuration
     * @returns {WebSocketStream} WebSocket stream for real-time analysis
     */
    createStream(config = {}) {
        const streamId = `stream_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        const wsUrl = this.baseUrl.replace('https://', 'wss://').replace('http://', 'ws://');
        
        return new WebSocketStream(`${wsUrl}/v1/stream/${streamId}`, {
            apiKey: this.apiKey,
            ...config
        });
    }
    
    /**
     * Get service health status
     * @returns {Promise<Object>} Health status
     */
    async getHealth() {
        return await this.request('/health', { method: 'GET' });
    }
    
    /**
     * Validate content in real-time (for forms, input fields)
     * @param {HTMLElement} element - Input element to monitor
     * @param {Object} config - Validation configuration
     */
    attachRealTimeValidation(element, config = {}) {
        const validator = new RealTimeValidator(this, config);
        validator.attach(element);
        return validator;
    }
}

/**
 * WebSocket stream for real-time content analysis
 */
class WebSocketStream {
    constructor(url, config) {
        this.url = url;
        this.config = config;
        this.ws = null;
        this.listeners = {};
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = config.maxReconnectAttempts || 5;
        this.reconnectDelay = config.reconnectDelay || 1000;
    }
    
    connect() {
        return new Promise((resolve, reject) => {
            try {
                this.ws = new WebSocket(this.url);
                
                this.ws.onopen = () => {
                    console.log('WebSocket connected');
                    this.reconnectAttempts = 0;
                    resolve(this);
                };
                
                this.ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.emit(data.type, data);
                };
                
                this.ws.onclose = () => {
                    console.log('WebSocket disconnected');
                    this.reconnect();
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    reject(error);
                };
                
            } catch (error) {
                reject(error);
            }
        });
    }
    
    reconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            setTimeout(() => {
                console.log(`Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
                this.connect();
            }, this.reconnectDelay * this.reconnectAttempts);
        }
    }
    
    analyze(content) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(content));
        } else {
            throw new Error('WebSocket not connected');
        }
    }
    
    on(event, callback) {
        if (!this.listeners[event]) {
            this.listeners[event] = [];
        }
        this.listeners[event].push(callback);
    }
    
    emit(event, data) {
        if (this.listeners[event]) {
            this.listeners[event].forEach(callback => callback(data));
        }
    }
    
    disconnect() {
        if (this.ws) {
            this.ws.close();
        }
    }
}

/**
 * Real-time form validation
 */
class RealTimeValidator {
    constructor(sdk, config) {
        this.sdk = sdk;
        this.config = {
            debounceMs: 1000,
            minLength: 10,
            showIndicator: true,
            indicatorPosition: 'after',
            ...config
        };
        this.timeouts = new Map();
    }
    
    attach(element) {
        // Create indicator element
        if (this.config.showIndicator) {
            this.createIndicator(element);
        }
        
        // Attach event listeners
        element.addEventListener('input', (event) => {
            this.handleInput(element, event.target.value);
        });
        
        element.addEventListener('blur', (event) => {
            this.validateNow(element, event.target.value);
        });
    }
    
    handleInput(element, value) {
        // Clear existing timeout
        if (this.timeouts.has(element)) {
            clearTimeout(this.timeouts.get(element));
        }
        
        // Set new timeout for debounced validation
        const timeout = setTimeout(() => {
            if (value.length >= this.config.minLength) {
                this.validateNow(element, value);
            }
        }, this.config.debounceMs);
        
        this.timeouts.set(element, timeout);
    }
    
    async validateNow(element, value) {
        if (!value || value.length < this.config.minLength) {
            this.updateIndicator(element, null);
            return;
        }
        
        try {
            this.updateIndicator(element, 'loading');
            
            const result = await this.sdk.analyzeContent({
                text: value,
                platform: this.config.platform || 'general',
                contentType: this.config.contentType || 'review'
            });
            
            this.updateIndicator(element, result);
            
            // Emit custom event
            element.dispatchEvent(new CustomEvent('authenticity-result', {
                detail: result
            }));
            
        } catch (error) {
            this.updateIndicator(element, 'error');
            console.error('Validation error:', error);
        }
    }
    
    createIndicator(element) {
        const indicator = document.createElement('div');
        indicator.className = 'authenticity-indicator';
        indicator.style.cssText = `
            display: inline-block;
            margin-left: 10px;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        `;
        
        // Insert indicator
        if (this.config.indicatorPosition === 'after') {
            element.parentNode.insertBefore(indicator, element.nextSibling);
        } else {
            element.parentNode.insertBefore(indicator, element);
        }
        
        element._authenticityIndicator = indicator;
    }
    
    updateIndicator(element, result) {
        const indicator = element._authenticityIndicator;
        if (!indicator) return;
        
        if (result === null) {
            indicator.style.display = 'none';
        } else if (result === 'loading') {
            indicator.style.display = 'inline-block';
            indicator.style.backgroundColor = '#f0f0f0';
            indicator.style.color = '#666';
            indicator.textContent = 'Checking...';
        } else if (result === 'error') {
            indicator.style.display = 'inline-block';
            indicator.style.backgroundColor = '#ff6b6b';
            indicator.style.color = 'white';
            indicator.textContent = 'Error';
        } else {
            indicator.style.display = 'inline-block';
            
            if (result.risk_level === 'low') {
                indicator.style.backgroundColor = '#51cf66';
                indicator.style.color = 'white';
                indicator.textContent = '✓ Authentic';
            } else if (result.risk_level === 'medium') {
                indicator.style.backgroundColor = '#ffd43b';
                indicator.style.color = '#333';
                indicator.textContent = '⚠ Suspicious';
            } else {
                indicator.style.backgroundColor = '#ff6b6b';
                indicator.style.color = 'white';
                indicator.textContent = '✗ High Risk';
            }
        }
    }
}

/**
 * Platform-specific integrations
 */
const PlatformIntegrations = {
    
    /**
     * Shopify integration helpers
     */
    shopify: {
        /**
         * Analyze product reviews from Shopify webhook
         * @param {Object} sdk - SDK instance
         * @param {Object} webhookData - Shopify webhook data
         */
        async handleReviewWebhook(sdk, webhookData) {
            if (webhookData.topic === 'product_reviews/create') {
                const review = webhookData.data;
                
                return await sdk.analyzeProductReview({
                    text: review.body,
                    rating: review.rating,
                    productId: review.product_id,
                    userId: review.author
                });
            }
        },
        
        /**
         * Bulk analyze existing Shopify reviews
         * @param {Object} sdk - SDK instance
         * @param {Array} reviews - Array of Shopify reviews
         */
        async bulkAnalyzeReviews(sdk, reviews) {
            const items = reviews.map(review => ({
                text: review.body,
                rating: review.rating,
                productId: review.product_id,
                userId: review.author,
                platform: 'ecommerce',
                contentType: 'review'
            }));
            
            return await sdk.analyzeBatch(items);
        }
    },
    
    /**
     * WordPress integration helpers
     */
    wordpress: {
        /**
         * Analyze WordPress comments
         * @param {Object} sdk - SDK instance
         * @param {Object} comment - WordPress comment object
         */
        async analyzeComment(sdk, comment) {
            return await sdk.analyzeContent({
                text: comment.comment_content,
                platform: 'blog',
                contentType: 'comment',
                user: {
                    id: comment.comment_author_email,
                    verified: comment.user_id > 0
                },
                metadata: {
                    timestamp: comment.comment_date,
                    device: comment.comment_agent
                }
            });
        },
        
        /**
         * Filter comments based on authenticity
         * @param {Object} sdk - SDK instance
         * @param {Array} comments - WordPress comments
         * @param {number} threshold - Risk threshold (0-1)
         */
        async filterComments(sdk, comments, threshold = 0.7) {
            const results = await sdk.analyzeBatch(
                comments.map(comment => ({
                    text: comment.comment_content,
                    platform: 'blog',
                    contentType: 'comment'
                }))
            );
            
            return comments.filter((comment, index) => {
                const result = results.results[index];
                return result.authenticity_score >= threshold;
            });
        }
    },
    
    /**
     * Social media integration helpers
     */
    social: {
        /**
         * Monitor social media mentions
         * @param {Object} sdk - SDK instance
         * @param {Array} mentions - Social media mentions
         */
        async monitorMentions(sdk, mentions) {
            const items = mentions.map(mention => ({
                text: mention.text,
                platform: 'social_media',
                contentType: 'post',
                user: {
                    id: mention.author_id,
                    followers: mention.author_followers,
                    verified: mention.author_verified
                },
                postId: mention.id
            }));
            
            return await sdk.analyzeBatch(items);
        }
    }
};

// Export for different module systems
if (typeof module !== 'undefined' && module.exports) {
    // Node.js
    module.exports = { ContentAuthenticitySDK, WebSocketStream, RealTimeValidator, PlatformIntegrations };
} else if (typeof define === 'function' && define.amd) {
    // AMD
    define(function() {
        return { ContentAuthenticitySDK, WebSocketStream, RealTimeValidator, PlatformIntegrations };
    });
} else {
    // Browser global
    window.ContentAuthenticitySDK = ContentAuthenticitySDK;
    window.WebSocketStream = WebSocketStream;
    window.RealTimeValidator = RealTimeValidator;
    window.PlatformIntegrations = PlatformIntegrations;
}
