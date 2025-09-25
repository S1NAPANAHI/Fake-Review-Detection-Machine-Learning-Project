<?php
/**
 * Plugin Name: Content Authenticity Checker
 * Plugin URI: https://github.com/S1NAPANAHI/Fake-Review-Detection-Machine-Learning-Project
 * Description: Automatically detect and filter fake reviews, comments, and user-generated content using advanced AI.
 * Version: 1.0.0
 * Author: SINA PANAHI
 * Author URI: https://github.com/S1NAPANAHI
 * License: MIT
 * Text Domain: content-authenticity
 * Domain Path: /languages
 */

// Prevent direct access
if (!defined('ABSPATH')) {
    exit;
}

// Define plugin constants
define('CONTENT_AUTHENTICITY_VERSION', '1.0.0');
define('CONTENT_AUTHENTICITY_PLUGIN_URL', plugin_dir_url(__FILE__));
define('CONTENT_AUTHENTICITY_PLUGIN_PATH', plugin_dir_path(__FILE__));

/**
 * Main Content Authenticity Plugin Class
 */
class ContentAuthenticity {
    
    private $api_key;
    private $api_url;
    private $settings;
    
    public function __construct() {
        $this->settings = get_option('content_authenticity_settings', []);
        $this->api_key = $this->settings['api_key'] ?? '';
        $this->api_url = $this->settings['api_url'] ?? 'https://api.contentauthenticity.com';
        
        add_action('init', [$this, 'init']);
    }
    
    public function init() {
        // Hook into WordPress comment system
        add_action('pre_comment_on_post', [$this, 'check_comment_before_submission']);
        add_filter('preprocess_comment', [$this, 'process_comment'], 1);
        add_action('comment_post', [$this, 'analyze_comment_after_post']);
        
        // Hook into WooCommerce reviews (if active)
        if (class_exists('WooCommerce')) {
            add_action('woocommerce_product_review_comment_form_args', [$this, 'modify_review_form']);
            add_filter('woocommerce_product_review_save_comment_data', [$this, 'process_product_review']);
        }
        
        // Admin hooks
        add_action('admin_menu', [$this, 'add_admin_menu']);
        add_action('admin_init', [$this, 'admin_init']);
        add_action('admin_enqueue_scripts', [$this, 'admin_scripts']);
        
        // AJAX hooks
        add_action('wp_ajax_test_api_connection', [$this, 'test_api_connection']);
        add_action('wp_ajax_bulk_analyze_comments', [$this, 'bulk_analyze_comments']);
        add_action('wp_ajax_get_authenticity_stats', [$this, 'get_authenticity_stats']);
        
        // Frontend hooks
        add_action('wp_enqueue_scripts', [$this, 'frontend_scripts']);
        add_action('wp_footer', [$this, 'add_real_time_validation']);
        
        // Dashboard widget
        add_action('wp_dashboard_setup', [$this, 'add_dashboard_widget']);
        
        // REST API endpoints
        add_action('rest_api_init', [$this, 'register_rest_routes']);
    }
    
    /**
     * Check comment before submission
     */
    public function check_comment_before_submission($post_id) {
        if (!$this->is_configured()) {
            return;
        }
        
        // Only check if real-time checking is enabled
        if (!($this->settings['real_time_checking'] ?? false)) {
            return;
        }
        
        $comment_content = $_POST['comment'] ?? '';
        
        if (empty($comment_content)) {
            return;
        }
        
        try {
            $result = $this->analyze_content([
                'text' => $comment_content,
                'platform' => 'blog',
                'content_type' => 'comment',
                'user_profile' => [
                    'user_id' => get_current_user_id(),
                    'verified_user' => is_user_logged_in()
                ]
            ]);
            
            $risk_threshold = floatval($this->settings['risk_threshold'] ?? 0.7);
            
            if ($result['authenticity_score'] < $risk_threshold) {
                wp_die(
                    __('Your comment appears to be suspicious and cannot be published. Please revise your comment and try again.', 'content-authenticity'),
                    __('Comment Blocked', 'content-authenticity'),
                    ['response' => 403, 'back_link' => true]
                );
            }
            
        } catch (Exception $e) {
            // Log error but don't block comment if API fails
            error_log('Content Authenticity API Error: ' . $e->getMessage());
        }
    }
    
    /**
     * Process comment data
     */
    public function process_comment($commentdata) {
        if (!$this->is_configured()) {
            return $commentdata;
        }
        
        // Add authenticity metadata
        $commentdata['comment_meta']['authenticity_checked'] = current_time('mysql');
        
        return $commentdata;
    }
    
    /**
     * Analyze comment after it's posted
     */
    public function analyze_comment_after_post($comment_id) {
        if (!$this->is_configured()) {
            return;
        }
        
        $comment = get_comment($comment_id);
        
        if (!$comment) {
            return;
        }
        
        try {
            $result = $this->analyze_content([
                'text' => $comment->comment_content,
                'platform' => 'blog',
                'content_type' => 'comment',
                'user_profile' => [
                    'user_id' => $comment->user_id,
                    'verified_user' => $comment->user_id > 0
                ],
                'content_metadata' => [
                    'timestamp' => $comment->comment_date,
                    'device_info' => $comment->comment_agent
                ]
            ]);
            
            // Store result as comment meta
            update_comment_meta($comment_id, 'authenticity_score', $result['authenticity_score']);
            update_comment_meta($comment_id, 'risk_level', $result['risk_level']);
            update_comment_meta($comment_id, 'analysis_timestamp', current_time('mysql'));
            
            // Auto-moderate if high risk
            if ($this->should_auto_moderate($result)) {
                wp_set_comment_status($comment_id, 'hold');
                
                // Notify admin
                $this->notify_admin_suspicious_content($comment_id, $result);
            }
            
        } catch (Exception $e) {
            error_log('Content Authenticity API Error: ' . $e->getMessage());
        }
    }
    
    /**
     * Process WooCommerce product review
     */
    public function process_product_review($comment_data) {
        if (!$this->is_configured()) {
            return $comment_data;
        }
        
        try {
            $result = $this->analyze_content([
                'text' => $comment_data['comment_content'],
                'platform' => 'ecommerce',
                'content_type' => 'review',
                'rating' => $_POST['rating'] ?? null,
                'product_id' => $comment_data['comment_post_ID'],
                'user_profile' => [
                    'user_id' => get_current_user_id(),
                    'verified_user' => is_user_logged_in()
                ]
            ]);
            
            // Store authenticity data
            $comment_data['comment_meta']['authenticity_score'] = $result['authenticity_score'];
            $comment_data['comment_meta']['risk_level'] = $result['risk_level'];
            
            // Auto-moderate if necessary
            if ($this->should_auto_moderate($result)) {
                $comment_data['comment_approved'] = 0; // Hold for moderation
            }
            
        } catch (Exception $e) {
            error_log('Content Authenticity API Error: ' . $e->getMessage());
        }
        
        return $comment_data;
    }
    
    /**
     * Analyze content using the API
     */
    private function analyze_content($data) {
        $response = wp_remote_post($this->api_url . '/v1/analyze/content', [
            'headers' => [
                'Authorization' => 'Bearer ' . $this->api_key,
                'Content-Type' => 'application/json'
            ],
            'body' => json_encode($data),
            'timeout' => 30
        ]);
        
        if (is_wp_error($response)) {
            throw new Exception('API request failed: ' . $response->get_error_message());
        }
        
        $status_code = wp_remote_retrieve_response_code($response);
        if ($status_code !== 200) {
            throw new Exception('API returned error: ' . $status_code);
        }
        
        $body = wp_remote_retrieve_body($response);
        $data = json_decode($body, true);
        
        if (json_last_error() !== JSON_ERROR_NONE) {
            throw new Exception('Invalid JSON response from API');
        }
        
        return $data;
    }
    
    /**
     * Check if auto-moderation should be applied
     */
    private function should_auto_moderate($result) {
        $auto_moderate_threshold = floatval($this->settings['auto_moderate_threshold'] ?? 0.3);
        return $result['authenticity_score'] < $auto_moderate_threshold;
    }
    
    /**
     * Notify admin about suspicious content
     */
    private function notify_admin_suspicious_content($comment_id, $result) {
        if (!($this->settings['email_notifications'] ?? false)) {
            return;
        }
        
        $comment = get_comment($comment_id);
        $post = get_post($comment->comment_post_ID);
        
        $subject = 'Suspicious Content Detected - ' . get_bloginfo('name');
        $message = "A potentially fake comment has been detected and held for moderation:\n\n";
        $message .= "Post: {$post->post_title}\n";
        $message .= "Author: {$comment->comment_author}\n";
        $message .= "Risk Level: {$result['risk_level']}\n";
        $message .= "Authenticity Score: " . ($result['authenticity_score'] * 100) . "%\n\n";
        $message .= "Comment Content:\n{$comment->comment_content}\n\n";
        $message .= "Review in admin: " . admin_url("comment.php?action=editcomment&c={$comment_id}");
        
        wp_mail(get_option('admin_email'), $subject, $message);
    }
    
    /**
     * Add admin menu
     */
    public function add_admin_menu() {
        add_options_page(
            __('Content Authenticity', 'content-authenticity'),
            __('Content Authenticity', 'content-authenticity'),
            'manage_options',
            'content-authenticity',
            [$this, 'admin_page']
        );
    }
    
    /**
     * Initialize admin settings
     */
    public function admin_init() {
        register_setting('content_authenticity_settings', 'content_authenticity_settings');
        
        // Add settings sections
        add_settings_section(
            'api_settings',
            __('API Settings', 'content-authenticity'),
            [$this, 'api_settings_section'],
            'content-authenticity'
        );
        
        add_settings_section(
            'moderation_settings',
            __('Content Moderation', 'content-authenticity'),
            [$this, 'moderation_settings_section'],
            'content-authenticity'
        );
        
        // Add settings fields
        add_settings_field(
            'api_key',
            __('API Key', 'content-authenticity'),
            [$this, 'api_key_field'],
            'content-authenticity',
            'api_settings'
        );
        
        add_settings_field(
            'api_url',
            __('API URL', 'content-authenticity'),
            [$this, 'api_url_field'],
            'content-authenticity',
            'api_settings'
        );
        
        add_settings_field(
            'risk_threshold',
            __('Risk Threshold', 'content-authenticity'),
            [$this, 'risk_threshold_field'],
            'content-authenticity',
            'moderation_settings'
        );
        
        add_settings_field(
            'auto_moderate_threshold',
            __('Auto-Moderate Threshold', 'content-authenticity'),
            [$this, 'auto_moderate_threshold_field'],
            'content-authenticity',
            'moderation_settings'
        );
        
        add_settings_field(
            'real_time_checking',
            __('Real-time Checking', 'content-authenticity'),
            [$this, 'real_time_checking_field'],
            'content-authenticity',
            'moderation_settings'
        );
        
        add_settings_field(
            'email_notifications',
            __('Email Notifications', 'content-authenticity'),
            [$this, 'email_notifications_field'],
            'content-authenticity',
            'moderation_settings'
        );
    }
    
    /**
     * Admin page content
     */
    public function admin_page() {
        ?>
        <div class="wrap">
            <h1><?php _e('Content Authenticity Settings', 'content-authenticity'); ?></h1>
            
            <?php if (!$this->is_configured()): ?>
                <div class="notice notice-warning">
                    <p><?php _e('Please configure your API settings to enable content authenticity checking.', 'content-authenticity'); ?></p>
                </div>
            <?php endif; ?>
            
            <form method="post" action="options.php">
                <?php
                settings_fields('content_authenticity_settings');
                do_settings_sections('content-authenticity');
                submit_button();
                ?>
            </form>
            
            <div class="authenticity-stats" id="authenticity-stats">
                <h2><?php _e('Authenticity Statistics', 'content-authenticity'); ?></h2>
                <div class="stats-loading"><?php _e('Loading statistics...', 'content-authenticity'); ?></div>
            </div>
            
            <div class="authenticity-tools">
                <h2><?php _e('Tools', 'content-authenticity'); ?></h2>
                <p>
                    <button type="button" class="button" id="test-api-connection">
                        <?php _e('Test API Connection', 'content-authenticity'); ?>
                    </button>
                    <button type="button" class="button" id="bulk-analyze-comments">
                        <?php _e('Analyze Existing Comments', 'content-authenticity'); ?>
                    </button>
                </p>
                <div id="tool-results"></div>
            </div>
        </div>
        <?php
    }
    
    /**
     * Settings field callbacks
     */
    public function api_settings_section() {
        echo '<p>' . __('Configure your Content Authenticity API settings.', 'content-authenticity') . '</p>';
    }
    
    public function moderation_settings_section() {
        echo '<p>' . __('Configure how content moderation should work.', 'content-authenticity') . '</p>';
    }
    
    public function api_key_field() {
        $value = $this->settings['api_key'] ?? '';
        echo '<input type="password" name="content_authenticity_settings[api_key]" value="' . esc_attr($value) . '" class="regular-text" />';
        echo '<p class="description">' . __('Your Content Authenticity API key.', 'content-authenticity') . '</p>';
    }
    
    public function api_url_field() {
        $value = $this->settings['api_url'] ?? 'https://api.contentauthenticity.com';
        echo '<input type="url" name="content_authenticity_settings[api_url]" value="' . esc_attr($value) . '" class="regular-text" />';
        echo '<p class="description">' . __('API endpoint URL.', 'content-authenticity') . '</p>';
    }
    
    public function risk_threshold_field() {
        $value = $this->settings['risk_threshold'] ?? 0.7;
        echo '<input type="number" name="content_authenticity_settings[risk_threshold]" value="' . esc_attr($value) . '" min="0" max="1" step="0.1" class="small-text" />';
        echo '<p class="description">' . __('Minimum authenticity score to allow content (0.0 - 1.0).', 'content-authenticity') . '</p>';
    }
    
    public function auto_moderate_threshold_field() {
        $value = $this->settings['auto_moderate_threshold'] ?? 0.3;
        echo '<input type="number" name="content_authenticity_settings[auto_moderate_threshold]" value="' . esc_attr($value) . '" min="0" max="1" step="0.1" class="small-text" />';
        echo '<p class="description">' . __('Automatically hold content for moderation below this score.', 'content-authenticity') . '</p>';
    }
    
    public function real_time_checking_field() {
        $checked = $this->settings['real_time_checking'] ?? false;
        echo '<input type="checkbox" name="content_authenticity_settings[real_time_checking]" value="1" ' . checked(1, $checked, false) . ' />';
        echo '<label>' . __('Block suspicious content before publication', 'content-authenticity') . '</label>';
    }
    
    public function email_notifications_field() {
        $checked = $this->settings['email_notifications'] ?? false;
        echo '<input type="checkbox" name="content_authenticity_settings[email_notifications]" value="1" ' . checked(1, $checked, false) . ' />';
        echo '<label>' . __('Send email notifications for suspicious content', 'content-authenticity') . '</label>';
    }
    
    /**
     * Load admin scripts
     */
    public function admin_scripts($hook) {
        if ($hook !== 'settings_page_content-authenticity') {
            return;
        }
        
        wp_enqueue_script(
            'content-authenticity-admin',
            CONTENT_AUTHENTICITY_PLUGIN_URL . 'assets/admin.js',
            ['jquery'],
            CONTENT_AUTHENTICITY_VERSION,
            true
        );
        
        wp_localize_script('content-authenticity-admin', 'contentAuthenticityAdmin', [
            'ajaxUrl' => admin_url('admin-ajax.php'),
            'nonce' => wp_create_nonce('content_authenticity_admin')
        ]);
    }
    
    /**
     * Load frontend scripts
     */
    public function frontend_scripts() {
        if (!is_singular() || !comments_open()) {
            return;
        }
        
        wp_enqueue_script(
            'content-authenticity-frontend',
            CONTENT_AUTHENTICITY_PLUGIN_URL . 'assets/frontend.js',
            ['jquery'],
            CONTENT_AUTHENTICITY_VERSION,
            true
        );
    }
    
    /**
     * Add real-time validation to comment forms
     */
    public function add_real_time_validation() {
        if (!is_singular() || !comments_open() || !($this->settings['real_time_validation'] ?? false)) {
            return;
        }
        
        ?>
        <script>
        (function($) {
            if (typeof ContentAuthenticitySDK !== 'undefined') {
                const sdk = new ContentAuthenticitySDK({
                    apiKey: '<?php echo esc_js($this->api_key); ?>',
                    baseUrl: '<?php echo esc_js($this->api_url); ?>'
                });
                
                const commentField = document.getElementById('comment');
                if (commentField) {
                    sdk.attachRealTimeValidation(commentField, {
                        platform: 'blog',
                        contentType: 'comment',
                        minLength: 10,
                        debounceMs: 2000
                    });
                }
            }
        })(jQuery);
        </script>
        <?php
    }
    
    /**
     * Add dashboard widget
     */
    public function add_dashboard_widget() {
        if (!$this->is_configured()) {
            return;
        }
        
        wp_add_dashboard_widget(
            'content_authenticity_widget',
            __('Content Authenticity', 'content-authenticity'),
            [$this, 'dashboard_widget_content']
        );
    }
    
    /**
     * Dashboard widget content
     */
    public function dashboard_widget_content() {
        $stats = $this->get_recent_stats();
        
        ?>
        <div class="authenticity-widget">
            <p><strong><?php _e('Recent Activity (Last 7 Days)', 'content-authenticity'); ?></strong></p>
            <ul>
                <li><?php printf(__('Comments Analyzed: %d', 'content-authenticity'), $stats['analyzed']); ?></li>
                <li><?php printf(__('High Risk Content: %d', 'content-authenticity'), $stats['high_risk']); ?></li>
                <li><?php printf(__('Auto-Moderated: %d', 'content-authenticity'), $stats['moderated']); ?></li>
            </ul>
            <p><a href="<?php echo admin_url('options-general.php?page=content-authenticity'); ?>" class="button"><?php _e('View Settings', 'content-authenticity'); ?></a></p>
        </div>
        <?php
    }
    
    /**
     * Register REST API routes
     */
    public function register_rest_routes() {
        register_rest_route('content-authenticity/v1', '/analyze', [
            'methods' => 'POST',
            'callback' => [$this, 'rest_analyze_content'],
            'permission_callback' => [$this, 'rest_permission_check']
        ]);
    }
    
    /**
     * REST API content analysis endpoint
     */
    public function rest_analyze_content($request) {
        try {
            $params = $request->get_json_params();
            $result = $this->analyze_content($params);
            return new WP_REST_Response($result, 200);
        } catch (Exception $e) {
            return new WP_Error('analysis_failed', $e->getMessage(), ['status' => 500]);
        }
    }
    
    /**
     * REST API permission check
     */
    public function rest_permission_check() {
        return current_user_can('moderate_comments');
    }
    
    /**
     * AJAX: Test API connection
     */
    public function test_api_connection() {
        check_ajax_referer('content_authenticity_admin', 'nonce');
        
        try {
            $response = wp_remote_get($this->api_url . '/health', [
                'headers' => [
                    'Authorization' => 'Bearer ' . $this->api_key
                ],
                'timeout' => 10
            ]);
            
            if (is_wp_error($response)) {
                wp_send_json_error('Connection failed: ' . $response->get_error_message());
            }
            
            $status_code = wp_remote_retrieve_response_code($response);
            if ($status_code === 200) {
                wp_send_json_success('API connection successful!');
            } else {
                wp_send_json_error('API returned error: ' . $status_code);
            }
            
        } catch (Exception $e) {
            wp_send_json_error('Test failed: ' . $e->getMessage());
        }
    }
    
    /**
     * Check if plugin is properly configured
     */
    private function is_configured() {
        return !empty($this->api_key) && !empty($this->api_url);
    }
    
    /**
     * Get recent authenticity statistics
     */
    private function get_recent_stats() {
        global $wpdb;
        
        $seven_days_ago = date('Y-m-d H:i:s', strtotime('-7 days'));
        
        $analyzed = $wpdb->get_var($wpdb->prepare(
            "SELECT COUNT(*) FROM {$wpdb->commentmeta} 
             WHERE meta_key = 'authenticity_score' 
             AND comment_id IN (
                 SELECT comment_ID FROM {$wpdb->comments} 
                 WHERE comment_date >= %s
             )",
            $seven_days_ago
        ));
        
        $high_risk = $wpdb->get_var($wpdb->prepare(
            "SELECT COUNT(*) FROM {$wpdb->commentmeta} cm
             INNER JOIN {$wpdb->comments} c ON cm.comment_id = c.comment_ID
             WHERE cm.meta_key = 'risk_level' 
             AND cm.meta_value IN ('high', 'critical')
             AND c.comment_date >= %s",
            $seven_days_ago
        ));
        
        $moderated = $wpdb->get_var($wpdb->prepare(
            "SELECT COUNT(*) FROM {$wpdb->comments} 
             WHERE comment_approved = '0' 
             AND comment_date >= %s
             AND comment_ID IN (
                 SELECT comment_id FROM {$wpdb->commentmeta}
                 WHERE meta_key = 'authenticity_score'
             )",
            $seven_days_ago
        ));
        
        return [
            'analyzed' => intval($analyzed),
            'high_risk' => intval($high_risk),
            'moderated' => intval($moderated)
        ];
    }
}

// Initialize the plugin
new ContentAuthenticity();

/**
 * Plugin activation hook
 */
register_activation_hook(__FILE__, function() {
    // Create default settings
    $default_settings = [
        'api_url' => 'https://api.contentauthenticity.com',
        'risk_threshold' => 0.7,
        'auto_moderate_threshold' => 0.3,
        'real_time_checking' => false,
        'email_notifications' => true
    ];
    
    add_option('content_authenticity_settings', $default_settings);
});

/**
 * Plugin deactivation hook
 */
register_deactivation_hook(__FILE__, function() {
    // Clean up scheduled events if any
    wp_clear_scheduled_hook('content_authenticity_cleanup');
});
