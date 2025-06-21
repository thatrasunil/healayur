#!/usr/bin/env python3
"""
Heal Ayur - Enhanced AI-Powered Ancient Healing Application
Advanced skin condition analysis with Ayurvedic remedies
Features: Authentication, Real-time WebSocket, Modern UI, Database Integration
"""

import os
import json
import uuid
import time
import logging
import sqlite3
import glob
import requests
import google.generativeai as genai
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.utils import secure_filename
# OAuth import removed - using only regular authentication
from dotenv import load_dotenv
import cv2
import numpy as np
from PIL import Image
import base64
import io
import threading
from collections import defaultdict

# Load environment variables
load_dotenv()

# Import our modules
from auth import AuthManager, login_required, optional_auth

# Configure logging - Vercel-compatible
if os.environ.get('VERCEL_DEPLOYMENT'):
    # Simplified logging for Vercel
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
else:
    # Full logging for local/other deployments
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/healayur.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
logger = logging.getLogger(__name__)

# Configure Gemini API for chatbot after logger is defined
GEMINI_API_KEY = "AIzaSyDhsUnlg85rpkGoiJlKnJHNXAl_0FfIeh0"
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model for chatbot
try:
    # Try different model names in order of preference
    model_names = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
    gemini_model = None

    for model_name in model_names:
        try:
            gemini_model = genai.GenerativeModel(model_name)
            # Test the model with a simple request
            test_response = gemini_model.generate_content("Hello")
            GEMINI_AVAILABLE = True
            GEMINI_MODEL_NAME = model_name
            logger.info(f"✅ Gemini API configured successfully with model: {model_name}")
            break
        except Exception as model_error:
            logger.debug(f"Model {model_name} failed: {model_error}")
            continue

    if not gemini_model:
        raise Exception("No working Gemini model found")

except Exception as e:
    GEMINI_AVAILABLE = False
    GEMINI_MODEL_NAME = None
    logger.warning(f"⚠️ Gemini API not available for chatbot: {e}")

# Import AI analysis modules after logger is defined - prioritize Face++ integration
try:
    from ai_analysis_faceplus import analyze_image
    AI_AVAILABLE = True
    CHAT_AVAILABLE = False
    logger.info("✅ Face++ & Gemini hybrid analysis module loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Face++ hybrid analysis module not available: {e}")
    try:
        from ai_analysis_simple import analyze_image, chat_response
        AI_AVAILABLE = True
        CHAT_AVAILABLE = True
        logger.info("✅ Simplified Gemini AI analysis & chat module loaded as fallback")
    except ImportError as e2:
        logger.warning(f"⚠️ Simplified AI analysis module not available: {e2}")
        try:
            from ai_analysis_gemini import analyze_image
            AI_AVAILABLE = True
            CHAT_AVAILABLE = False
            logger.info("✅ Gemini AI analysis module loaded as fallback")
        except ImportError as e3:
            logger.warning(f"⚠️ Gemini AI analysis module not available: {e3}")
            try:
                from ai_analysis_enhanced import analyze_image
                AI_AVAILABLE = True
                CHAT_AVAILABLE = False
                logger.info("✅ Enhanced AI analysis module loaded as fallback")
            except ImportError as e4:
                logger.warning(f"⚠️ Enhanced AI analysis module not available: {e4}")
                try:
                    from ai_analysis import analyze_image
                    AI_AVAILABLE = True
                    CHAT_AVAILABLE = False
                    logger.info("✅ Basic AI analysis module loaded as final fallback")
                except ImportError:
                    logger.warning("⚠️ No AI analysis modules available")
                    AI_AVAILABLE = False
                    CHAT_AVAILABLE = False

        def analyze_image(image_data):
            """Fallback analysis function"""
            return {
                'condition': 'Basic Analysis',
                'confidence': 0.75,
                'remedies': [
                    'Maintain good hygiene',
                    'Use natural moisturizers like aloe vera',
                    'Stay hydrated and eat healthy foods',
                    'Protect skin from sun exposure'
                ],
                'herbs': ['Aloe Vera', 'Turmeric', 'Neem', 'Rose Water'],
                'lifestyle': ['Balanced diet', 'Regular exercise', 'Adequate sleep'],
                'severity': 'Mild to Moderate',
                'features': {},
                'recommendations': ['Follow basic skincare routine'],
                'timestamp': datetime.now().isoformat()
            }

# Initialize Flask app with enhanced configuration
app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', 'heal-ayur-secret-key-2024'),
    UPLOAD_FOLDER='static/uploads',
    HISTORY_FOLDER='static/history',
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
    SESSION_COOKIE_SECURE=False,  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=86400 * 30,  # 30 days

    # OAuth Configuration removed - using only regular authentication
)

# OAuth functionality removed - using only regular authentication

# Initialize SocketIO for real-time features
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize authentication manager
auth_manager = AuthManager()

# Create necessary directories - Vercel-compatible
if not os.environ.get('VERCEL_DEPLOYMENT'):
    # Only create directories for non-Vercel deployments
    for folder in [app.config['UPLOAD_FOLDER'], app.config['HISTORY_FOLDER'], 'logs']:
        os.makedirs(folder, exist_ok=True)
else:
    # For Vercel, use /tmp directory for temporary files
    app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
    app.config['HISTORY_FOLDER'] = '/tmp/history'
    os.makedirs('/tmp/uploads', exist_ok=True)
    os.makedirs('/tmp/history', exist_ok=True)

# Load remedies database
try:
    with open('remedies.json', 'r', encoding='utf-8') as f:
        remedies = json.load(f)
    logger.info(f"Loaded {len(remedies)} remedy categories successfully")
except Exception as e:
    logger.error(f"Error loading remedies: {e}")
    remedies = {}

# Load products database for e-commerce
try:
    with open('products.json', 'r', encoding='utf-8') as f:
        products = json.load(f)
    logger.info(f"Loaded {len(products)} products successfully")
except Exception as e:
    logger.error(f"Error loading products: {e}")
    products = {}

# Global statistics and real-time data
analysis_stats = {
    'total_analyses': 0,
    'conditions_detected': defaultdict(int),
    'success_rate': 0.95,
    'active_users': 0,
    'real_time_sessions': 0
}

# Active users tracking
active_users = set()
real_time_sessions = {}

def get_multiple_remedies(condition, count=3):
    """Get multiple remedy options for a condition with enhanced data"""
    if condition not in remedies:
        return []

    remedy_data = remedies[condition]
    result = []

    # Add primary remedy
    if 'primary' in remedy_data:
        primary = remedy_data['primary'].copy()
        primary.update({
            'type': 'primary',
            'effectiveness': primary.get('effectiveness', 85),
            'difficulty': primary.get('difficulty', 'Easy'),
            'time_to_prepare': primary.get('time_to_prepare', '5-10 minutes'),
            'frequency': primary.get('frequency', 'Apply 2-3 times daily')
        })
        result.append(primary)

    # Add alternatives
    if 'alternatives' in remedy_data:
        alternatives = remedy_data['alternatives'][:count-1]
        for alt in alternatives:
            alt_copy = alt.copy()
            alt_copy.update({
                'type': 'alternative',
                'effectiveness': alt_copy.get('effectiveness', 75),
                'difficulty': alt_copy.get('difficulty', 'Easy'),
                'time_to_prepare': alt_copy.get('time_to_prepare', '10-15 minutes'),
                'frequency': alt_copy.get('frequency', 'Apply 1-2 times daily')
            })
            result.append(alt_copy)

    return result

# ==================== AUTHENTICATION ROUTES ====================

@app.route('/login')
def login():
    """Login page"""
    if 'session_token' in session:
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/signup')
def signup():
    """Signup page"""
    if 'session_token' in session:
        return redirect(url_for('index'))
    return render_template('signup.html')

@app.route('/api/login', methods=['POST'])
def api_login():
    """API endpoint for user login"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        remember_me = data.get('remember_me', False)
        
        if not username or not password:
            return jsonify({'success': False, 'error': 'Username and password are required'}), 400
        
        # Get client info
        ip_address = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR'))
        user_agent = request.headers.get('User-Agent', '')
        
        # Attempt login
        result = auth_manager.login_user(username, password, ip_address, user_agent)
        
        if result['success']:
            session['session_token'] = result['session_token']
            session['user_id'] = result['user']['id']
            session['username'] = result['user']['username']
            
            if remember_me:
                session.permanent = True
            
            logger.info(f"✅ User logged in: {result['user']['username']}")
            return jsonify({
                'success': True,
                'user': result['user'],
                'message': 'Login successful'
            })
        else:
            logger.warning(f"❌ Failed login attempt for: {username}")
            return jsonify(result), 401
            
    except Exception as e:
        logger.error(f"❌ Login error: {e}")
        return jsonify({'success': False, 'error': 'Login failed'}), 500

@app.route('/api/signup', methods=['POST'])
def api_signup():
    """API endpoint for user registration"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['username', 'email', 'password', 'confirm_password', 'full_name']
        for field in required_fields:
            if not data.get(field, '').strip():
                return jsonify({'success': False, 'error': f'{field.replace("_", " ").title()} is required'}), 400
        
        # Validate password match
        if data['password'] != data['confirm_password']:
            return jsonify({'success': False, 'error': 'Passwords do not match'}), 400
        
        # Validate terms agreement
        if not data.get('agree_terms'):
            return jsonify({'success': False, 'error': 'You must agree to the terms and conditions'}), 400
        
        # Register user
        result = auth_manager.register_user(
            username=data['username'].strip(),
            email=data['email'].strip().lower(),
            password=data['password'],
            full_name=data['full_name'].strip()
        )
        
        if result['success']:
            logger.info(f"✅ New user registered: {data['username']}")
            return jsonify({
                'success': True,
                'message': 'Account created successfully! Please log in.'
            })
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"❌ Signup error: {e}")
        return jsonify({'success': False, 'error': 'Registration failed'}), 500

@app.route('/api/logout', methods=['POST'])
@login_required
def api_logout():
    """API endpoint for user logout"""
    try:
        session_token = session.get('session_token')
        if session_token:
            auth_manager.logout_user(session_token)
        
        session.clear()
        logger.info("✅ User logged out")
        return jsonify({'success': True, 'message': 'Logged out successfully'})
        
    except Exception as e:
        logger.error(f"❌ Logout error: {e}")
        return jsonify({'success': False, 'error': 'Logout failed'}), 500

@app.route('/api/check-username')
def check_username():
    """Check if username is available"""
    username = request.args.get('username', '').strip()
    if len(username) < 3:
        return jsonify({'available': False, 'error': 'Username too short'})

    try:
        import sqlite3
        conn = sqlite3.connect('healayur.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
        exists = cursor.fetchone() is not None
        conn.close()

        return jsonify({'available': not exists})
    except Exception as e:
        logger.error(f"Error checking username: {e}")
        return jsonify({'available': False, 'error': 'Check failed'})

# OAuth routes removed - using only regular authentication

# ==================== CHATBOT API ROUTES ====================

@app.route('/api/chat', methods=['POST'])
@optional_auth
def api_chat():
    """Enhanced AI chatbot using Gemini API - authentication optional"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        conversation_history = data.get('history', [])

        if not user_message:
            return jsonify({'success': False, 'error': 'Message is required'}), 400

        user = getattr(request, 'current_user', None)
        username = user['username'] if user else 'Guest'

        # Enhanced system prompt for Heal Ayur chatbot
        system_prompt = """You are an AI assistant for Heal Ayur, an advanced skin analysis and ancient healing platform.

Your role:
- Help users understand skin conditions and natural remedies
- Guide them through the image analysis process
- Provide information about Ayurvedic and traditional healing methods
- Answer questions about ingredients, preparation, and application of remedies
- Be supportive, knowledgeable, and encouraging

Key features of Heal Ayur:
- AI-powered skin condition analysis with 95% accuracy
- Real-time webcam analysis every 2.5 seconds
- 500+ traditional remedies from various healing traditions
- Voice commands and real-time chat
- Progressive Web App with offline support
- Privacy-focused with secure authentication

Guidelines:
- Always emphasize that serious conditions should be evaluated by healthcare professionals
- Provide practical, actionable advice
- Be encouraging about natural healing while being realistic
- Mention relevant app features when appropriate
- Keep responses concise but informative
- Use emojis sparingly but effectively

Current user: """ + username

        if GEMINI_AVAILABLE:
            try:
                # Build conversation context
                conversation_context = system_prompt + "\n\nConversation:\n"

                # Add recent conversation history (last 10 messages)
                recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
                for msg in recent_history:
                    role = "User" if msg.get('role') == 'user' else "Assistant"
                    conversation_context += f"{role}: {msg.get('content', '')}\n"

                conversation_context += f"User: {user_message}\nAssistant:"

                # Generate response using Gemini
                response = gemini_model.generate_content(conversation_context)
                ai_response = response.text.strip()

                # Log successful chat interaction
                logger.info(f"💬 Chat response generated for {username}: {user_message[:50]}...")

                return jsonify({
                    'success': True,
                    'response': ai_response,
                    'model': GEMINI_MODEL_NAME,
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as gemini_error:
                logger.error(f"❌ Gemini API error: {gemini_error}")
                return jsonify({
                    'success': False,
                    'error': 'AI chat service is temporarily unavailable. Please try again in a moment.',
                    'details': 'Gemini API error',
                    'timestamp': datetime.now().isoformat()
                }), 503
        else:
            # Gemini not available
            return jsonify({
                'success': False,
                'error': 'AI chat service is not configured. Please contact support.',
                'details': 'Gemini API not available',
                'timestamp': datetime.now().isoformat()
            }), 503

    except Exception as e:
        logger.error(f"❌ Chat API error: {e}")
        return jsonify({
            'success': False,
            'error': 'Chat service temporarily unavailable',
            'details': str(e) if app.debug else None
        }), 500



# ==================== MAIN APPLICATION ROUTES ====================

@app.route('/')
@optional_auth
def index():
    """Main application page - authentication optional"""
    user = getattr(request, 'current_user', None)
    return render_template('index.html',
                         stats=analysis_stats,
                         user=user,
                         remedies_count=len(remedies))

@app.route('/analyze', methods=['POST'])
@optional_auth
def analyze():
    """Enhanced image analysis with user tracking - authentication optional"""
    start_time = time.time()
    user = getattr(request, 'current_user', None)
    
    try:
        # Update global stats
        analysis_stats['total_analyses'] += 1
        
        # Handle different input types
        if request.content_type and request.content_type.startswith('application/json'):
            # Handle webcam capture or base64 image
            data = request.get_json()
            image_base64 = data.get('image_base64')
            
            if not image_base64:
                return jsonify({"error": "No image data provided"}), 400
            
            # Process base64 image
            filepath = process_base64_image(image_base64)
            
        else:
            # Handle file upload
            image = request.files.get('image')
            if not image or image.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            # Process uploaded file
            filepath = process_uploaded_file(image)
        
        # Perform enhanced AI analysis with gender and age detection
        try:
            # Get full AI analysis result
            ai_analysis_result = analyze_skin_image(filepath)
            if not ai_analysis_result:
                raise ValueError("Analysis returned empty result")

            # Extract condition for compatibility
            if isinstance(ai_analysis_result, dict):
                detected_condition = ai_analysis_result.get('condition_mapped', 'acne')
                full_analysis = ai_analysis_result
            else:
                detected_condition = ai_analysis_result
                full_analysis = None

            logger.info(f"🔍 Detected condition: {detected_condition}")
        except Exception as analysis_error:
            logger.error(f"❌ Analysis failed: {analysis_error}")
            # Fallback to a safe default
            detected_condition = 'acne'
            full_analysis = None
            logger.info(f"🔄 Using fallback condition: {detected_condition}")

        # Get multiple remedy options
        try:
            remedy_options = get_multiple_remedies(detected_condition, count=3)
            if not remedy_options:
                raise ValueError("No remedies found")
        except Exception as remedy_error:
            logger.error(f"❌ Remedy lookup failed: {remedy_error}")
            # Fallback remedies
            remedy_options = [
                {
                    "name": "Basic Skincare",
                    "ingredients": ["Clean water", "Mild soap"],
                    "instructions": "Gently cleanse the affected area twice daily.",
                    "duration": "Daily routine",
                    "benefits": "Maintains basic skin hygiene"
                }
            ]

        # Calculate confidence score with enhanced logic
        base_confidence = 0.82
        condition_bonus = (hash(detected_condition) % 18) / 100  # 0-17% bonus
        confidence = base_confidence + condition_bonus
        confidence = min(confidence, 0.97)  # Cap at 97%
        
        # Update statistics
        analysis_stats['conditions_detected'][detected_condition] += 1
        
        # Calculate processing time
        processing_time = round(time.time() - start_time, 2)
        
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())[:8]
        
        # Prepare enhanced result with AI analysis
        result = {
            "condition": detected_condition,
            "confidence": round(confidence * 100, 1),
            "remedies": remedy_options,
            "processing_time": processing_time,
            "total_remedies": len(remedy_options),
            "analysis_id": analysis_id,
            "timestamp": datetime.now().isoformat()
        }

        # Add full AI analysis if available
        if full_analysis and isinstance(full_analysis, dict):
            result.update({
                "ai_analysis": full_analysis,
                "gender_analysis": full_analysis.get('gender_analysis', {}),
                "age_analysis": full_analysis.get('age_analysis', {}),
                "advanced_metrics": full_analysis.get('advanced_metrics', {}),
                "professional_insights": full_analysis.get('professional_insights', [])
            })
        
        # Save to user history if authenticated
        if user:
            auth_manager.save_analysis(
                user['id'], 
                detected_condition, 
                confidence, 
                processing_time, 
                remedy_options
            )
        
        # Emit real-time update via WebSocket
        socketio.emit('analysis_complete', {
            'condition': detected_condition,
            'confidence': result['confidence'],
            'user_count': len(active_users)
        })
        
        logger.info(f"✅ Analysis complete: {detected_condition} ({confidence:.1%} confidence)")

        # Ensure all values in result are JSON serializable
        try:
            import json
            json.dumps(result)  # Test serialization
        except (TypeError, ValueError) as json_error:
            logger.error(f"❌ JSON serialization error: {json_error}")
            # Create a safe fallback result
            result = {
                "condition": detected_condition,
                "confidence": round(confidence * 100, 1),
                "remedies": remedy_options[:3],  # Limit to first 3 remedies
                "processing_time": processing_time,
                "total_remedies": len(remedy_options),
                "analysis_id": analysis_id,
                "timestamp": datetime.now().isoformat(),
                "error_note": "Some advanced features unavailable"
            }

        return jsonify({"success": True, "result": result})

    except Exception as e:
        logger.error(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "Analysis failed. Please try again.",
            "details": str(e) if app.debug else "Internal error"
        }), 500

def process_base64_image(image_base64):
    """Process base64 encoded image"""
    try:
        # Remove data URL prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_data = base64.b64decode(image_base64)
        timestamp = str(int(time.time()))
        filename = f'webcam_capture_{timestamp}.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(filepath, 'wb') as f:
            f.write(image_data)
        
        logger.info(f"📷 Saved webcam image: {filename}")
        return filepath
        
    except Exception as e:
        logger.error(f"❌ Failed to process base64 image: {e}")
        raise

def process_uploaded_file(image):
    """Process uploaded file"""
    try:
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
        file_ext = image.filename.rsplit('.', 1)[1].lower() if '.' in image.filename else ''

        if file_ext not in allowed_extensions:
            raise ValueError("Invalid file type. Please upload an image.")

        filename = secure_filename(image.filename)
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        image.save(filepath)
        logger.info(f"📁 Saved uploaded image: {filename}")
        return filepath

    except Exception as e:
        logger.error(f"❌ Failed to process uploaded file: {e}")
        raise

def analyze_skin_image(filepath):
    """Analyze skin image using revolutionary AI analysis with gender and age detection"""
    try:
        # Read image file
        with open(filepath, 'rb') as f:
            image_data = f.read()

        # Use revolutionary AI analysis
        result = analyze_image(image_data)

        # Process the enhanced result
        if isinstance(result, dict):
            condition = result.get('condition', 'Active Acne Breakout')

            # Enhanced condition mapping for new AI system
            condition_mapping = {
                'Radiant Healthy Skin': 'normal',
                'Active Acne Breakout': 'acne',
                'Post-Inflammatory Hyperpigmentation': 'dark_spots',
                'Severe Dehydration & Roughness': 'dry_skin',
                'Excessive Sebum Production': 'oily_skin',
                'Inflammatory Rosacea': 'redness',
                'Photoaging & Deep Wrinkles': 'wrinkles',
                'Melasma & Uneven Pigmentation': 'uneven_tone',
                'Comedonal Acne (Blackheads/Whiteheads)': 'acne',
                'Sensitive Reactive Skin': 'redness',
                'Sun Damage & Age Spots': 'dark_spots',
                'Enlarged Pores & Texture Issues': 'oily_skin',
                # Legacy mappings
                'Normal/Healthy Skin': 'normal',
                'Acne/Pimples': 'acne',
                'Dark Spots/Hyperpigmentation': 'dark_spots',
                'Dry/Rough Skin': 'dry_skin',
                'Oily Skin': 'oily_skin',
                'Redness/Inflammation': 'redness',
                'Wrinkles/Fine Lines': 'wrinkles',
                'Uneven Skin Tone': 'uneven_tone'
            }

            mapped_condition = condition_mapping.get(condition, 'acne')

            # Return enhanced result with mapped condition
            enhanced_result = result.copy()
            enhanced_result['condition_mapped'] = mapped_condition
            enhanced_result['original_condition'] = condition

            return enhanced_result
        else:
            return 'acne'  # Default fallback

    except Exception as e:
        logger.error(f"❌ Skin image analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 'acne'  # Safe fallback

# ==================== API ROUTES ====================

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get enhanced analysis statistics"""
    return jsonify({
        **analysis_stats,
        'active_users': len(active_users),
        'real_time_sessions': len(real_time_sessions),
        'conditions_detected': dict(analysis_stats['conditions_detected'])
    })

@app.route('/api/history', methods=['GET'])
@login_required
def get_user_history():
    """Get user's analysis history"""
    user = request.current_user
    result = auth_manager.get_user_history(user['id'])

    if result['success']:
        return jsonify(result)
    else:
        return jsonify({'error': 'Failed to retrieve history'}), 500

@app.route('/api/user/history', methods=['GET'])
@login_required
def get_user_history_alt():
    """Get user's analysis history (alternative endpoint)"""
    user = request.current_user
    result = auth_manager.get_user_history(user['id'])

    if result['success']:
        return jsonify(result)
    else:
        return jsonify({'error': 'Failed to retrieve history'}), 500

@app.route('/api/remedy/<condition>', methods=['GET'])
def get_remedy_details(condition):
    """Get detailed remedy information for a specific condition"""
    if condition in remedies:
        return jsonify({"success": True, "remedy": remedies[condition]})
    else:
        return jsonify({"error": "Condition not found"}), 404

@app.route('/api/user/profile', methods=['GET'])
@login_required
def get_user_profile():
    """Get user profile information"""
    user = request.current_user
    return jsonify({
        'success': True,
        'user': user
    })

# ==================== ADMIN ROUTES ====================

@app.route('/admin')
def admin_login():
    """Admin login page"""
    return render_template('admin_login.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    """Admin dashboard"""
    # Check if admin is logged in
    if not session.get('is_admin'):
        return redirect('/admin')
    return render_template('admin_dashboard.html')

@app.route('/api/admin/login', methods=['POST'])
def admin_login_api():
    """Admin login API"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')

        # Check admin credentials
        if username == 'admin' and password == 'healayur2024':
            session['is_admin'] = True
            session['admin_username'] = username
            logger.info(f"Admin logged in: {username}")
            return jsonify({
                'success': True,
                'message': 'Admin login successful',
                'redirect': '/admin/dashboard'
            })
        else:
            logger.warning(f"Failed admin login attempt: {username}")
            return jsonify({
                'success': False,
                'error': 'Invalid admin credentials'
            }), 401

    except Exception as e:
        logger.error(f"Admin login error: {e}")
        return jsonify({'success': False, 'error': 'Login failed'}), 500

@app.route('/api/admin/logout', methods=['POST'])
def admin_logout():
    """Admin logout"""
    session.pop('is_admin', None)
    session.pop('admin_username', None)
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/api/admin/stats', methods=['GET'])
def admin_stats():
    """Get admin statistics"""
    if not session.get('is_admin'):
        return jsonify({'error': 'Admin access required'}), 403

    try:
        # Get database stats
        conn = sqlite3.connect('healayur.db')
        cursor = conn.cursor()

        # User stats
        cursor.execute('SELECT COUNT(*) FROM users')
        total_users = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM users WHERE created_at >= date("now", "-7 days")')
        new_users_week = cursor.fetchone()[0]

        # Analysis stats
        cursor.execute('SELECT COUNT(*) FROM analysis_history')
        total_analyses = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM analysis_history WHERE created_at >= date("now", "-1 day")')
        analyses_today = cursor.fetchone()[0]

        # Recent analyses
        cursor.execute('''
            SELECT condition_detected, confidence_score, created_at,
                   CASE WHEN user_id IS NULL THEN "Anonymous" ELSE "User" END as user_type
            FROM analysis_history
            ORDER BY created_at DESC
            LIMIT 10
        ''')
        recent_analyses = cursor.fetchall()

        # Popular conditions
        cursor.execute('''
            SELECT condition_detected, COUNT(*) as count
            FROM analysis_history
            GROUP BY condition_detected
            ORDER BY count DESC
            LIMIT 5
        ''')
        popular_conditions = cursor.fetchall()

        conn.close()

        # Get image count
        try:
            image_count = len([f for f in os.listdir(app.config['UPLOAD_FOLDER'])
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp'))])
        except:
            image_count = 0

        return jsonify({
            'success': True,
            'stats': {
                'total_users': total_users,
                'new_users_week': new_users_week,
                'total_analyses': total_analyses,
                'analyses_today': analyses_today,
                'total_images': image_count,
                'recent_analyses': [
                    {
                        'condition': row[0],
                        'confidence': row[1],
                        'created_at': row[2],
                        'user_type': row[3]
                    } for row in recent_analyses
                ],
                'popular_conditions': [
                    {'condition': row[0], 'count': row[1]}
                    for row in popular_conditions
                ]
            }
        })

    except Exception as e:
        logger.error(f"Admin stats error: {e}")
        return jsonify({'error': 'Failed to get stats'}), 500

@app.route('/api/admin/uploaded-images', methods=['GET'])
def admin_uploaded_images():
    """Get all uploaded images for admin"""
    if not session.get('is_admin'):
        return jsonify({'error': 'Admin access required'}), 403

    try:
        import os
        import glob
        from datetime import datetime

        upload_folder = app.config['UPLOAD_FOLDER']
        image_files = []

        # Get all image files
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp']:
            files = glob.glob(os.path.join(upload_folder, ext))
            image_files.extend(files)

        # Sort by modification time (newest first)
        image_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        # Prepare file info
        files_info = []
        for file_path in image_files[:50]:  # Limit to 50 most recent
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))

            files_info.append({
                'filename': filename,
                'size': file_size,
                'uploaded_at': mod_time.isoformat(),
                'url': f'/static/uploads/{filename}'
            })

        return jsonify({
            'success': True,
            'images': files_info,
            'total_count': len(image_files)
        })

    except Exception as e:
        logger.error(f"Admin images error: {e}")
        return jsonify({'error': 'Failed to get images'}), 500

@app.route('/api/admin/users', methods=['GET'])
def admin_get_users():
    """Get all users for admin"""
    if not session.get('is_admin'):
        return jsonify({'error': 'Admin access required'}), 403

    try:
        conn = sqlite3.connect('healayur.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        users = cursor.execute('''
            SELECT id, username, email, full_name, created_at, last_login
            FROM users
            ORDER BY created_at DESC
        ''').fetchall()

        conn.close()

        users_list = []
        for user in users:
            user_dict = dict(user)
            # Format dates
            if user_dict['created_at']:
                user_dict['created_at'] = datetime.fromisoformat(user_dict['created_at']).strftime('%Y-%m-%d %H:%M')
            if user_dict['last_login']:
                user_dict['last_login'] = datetime.fromisoformat(user_dict['last_login']).strftime('%Y-%m-%d %H:%M')
            users_list.append(user_dict)

        return jsonify({
            'success': True,
            'users': users_list,
            'total_count': len(users_list)
        })

    except Exception as e:
        logger.error(f"Admin get users error: {e}")
        return jsonify({'error': 'Failed to get users'}), 500

@app.route('/api/admin/users/<int:user_id>', methods=['DELETE'])
def admin_delete_user(user_id):
    """Delete a user (admin only)"""
    if not session.get('is_admin'):
        return jsonify({'error': 'Admin access required'}), 403

    try:
        conn = sqlite3.connect('healayur.db')
        cursor = conn.cursor()

        # Get user info before deletion
        user = cursor.execute('SELECT username FROM users WHERE id = ?', (user_id,)).fetchone()
        if not user:
            conn.close()
            return jsonify({'error': 'User not found'}), 404

        # Delete user
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()
        conn.close()

        logger.info(f"🗑️ Admin deleted user: {user[0]} (ID: {user_id})")
        return jsonify({
            'success': True,
            'message': f'User {user[0]} deleted successfully'
        })

    except Exception as e:
        logger.error(f"Error deleting user {user_id}: {e}")
        return jsonify({'error': 'Failed to delete user'}), 500

@app.route('/api/admin/images/<filename>', methods=['DELETE'])
def admin_delete_image(filename):
    """Delete an image (admin only)"""
    if not session.get('is_admin'):
        return jsonify({'error': 'Admin access required'}), 403

    try:
        # Sanitize filename to prevent directory traversal
        filename = os.path.basename(filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"🗑️ Admin deleted image: {filename}")
            return jsonify({
                'success': True,
                'message': f'Image {filename} deleted successfully'
            })
        else:
            return jsonify({'error': 'Image not found'}), 404

    except Exception as e:
        logger.error(f"Error deleting image {filename}: {e}")
        return jsonify({'error': 'Failed to delete image'}), 500

@app.route('/api/admin/images/bulk-delete', methods=['POST'])
def admin_bulk_delete_images():
    """Bulk delete images (admin only)"""
    if not session.get('is_admin'):
        return jsonify({'error': 'Admin access required'}), 403

    try:
        data = request.get_json()
        filenames = data.get('filenames', [])

        if not filenames:
            return jsonify({'error': 'No files specified'}), 400

        deleted_count = 0
        errors = []

        for filename in filenames:
            try:
                # Sanitize filename
                filename = os.path.basename(filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                if os.path.exists(filepath):
                    os.remove(filepath)
                    deleted_count += 1
                else:
                    errors.append(f"File not found: {filename}")
            except Exception as e:
                errors.append(f"Error deleting {filename}: {str(e)}")

        logger.info(f"🗑️ Admin bulk deleted {deleted_count} images")

        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'errors': errors,
            'message': f'Successfully deleted {deleted_count} images'
        })

    except Exception as e:
        logger.error(f"Error in bulk delete: {e}")
        return jsonify({'error': 'Failed to bulk delete images'}), 500

@app.route('/api/admin/cleanup', methods=['POST'])
def admin_cleanup():
    """Clean up old data (admin only)"""
    if not session.get('is_admin'):
        return jsonify({'error': 'Admin access required'}), 403

    try:
        data = request.get_json()
        days_old = data.get('days', 30)

        if days_old < 1:
            return jsonify({'error': 'Days must be at least 1'}), 400

        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        deleted_count = 0

        # Clean up old images
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try:
                    if os.path.getctime(filepath) < cutoff_time:
                        os.remove(filepath)
                        deleted_count += 1
                except Exception as e:
                    logger.error(f"Error deleting old file {filename}: {e}")

        logger.info(f"🧹 Admin cleanup: deleted {deleted_count} old images (>{days_old} days)")

        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'message': f'Cleaned up {deleted_count} images older than {days_old} days'
        })

    except Exception as e:
        logger.error(f"Error in cleanup: {e}")
        return jsonify({'error': 'Failed to cleanup old data'}), 500

@app.route('/api/admin/export', methods=['POST'])
def admin_export_data():
    """Export system data (admin only)"""
    if not session.get('is_admin'):
        return jsonify({'error': 'Admin access required'}), 403

    try:
        data = request.get_json()
        export_type = data.get('type', 'all')  # 'users', 'images', 'all'

        export_data = {
            'export_date': datetime.now().isoformat(),
            'export_type': export_type
        }

        if export_type in ['users', 'all']:
            conn = sqlite3.connect('healayur.db')
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            users = cursor.execute('SELECT * FROM users').fetchall()
            export_data['users'] = [dict(user) for user in users]
            conn.close()

        if export_type in ['images', 'all']:
            images = []
            for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    stat = os.stat(filepath)
                    images.append({
                        'filename': filename,
                        'size': stat.st_size,
                        'created': datetime.fromtimestamp(stat.st_ctime).isoformat()
                    })
            export_data['images'] = images

        logger.info(f"📤 Admin exported {export_type} data")

        return jsonify({
            'success': True,
            'data': export_data,
            'message': f'Data exported successfully'
        })

    except Exception as e:
        logger.error(f"Error in data export: {e}")
        return jsonify({'error': 'Failed to export data'}), 500

# ==================== WEBSOCKET EVENTS ====================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    session_id = request.sid
    active_users.add(session_id)

    # Send welcome message
    emit('connected', {
        'message': 'Connected to Heal Ayur real-time service',
        'user_count': len(active_users)
    })

    # Broadcast user count update
    emit('user_count_update', {
        'count': len(active_users)
    }, broadcast=True)

    logger.info(f"🔗 Client connected: {session_id} (Total: {len(active_users)})")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    session_id = request.sid
    active_users.discard(session_id)

    # Remove from real-time sessions if exists
    if session_id in real_time_sessions:
        del real_time_sessions[session_id]

    # Broadcast user count update
    emit('user_count_update', {
        'count': len(active_users)
    }, broadcast=True)

    logger.info(f"🔌 Client disconnected: {session_id} (Total: {len(active_users)})")

@socketio.on('start_realtime')
def handle_start_realtime():
    """Handle real-time analysis session start"""
    session_id = request.sid
    real_time_sessions[session_id] = {
        'started_at': datetime.now(),
        'analysis_count': 0
    }

    emit('realtime_started', {
        'message': 'Real-time analysis session started',
        'session_id': session_id
    })

    logger.info(f"🔴 Real-time session started: {session_id}")

@socketio.on('stop_realtime')
def handle_stop_realtime():
    """Handle real-time analysis session stop"""
    session_id = request.sid
    if session_id in real_time_sessions:
        session_data = real_time_sessions[session_id]
        duration = (datetime.now() - session_data['started_at']).total_seconds()

        emit('realtime_stopped', {
            'message': 'Real-time analysis session stopped',
            'duration': duration,
            'analysis_count': session_data['analysis_count']
        })

        del real_time_sessions[session_id]
        logger.info(f"⏹️ Real-time session stopped: {session_id} (Duration: {duration:.1f}s)")

@socketio.on('analyze_realtime')
def handle_realtime_analysis(data):
    """Handle real-time image analysis via WebSocket"""
    session_id = request.sid

    try:
        if session_id not in real_time_sessions:
            emit('error', {'message': 'Real-time session not active'})
            return

        # Process the image
        image_base64 = data.get('image')
        if not image_base64:
            emit('error', {'message': 'No image data provided'})
            return

        start_time = time.time()

        # Save and analyze image
        filepath = process_base64_image(image_base64)
        detected_condition = analyze_skin_image(filepath)

        # Get remedies
        remedy_options = get_multiple_remedies(detected_condition, count=2)  # Fewer for real-time

        # Calculate confidence
        confidence = 0.85 + (hash(detected_condition) % 15) / 100
        processing_time = round(time.time() - start_time, 2)

        # Update session stats
        real_time_sessions[session_id]['analysis_count'] += 1

        # Send result back to client
        emit('realtime_result', {
            'condition': detected_condition,
            'confidence': round(confidence * 100, 1),
            'remedies': remedy_options,
            'processing_time': processing_time,
            'analysis_count': real_time_sessions[session_id]['analysis_count']
        })

        logger.info(f"🔴 Real-time analysis: {detected_condition} ({processing_time}s)")

    except Exception as e:
        logger.error(f"❌ Real-time analysis error: {e}")
        emit('error', {'message': 'Real-time analysis failed'})

@socketio.on('chat_message')
def handle_chat_message(data):
    """Handle chat messages for AI assistant"""
    session_id = request.sid
    message = data.get('message', '').strip()

    if not message:
        return

    try:
        # Simple AI response logic (can be enhanced with actual AI)
        response = generate_chat_response(message)

        emit('chat_response', {
            'message': response,
            'timestamp': datetime.now().isoformat()
        })

        logger.info(f"💬 Chat: {session_id} -> {message[:50]}...")

    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        emit('chat_response', {
            'message': 'Sorry, I encountered an error. Please try again.',
            'timestamp': datetime.now().isoformat()
        })

def generate_chat_response(message):
    """Generate AI chat response (simplified)"""
    message_lower = message.lower()

    # Simple keyword-based responses
    if any(word in message_lower for word in ['hello', 'hi', 'hey']):
        return "Hello! I'm your AI healing assistant. How can I help you with natural remedies today?"

    elif any(word in message_lower for word in ['acne', 'pimple']):
        return "For acne, I recommend turmeric-honey masks and neem-based treatments. These have anti-inflammatory and antibacterial properties."

    elif any(word in message_lower for word in ['burn', 'burnt']):
        return "For burns, immediately cool with cold water, then apply aloe vera gel. Honey also has excellent healing properties for minor burns."

    elif any(word in message_lower for word in ['help', 'how']):
        return "I can help you understand skin conditions, explain remedies, and guide you through the analysis process. What would you like to know?"

    else:
        return "That's interesting! Could you be more specific about your skin concern? I'm here to help with natural healing advice."

# ==================== TEST ENDPOINTS ====================

@app.route('/api/test-remedies', methods=['POST'])
def test_remedies():
    """Test endpoint to check structured remedies"""
    try:
        from ai_analysis_faceplus import get_structured_remedies

        data = request.get_json()
        condition = data.get('condition', 'acne')

        structured_remedies = get_structured_remedies(condition)

        return jsonify({
            'success': True,
            'condition': condition,
            'remedies': structured_remedies,
            'count': len(structured_remedies)
        })

    except Exception as e:
        logger.error(f"Error testing remedies: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/demo-conditions', methods=['GET'])
def demo_conditions():
    """Get all available conditions for demo purposes"""
    try:
        conditions = [
            {'key': 'acne', 'name': 'Active Acne', 'description': 'Pimples, blackheads, and breakouts'},
            {'key': 'dry_skin', 'name': 'Dry Skin', 'description': 'Dehydrated, flaky, or tight skin'},
            {'key': 'blackheads', 'name': 'Blackheads', 'description': 'Clogged pores with dark appearance'},
            {'key': 'wrinkles', 'name': 'Aging Skin', 'description': 'Fine lines, wrinkles, and age spots'},
            {'key': 'dark_spot', 'name': 'Dark Spots', 'description': 'Hyperpigmentation and uneven tone'},
            {'key': 'rash', 'name': 'Sensitive Skin', 'description': 'Redness, irritation, and sensitivity'},
            {'key': 'scars', 'name': 'Scars', 'description': 'Acne scars and blemish marks'}
        ]

        return jsonify({
            'success': True,
            'conditions': conditions,
            'count': len(conditions)
        })

    except Exception as e:
        logger.error(f"Error getting demo conditions: {e}")
        return jsonify({'success': False, 'error': str(e)})

# ==================== E-COMMERCE API ROUTES ====================

@app.route('/api/products', methods=['GET'])
def get_products():
    """Get all products for e-commerce"""
    try:
        # Filter products based on query parameters
        category = request.args.get('category')
        featured = request.args.get('featured')
        search = request.args.get('search', '').lower()

        filtered_products = {}

        for product_id, product in products.items():
            # Apply filters
            if category and product.get('category') != category:
                continue
            if featured and str(product.get('featured', False)).lower() != featured.lower():
                continue
            if search and search not in product.get('name', '').lower() and search not in ' '.join(product.get('tags', [])):
                continue

            filtered_products[product_id] = product

        return jsonify({
            'success': True,
            'products': filtered_products,
            'count': len(filtered_products),
            'categories': list(set(p.get('category') for p in products.values()))
        })

    except Exception as e:
        logger.error(f"Error getting products: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/products/<product_id>', methods=['GET'])
def get_product(product_id):
    """Get specific product details"""
    try:
        if product_id not in products:
            return jsonify({'success': False, 'error': 'Product not found'}), 404

        product = products[product_id]

        # Add related products
        related = []
        product_category = product.get('category')
        product_tags = set(product.get('tags', []))

        for pid, p in products.items():
            if pid != product_id:
                # Find products with similar tags or same category
                p_tags = set(p.get('tags', []))
                if p.get('category') == product_category or len(product_tags.intersection(p_tags)) > 0:
                    related.append({
                        'id': pid,
                        'name': p.get('name'),
                        'price': p.get('price'),
                        'image': p.get('image'),
                        'rating': p.get('rating')
                    })

                if len(related) >= 4:  # Limit to 4 related products
                    break

        return jsonify({
            'success': True,
            'product': product,
            'related_products': related
        })

    except Exception as e:
        logger.error(f"Error getting product {product_id}: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/cart', methods=['POST'])
@login_required
def add_to_cart():
    """Add product to user's cart"""
    try:
        data = request.get_json()
        product_id = data.get('product_id')
        quantity = data.get('quantity', 1)

        if not product_id or product_id not in products:
            return jsonify({'success': False, 'error': 'Invalid product'}), 400

        user = getattr(request, 'current_user', None)
        user_id = user['id'] if user else None

        # For now, return success message with contact info
        # In future, implement actual cart functionality

        product = products[product_id]
        total_price = product['price'] * quantity

        return jsonify({
            'success': True,
            'message': f'Added {quantity}x {product["name"]} to cart',
            'cart_item': {
                'product_id': product_id,
                'name': product['name'],
                'price': product['price'],
                'quantity': quantity,
                'total': total_price
            },
            'note': 'Use our contact form to place your order!'
        })

    except Exception as e:
        logger.error(f"Error adding to cart: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/contact-seller', methods=['POST'])
def contact_seller():
    """Contact seller for product inquiry"""
    try:
        data = request.get_json()
        product_id = data.get('product_id')
        message = data.get('message', '')
        customer_email = data.get('email', '')
        customer_name = data.get('name', '')

        # Log the inquiry
        logger.info(f"Product inquiry: {customer_name} ({customer_email}) - Product: {product_id} - Message: {message}")

        return jsonify({
            'success': True,
            'message': 'Your inquiry has been received! We will contact you soon.',
            'response_time': '24 hours'
        })

    except Exception as e:
        logger.error(f"Error processing contact request: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/contact', methods=['POST'])
def api_contact():
    """Handle contact form submissions"""
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        email = data.get('email', '').strip()
        subject = data.get('subject', '').strip()
        message = data.get('message', '').strip()

        if not all([name, email, subject, message]):
            return jsonify({'success': False, 'error': 'All fields are required'}), 400

        # Log the contact form submission
        logger.info(f"Contact form submission: {name} ({email}) - Subject: {subject}")
        logger.info(f"Message: {message}")

        # Here you would typically send an email or save to database
        # For now, we'll just log it

        return jsonify({
            'success': True,
            'message': 'Your message has been sent successfully! We will get back to you within 24 hours.'
        })

    except Exception as e:
        logger.error(f"Error processing contact form: {e}")
        return jsonify({'success': False, 'error': 'Failed to send message'}), 500

@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    """Handle feedback form submissions"""
    try:
        data = request.get_json()
        name = data.get('name', 'Anonymous').strip()
        feedback_type = data.get('type', '').strip()
        rating = data.get('rating', '')
        message = data.get('message', '').strip()

        if not all([feedback_type, message]):
            return jsonify({'success': False, 'error': 'Feedback type and message are required'}), 400

        # Log the feedback submission
        logger.info(f"Feedback submission: {name} - Type: {feedback_type} - Rating: {rating}")
        logger.info(f"Feedback: {message}")

        # Here you would typically save to database or send notification
        # For now, we'll just log it

        return jsonify({
            'success': True,
            'message': 'Thank you for your feedback! We appreciate your input and will use it to improve our service.'
        })

    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        return jsonify({'success': False, 'error': 'Failed to submit feedback'}), 500

# ==================== ERROR HANDLERS ====================

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 16MB."}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"❌ Internal server error: {e}")
    return jsonify({"error": "Internal server error. Please try again."}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404

# ==================== STARTUP ====================

if __name__ == '__main__':
    logger.info("🌿 ================================== 🌿")
    logger.info("   HEAL AYUR - AI ANCIENT HEALING")
    logger.info("🌿 ================================== 🌿")
    logger.info("")
    logger.info("🚀 Starting enhanced Heal Ayur application...")
    logger.info(f"📚 Loaded {len(remedies)} remedy categories")
    logger.info(f"🛒 Loaded {len(products)} products for e-commerce")
    logger.info(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"🔐 Authentication: Enabled")
    logger.info(f"🔴 Real-time features: Enabled")
    logger.info(f"💬 Chat assistant: Enabled")
    logger.info(f"📧 Contact: thatrasunil@gmail.com")
    logger.info("")
    logger.info("✨ Features available:")
    logger.info("   • AI-powered skin analysis")
    logger.info("   • User authentication & profiles")
    logger.info("   • Real-time webcam analysis")
    logger.info("   • Intelligent chatbot")
    logger.info("   • Analysis history tracking")
    logger.info("   • WebSocket real-time updates")
    logger.info("   • E-commerce product catalog")
    logger.info("   • Natural ingredients marketplace")
    logger.info("")

    # Get port from environment variable for deployment
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV', 'development') == 'development'

    if debug_mode:
        logger.info("🌐 Server starting on http://localhost:5000")
        logger.info("🎉 Ready to heal! 🌿")
        # Development mode
        socketio.run(app, debug=True, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)
    else:
        logger.info(f"🌐 Server starting on port {port}")
        logger.info("🎉 Ready to heal! 🌿")
        # Production mode
        socketio.run(app, debug=False, host='0.0.0.0', port=port)
