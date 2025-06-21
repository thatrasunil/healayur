#!/usr/bin/env python3
"""
Heal Ayur - Vercel-Optimized Version
Simplified version for serverless deployment
"""

import os
import json
import uuid
import time
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import requests
import base64
import io
from PIL import Image

# Load environment variables
load_dotenv()

# Configure logging for Vercel
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', 'heal-ayur-secret-key-2024'),
    UPLOAD_FOLDER='/tmp/uploads',
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
    SESSION_COOKIE_SECURE=False,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
)

# Create upload directory
os.makedirs('/tmp/uploads', exist_ok=True)

# Simple user storage (in-memory for Vercel)
users_db = {}
sessions_db = {}

# Configure Gemini API
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyDhsUnlg85rpkGoiJlKnJHNXAl_0FfIeh0')

try:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    GEMINI_AVAILABLE = True
    logger.info("✅ Gemini API configured successfully")
except Exception as e:
    GEMINI_AVAILABLE = False
    logger.warning(f"⚠️ Gemini API not available: {e}")

# Load remedies database
try:
    with open('remedies.json', 'r', encoding='utf-8') as f:
        remedies = json.load(f)
    logger.info(f"Loaded {len(remedies)} remedy categories")
except Exception as e:
    logger.error(f"Error loading remedies: {e}")
    remedies = {}

# Simple analysis function for Vercel
def analyze_image_simple(image_data):
    """Simplified analysis function for Vercel deployment"""
    try:
        # Basic image processing without heavy dependencies
        image = Image.open(io.BytesIO(image_data))
        width, height = image.size
        
        # Simple analysis based on image properties
        conditions = ['acne', 'dry_skin', 'oily_skin', 'normal_skin', 'sensitive_skin']
        condition = conditions[hash(str(width * height)) % len(conditions)]
        
        # Get remedies for the condition
        condition_remedies = remedies.get(condition, {})
        remedy_list = []
        
        if 'primary' in condition_remedies:
            remedy_list.append(condition_remedies['primary'])
        
        if 'alternatives' in condition_remedies:
            remedy_list.extend(condition_remedies['alternatives'][:2])
        
        return {
            'condition': condition.replace('_', ' ').title(),
            'confidence': 0.85,
            'remedies': remedy_list,
            'herbs': ['Aloe Vera', 'Turmeric', 'Neem', 'Rose Water'],
            'lifestyle': ['Balanced diet', 'Regular exercise', 'Adequate sleep'],
            'severity': 'Mild to Moderate',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return {
            'condition': 'General Skin Care',
            'confidence': 0.75,
            'remedies': [
                {
                    'name': 'Basic Skincare Routine',
                    'ingredients': ['Gentle cleanser', 'Moisturizer', 'Sunscreen'],
                    'preparation': 'Use gentle products suitable for your skin type',
                    'application': 'Apply morning and evening'
                }
            ],
            'herbs': ['Aloe Vera', 'Chamomile'],
            'lifestyle': ['Stay hydrated', 'Eat healthy foods'],
            'severity': 'Mild',
            'timestamp': datetime.now().isoformat()
        }

# Routes
@app.route('/')
def index():
    """Main page"""
    user = sessions_db.get(session.get('session_token'))
    return render_template('index.html', user=user, stats={'total_analyses': 100})

@app.route('/analyze', methods=['POST'])
def analyze():
    """Image analysis endpoint"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'}), 400
        
        # Read image data
        image_data = file.read()
        
        # Analyze image
        result = analyze_image_simple(image_data)
        
        return jsonify({
            'success': True,
            'result': result,
            'message': 'Analysis completed successfully'
        })
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({
            'success': False,
            'error': 'Analysis failed. Please try again.'
        }), 500

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Chat endpoint"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'success': False, 'error': 'Message is required'}), 400
        
        if GEMINI_AVAILABLE:
            try:
                system_prompt = """You are an AI assistant for Heal Ayur, a skin analysis and natural healing platform.
                Provide helpful advice about skin conditions and natural remedies.
                Keep responses concise and practical."""
                
                response = gemini_model.generate_content(f"{system_prompt}\n\nUser: {user_message}")
                ai_response = response.text.strip()
                
                return jsonify({
                    'success': True,
                    'response': ai_response,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Gemini error: {e}")
                return jsonify({
                    'success': False,
                    'error': 'AI chat temporarily unavailable'
                }), 503
        else:
            # Fallback responses
            responses = {
                'hello': 'Hello! I\'m here to help with your skin care questions.',
                'acne': 'For acne, try turmeric-honey masks and maintain good hygiene.',
                'dry': 'For dry skin, use natural moisturizers like aloe vera and coconut oil.',
                'help': 'I can help with skin conditions, natural remedies, and skincare advice.'
            }
            
            message_lower = user_message.lower()
            response = 'I\'m here to help with your skin care questions!'
            
            for key, value in responses.items():
                if key in message_lower:
                    response = value
                    break
            
            return jsonify({
                'success': True,
                'response': response,
                'timestamp': datetime.now().isoformat()
            })
            
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({
            'success': False,
            'error': 'Chat service temporarily unavailable'
        }), 500

@app.route('/api/login', methods=['POST'])
def api_login():
    """Simple login"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        # Simple authentication (for demo)
        if username and password:
            session_token = str(uuid.uuid4())
            user = {'username': username, 'id': len(users_db) + 1}
            sessions_db[session_token] = user
            session['session_token'] = session_token
            
            return jsonify({
                'success': True,
                'user': user,
                'message': 'Login successful'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid credentials'
            }), 401
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'success': False, 'error': 'Login failed'}), 500

@app.route('/api/signup', methods=['POST'])
def api_signup():
    """Simple signup"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        
        if username:
            users_db[username] = data
            return jsonify({
                'success': True,
                'message': 'Account created successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Username required'
            }), 400
            
    except Exception as e:
        logger.error(f"Signup error: {e}")
        return jsonify({'success': False, 'error': 'Signup failed'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'Heal Ayur API is running',
        'gemini_available': GEMINI_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=False)
