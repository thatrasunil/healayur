#!/usr/bin/env python3
"""
Heal Ayur - Authentication System
Advanced user authentication with session management and security features
"""

import hashlib
import secrets
import sqlite3
import json
from datetime import datetime, timedelta
from functools import wraps
from flask import session, request, jsonify, redirect, url_for

class AuthManager:
    def __init__(self, db_path='healayur.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the user database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT,
                salt TEXT,
                full_name TEXT,
                profile_picture TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                email_verified BOOLEAN DEFAULT 0,
                preferences TEXT DEFAULT '{}',
                total_analyses INTEGER DEFAULT 0,
                oauth_provider TEXT,
                oauth_provider_id TEXT,
                avatar_url TEXT
            )
        ''')
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_token TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Analysis history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                condition_detected TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                processing_time REAL NOT NULL,
                image_path TEXT,
                remedies_suggested TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                feedback_rating INTEGER,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Chat history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                message TEXT NOT NULL,
                sender TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully")
    
    def hash_password(self, password):
        """Hash password with salt"""
        salt = secrets.token_hex(32)
        password_hash = hashlib.pbkdf2_hmac('sha256', 
                                          password.encode('utf-8'), 
                                          salt.encode('utf-8'), 
                                          100000)
        return password_hash.hex(), salt
    
    def verify_password(self, password, password_hash, salt):
        """Verify password against hash"""
        return hashlib.pbkdf2_hmac('sha256', 
                                 password.encode('utf-8'), 
                                 salt.encode('utf-8'), 
                                 100000).hex() == password_hash
    
    def register_user(self, username, email, password, full_name=None):
        """Register a new user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if user already exists
            cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', 
                         (username, email))
            if cursor.fetchone():
                return {'success': False, 'error': 'Username or email already exists'}
            
            # Hash password
            password_hash, salt = self.hash_password(password)
            
            # Insert user
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, salt, full_name)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, email, password_hash, salt, full_name))
            
            user_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return {
                'success': True, 
                'user_id': user_id,
                'message': 'User registered successfully'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def login_user(self, username_or_email, password, ip_address=None, user_agent=None):
        """Authenticate user and create session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find user
            cursor.execute('''
                SELECT id, username, email, password_hash, salt, full_name, is_active
                FROM users 
                WHERE (username = ? OR email = ?) AND is_active = 1
            ''', (username_or_email, username_or_email))
            
            user = cursor.fetchone()
            if not user:
                return {'success': False, 'error': 'Invalid credentials'}
            
            user_id, username, email, password_hash, salt, full_name, is_active = user
            
            # Verify password
            if not self.verify_password(password, password_hash, salt):
                return {'success': False, 'error': 'Invalid credentials'}
            
            # Create session token
            session_token = secrets.token_urlsafe(32)
            expires_at = datetime.now() + timedelta(days=30)  # 30 days
            
            # Store session
            cursor.execute('''
                INSERT INTO user_sessions 
                (user_id, session_token, expires_at, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, session_token, expires_at, ip_address, user_agent))
            
            # Update last login
            cursor.execute('''
                UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
            ''', (user_id,))
            
            conn.commit()
            conn.close()
            
            return {
                'success': True,
                'session_token': session_token,
                'user': {
                    'id': user_id,
                    'username': username,
                    'email': email,
                    'full_name': full_name
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # OAuth functionality removed - using only regular authentication

    def verify_session(self, session_token):
        """Verify session token and return user info"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT u.id, u.username, u.email, u.full_name, s.expires_at
                FROM users u
                JOIN user_sessions s ON u.id = s.user_id
                WHERE s.session_token = ? AND s.is_active = 1 AND s.expires_at > ?
            ''', (session_token, datetime.now()))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                user_id, username, email, full_name, expires_at = result
                return {
                    'success': True,
                    'user': {
                        'id': user_id,
                        'username': username,
                        'email': email,
                        'full_name': full_name
                    }
                }
            else:
                return {'success': False, 'error': 'Invalid or expired session'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def logout_user(self, session_token):
        """Logout user by invalidating session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE user_sessions SET is_active = 0 WHERE session_token = ?
            ''', (session_token,))
            
            conn.commit()
            conn.close()
            
            return {'success': True, 'message': 'Logged out successfully'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def save_analysis(self, user_id, condition, confidence, processing_time, remedies):
        """Save analysis result to user history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO analysis_history 
                (user_id, condition_detected, confidence_score, processing_time, remedies_suggested)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, condition, confidence, processing_time, json.dumps(remedies)))
            
            # Update user's total analyses count
            cursor.execute('''
                UPDATE users SET total_analyses = total_analyses + 1 WHERE id = ?
            ''', (user_id,))
            
            conn.commit()
            conn.close()
            
            return {'success': True}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_user_history(self, user_id, limit=50):
        """Get user's analysis history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT condition_detected, confidence_score, processing_time, 
                       remedies_suggested, created_at, feedback_rating
                FROM analysis_history 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (user_id, limit))
            
            history = []
            for row in cursor.fetchall():
                condition, confidence, time, remedies, created_at, rating = row
                history.append({
                    'condition': condition,
                    'confidence': confidence,
                    'processing_time': time,
                    'remedies': json.loads(remedies) if remedies else [],
                    'created_at': created_at,
                    'rating': rating
                })
            
            conn.close()
            return {'success': True, 'history': history}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

# Authentication decorators
def login_required(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        session_token = session.get('session_token') or request.headers.get('Authorization')
        
        if not session_token:
            if request.is_json:
                return jsonify({'error': 'Authentication required'}), 401
            return redirect(url_for('login'))
        
        auth_manager = AuthManager()
        result = auth_manager.verify_session(session_token)
        
        if not result['success']:
            if request.is_json:
                return jsonify({'error': 'Invalid session'}), 401
            return redirect(url_for('login'))
        
        # Add user info to request context
        request.current_user = result['user']
        return f(*args, **kwargs)
    
    return decorated_function

def optional_auth(f):
    """Decorator for optional authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        session_token = session.get('session_token') or request.headers.get('Authorization')
        
        if session_token:
            auth_manager = AuthManager()
            result = auth_manager.verify_session(session_token)
            if result['success']:
                request.current_user = result['user']
            else:
                request.current_user = None
        else:
            request.current_user = None
        
        return f(*args, **kwargs)
    
    return decorated_function
