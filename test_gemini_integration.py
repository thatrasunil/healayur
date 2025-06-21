#!/usr/bin/env python3
"""
Complete test script for Gemini integration in Heal Ayur
"""

import time
import subprocess
import sys
import requests
import json
from threading import Thread

def test_direct_gemini():
    """Test Gemini API directly"""
    print("🔑 Testing Gemini API directly...")
    try:
        import google.generativeai as genai
        
        # Configure API
        genai.configure(api_key="AIzaSyDhsUnlg85rpkGoiJlKnJHNXAl_0FfIeh0")
        
        # Test model
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Hello! This is a test for Heal Ayur chatbot.")
        
        print("✅ Direct Gemini API test successful!")
        print(f"🤖 Response: {response.text[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Direct Gemini API test failed: {e}")
        return False

def test_flask_import():
    """Test Flask app import"""
    print("\n📦 Testing Flask app import...")
    try:
        from app import app, GEMINI_AVAILABLE, GEMINI_MODEL_NAME
        print("✅ Flask app imported successfully!")
        print(f"🤖 Gemini Available: {GEMINI_AVAILABLE}")
        if GEMINI_AVAILABLE:
            print(f"🎯 Model: {GEMINI_MODEL_NAME}")
        return True, app
    except Exception as e:
        print(f"❌ Flask app import failed: {e}")
        return False, None

def test_chat_endpoint(app):
    """Test chat endpoint directly"""
    print("\n🧪 Testing chat endpoint...")
    try:
        with app.test_client() as client:
            # Test chat endpoint
            response = client.post('/api/chat', 
                json={
                    'message': 'Hello! How can you help me with skin care?',
                    'history': []
                },
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.get_json()
                if data.get('success'):
                    print("✅ Chat endpoint test successful!")
                    print(f"🤖 Model: {data.get('model', 'unknown')}")
                    print(f"💬 Response: {data.get('response', '')[:100]}...")
                    return True
                else:
                    print(f"❌ Chat endpoint returned error: {data.get('error')}")
                    return False
            else:
                print(f"❌ Chat endpoint HTTP error: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Chat endpoint test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🌿 Heal Ayur - Gemini Integration Test Suite")
    print("=" * 50)
    
    # Test 1: Direct Gemini API
    if not test_direct_gemini():
        print("❌ Cannot proceed - Gemini API not working")
        return False
    
    # Test 2: Flask app import
    success, app = test_flask_import()
    if not success:
        print("❌ Cannot proceed - Flask app import failed")
        return False
    
    # Test 3: Chat endpoint
    if not test_chat_endpoint(app):
        print("❌ Chat endpoint test failed")
        return False
    
    print("\n🎉 All tests passed! Gemini integration is working correctly!")
    print("\n📋 Summary:")
    print("✅ Gemini API key is valid and working")
    print("✅ Flask app imports successfully with Gemini support")
    print("✅ Chat endpoint responds correctly")
    print("✅ Ready for deployment!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
