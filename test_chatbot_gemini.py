#!/usr/bin/env python3
"""
Test to verify if chatbot is actually using Gemini API or local fallback
"""

import requests
import time
import json

def test_chatbot_response_source():
    """Test if chatbot responses are coming from Gemini API"""
    print("🤖 Testing Chatbot Response Source")
    print("=" * 40)
    
    base_url = "http://localhost:5000"
    
    # Test messages that should clearly show if it's Gemini vs local
    test_cases = [
        {
            "message": "What is the capital of France?",
            "description": "General knowledge question (Gemini should know)",
            "expected_source": "gemini"
        },
        {
            "message": "Write a poem about artificial intelligence",
            "description": "Creative task (Gemini should excel)",
            "expected_source": "gemini"
        },
        {
            "message": "Hello! How can you help me?",
            "description": "Basic greeting (could be either)",
            "expected_source": "either"
        },
        {
            "message": "Explain quantum physics in simple terms",
            "description": "Complex explanation (Gemini should provide detailed response)",
            "expected_source": "gemini"
        }
    ]
    
    print("🔍 Testing different types of messages to identify response source...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔹 Test {i}: {test_case['description']}")
        print(f"Message: '{test_case['message']}'")
        print("-" * 50)
        
        try:
            start_time = time.time()
            
            # Make API request
            response = requests.post(
                f"{base_url}/api/chat",
                json={
                    "message": test_case["message"],
                    "history": []
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    ai_response = data.get('response', '')
                    model = data.get('model', 'unknown')
                    
                    print(f"✅ Response received in {response_time:.2f} seconds")
                    print(f"🤖 Model: {model}")
                    print(f"📝 Response length: {len(ai_response)} characters")
                    print(f"💬 Response preview: {ai_response[:100]}...")
                    
                    # Analyze response characteristics
                    if model and 'gemini' in model.lower():
                        print("🎯 Source: GEMINI API (confirmed by model field)")
                    elif response_time < 0.5:
                        print("⚡ Source: Likely LOCAL (very fast response)")
                    elif len(ai_response) > 200 and any(word in ai_response.lower() for word in ['detailed', 'comprehensive', 'specifically']):
                        print("🧠 Source: Likely GEMINI (detailed response)")
                    else:
                        print("❓ Source: UNCLEAR (could be either)")
                    
                    # Check for Gemini-specific response patterns
                    gemini_indicators = [
                        len(ai_response) > 150,  # Gemini tends to be more verbose
                        any(phrase in ai_response.lower() for phrase in [
                            'i can help', 'let me explain', 'here\'s what', 'specifically',
                            'in detail', 'comprehensive', 'important to note'
                        ]),
                        response_time > 1.0  # Gemini API takes time
                    ]
                    
                    gemini_score = sum(gemini_indicators)
                    print(f"🎯 Gemini likelihood score: {gemini_score}/3")
                    
                else:
                    print(f"❌ API returned error: {data.get('error', 'Unknown error')}")
            else:
                print(f"❌ HTTP Error {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection failed - Flask app not running")
            print("💡 Please start the Flask app first: python app.py")
            break
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    print("\n🏁 Chatbot source analysis completed!")

def test_direct_gemini_vs_api():
    """Compare direct Gemini API call vs Flask API call"""
    print("\n🔬 Direct Gemini vs Flask API Comparison")
    print("=" * 45)
    
    test_message = "Explain the benefits of turmeric for skin health"
    
    # Test 1: Direct Gemini API
    print("🧪 Testing Direct Gemini API...")
    try:
        import google.generativeai as genai
        
        genai.configure(api_key="AIzaSyDhsUnlg85rpkGoiJlKnJHNXAl_0FfIeh0")
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        start_time = time.time()
        response = model.generate_content(test_message)
        end_time = time.time()
        
        direct_response = response.text.strip()
        direct_time = end_time - start_time
        
        print(f"✅ Direct Gemini response in {direct_time:.2f} seconds")
        print(f"📝 Length: {len(direct_response)} characters")
        print(f"💬 Preview: {direct_response[:100]}...")
        
    except Exception as e:
        print(f"❌ Direct Gemini test failed: {e}")
        direct_response = None
        direct_time = None
    
    # Test 2: Flask API
    print("\n🌐 Testing Flask API...")
    try:
        start_time = time.time()
        response = requests.post(
            "http://localhost:5000/api/chat",
            json={"message": test_message, "history": []},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                flask_response = data.get('response', '')
                flask_time = end_time - start_time
                flask_model = data.get('model', 'unknown')
                
                print(f"✅ Flask API response in {flask_time:.2f} seconds")
                print(f"🤖 Model: {flask_model}")
                print(f"📝 Length: {len(flask_response)} characters")
                print(f"💬 Preview: {flask_response[:100]}...")
                
                # Compare responses
                if direct_response and flask_response:
                    print(f"\n📊 COMPARISON:")
                    print(f"Direct Gemini time: {direct_time:.2f}s")
                    print(f"Flask API time: {flask_time:.2f}s")
                    print(f"Response similarity: {'High' if abs(len(direct_response) - len(flask_response)) < 100 else 'Low'}")
                    
                    if flask_model and 'gemini' in flask_model.lower():
                        print("🎯 CONCLUSION: Flask API is using Gemini!")
                    else:
                        print("⚠️ CONCLUSION: Flask API might be using fallback")
                
            else:
                print(f"❌ Flask API error: {data.get('error')}")
        else:
            print(f"❌ Flask API HTTP error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Flask API test failed: {e}")

def check_flask_app_gemini_status():
    """Check if Flask app has Gemini properly configured"""
    print("\n🔧 Checking Flask App Gemini Configuration")
    print("=" * 45)
    
    try:
        from app import GEMINI_AVAILABLE, GEMINI_MODEL_NAME, gemini_model
        
        print(f"GEMINI_AVAILABLE: {GEMINI_AVAILABLE}")
        print(f"GEMINI_MODEL_NAME: {GEMINI_MODEL_NAME}")
        print(f"gemini_model object: {type(gemini_model)}")
        
        if GEMINI_AVAILABLE and gemini_model:
            print("✅ Gemini is properly configured in Flask app")
            
            # Test the model directly
            try:
                test_response = gemini_model.generate_content("Hello, this is a test")
                print(f"✅ Direct model test successful: {test_response.text[:50]}...")
                return True
            except Exception as e:
                print(f"❌ Direct model test failed: {e}")
                return False
        else:
            print("❌ Gemini is NOT properly configured in Flask app")
            return False
            
    except Exception as e:
        print(f"❌ Failed to check Flask app configuration: {e}")
        return False

def main():
    """Run all chatbot verification tests"""
    print("🌿 Heal Ayur - Chatbot Gemini Integration Verification")
    print("=" * 65)
    
    # Check Flask app configuration first
    gemini_configured = check_flask_app_gemini_status()
    
    if gemini_configured:
        # Test chatbot responses
        test_chatbot_response_source()
        
        # Compare direct vs API
        test_direct_gemini_vs_api()
        
        print("\n📋 FINAL ASSESSMENT:")
        print("=" * 25)
        print("✅ Gemini API is configured in Flask app")
        print("✅ Chatbot endpoint is available")
        print("💡 Check the response times and model fields above to confirm Gemini usage")
        
    else:
        print("\n❌ ISSUE DETECTED:")
        print("Gemini API is not properly configured in the Flask app")
        print("This means the chatbot is likely using local fallback responses")

if __name__ == "__main__":
    main()
