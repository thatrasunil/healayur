#!/usr/bin/env python3
"""
Test script for the chat API endpoint
"""

import requests
import json

def test_chat_api():
    """Test the chat API endpoint"""
    
    # Start the Flask app in the background
    print("🧪 Testing Chat API Endpoint")
    print("=" * 40)
    
    # Test data
    test_messages = [
        {
            "message": "Hello! How can you help me?",
            "expected_keywords": ["hello", "help", "assistant"]
        },
        {
            "message": "I have acne on my face. What should I do?",
            "expected_keywords": ["acne", "natural", "remedies"]
        },
        {
            "message": "How does the real-time analysis work?",
            "expected_keywords": ["real-time", "analysis", "webcam"]
        }
    ]
    
    base_url = "http://localhost:5000"
    
    print("📡 Testing chat API endpoint...")
    
    for i, test_case in enumerate(test_messages, 1):
        print(f"\n🔹 Test {i}: {test_case['message']}")
        print("-" * 30)
        
        try:
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
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    print(f"✅ Success! Model: {data.get('model', 'unknown')}")
                    print(f"🤖 Response: {data.get('response', '')[:100]}...")
                    
                    # Check for expected keywords
                    response_text = data.get('response', '').lower()
                    found_keywords = [kw for kw in test_case['expected_keywords'] if kw in response_text]
                    if found_keywords:
                        print(f"🎯 Found keywords: {found_keywords}")
                    else:
                        print("⚠️ No expected keywords found")
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
    
    print("\n🏁 Chat API tests completed!")

if __name__ == "__main__":
    test_chat_api()
