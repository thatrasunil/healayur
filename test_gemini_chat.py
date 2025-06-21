#!/usr/bin/env python3
"""
Test script for Gemini API integration in Heal Ayur chatbot
"""

import google.generativeai as genai
import json

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyDhsUnlg85rpkGoiJlKnJHNXAl_0FfIeh0"
genai.configure(api_key=GEMINI_API_KEY)

def test_gemini_chat(model_name='gemini-1.5-flash'):
    """Test the Gemini API integration"""
    try:
        # Initialize the model
        model = genai.GenerativeModel(model_name)
        
        # Test system prompt
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

Current user: Test User"""

        # Test messages
        test_messages = [
            "Hello! How can you help me?",
            "I have acne on my face. What natural remedies do you recommend?",
            "How does the real-time analysis feature work?",
            "What ingredients are good for dry skin?"
        ]
        
        print("🧪 Testing Gemini API Integration for Heal Ayur Chatbot")
        print("=" * 60)
        
        for i, user_message in enumerate(test_messages, 1):
            print(f"\n🔹 Test {i}: {user_message}")
            print("-" * 40)
            
            # Build conversation context
            conversation_context = system_prompt + f"\n\nUser: {user_message}\nAssistant:"
            
            # Generate response
            response = model.generate_content(conversation_context)
            ai_response = response.text.strip()
            
            print(f"🤖 AI Response: {ai_response}")
            print()
        
        print("✅ All tests completed successfully!")
        print("🌿 Gemini API is working correctly for Heal Ayur chatbot")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Gemini API: {e}")
        return False

def test_api_key():
    """Test if the API key is valid"""
    try:
        # Try different model names
        model_names = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']

        for model_name in model_names:
            try:
                print(f"🔍 Trying model: {model_name}")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("Hello, this is a test message.")
                print(f"✅ API Key is valid and working with model: {model_name}")
                return True, model_name
            except Exception as e:
                print(f"❌ Model {model_name} failed: {e}")
                continue

        print("❌ All models failed")
        return False, None
    except Exception as e:
        print(f"❌ API Key test failed: {e}")
        return False, None

if __name__ == "__main__":
    print("🔑 Testing API Key...")
    success, working_model = test_api_key()
    if success:
        print(f"\n🧪 Running full chatbot tests with {working_model}...")
        test_gemini_chat(working_model)
    else:
        print("❌ Cannot proceed with tests due to API key issues")
