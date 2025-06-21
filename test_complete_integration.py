#!/usr/bin/env python3
"""
Complete integration test for Face++ and Gemini in Heal Ayur
"""

import time
import sys
from PIL import Image
import io

def test_face_plus_api_credentials():
    """Test Face++ API credentials"""
    print("🔑 Testing Face++ API Credentials...")
    try:
        import requests
        
        API_KEY = "8f0FzOLKgxrCCai5NzAkiGp_D-5XRd-f"
        API_SECRET = "8YbEz4LCUZ2-Furl5Oohw0RonwgV18CX"
        API_URL = "https://api-us.faceplusplus.com/facepp/v3/detect"
        
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color=(200, 150, 120))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        image_data = buf.getvalue()
        
        # Test API credentials
        files = {'image_file': ('test.jpg', image_data, 'image/jpeg')}
        data = {
            'api_key': API_KEY,
            'api_secret': API_SECRET,
            'return_attributes': 'gender,age'
        }
        
        response = requests.post(API_URL, data=data, files=files, timeout=10)
        
        if response.status_code == 200:
            print("✅ Face++ API credentials are valid")
            return True
        else:
            print(f"❌ Face++ API error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Face++ API test failed: {e}")
        return False

def test_gemini_api():
    """Test Gemini API"""
    print("\n🧠 Testing Gemini API...")
    try:
        import google.generativeai as genai
        
        genai.configure(api_key="AIzaSyDhsUnlg85rpkGoiJlKnJHNXAl_0FfIeh0")
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Hello, this is a test.")
        
        print("✅ Gemini API is working")
        return True
        
    except Exception as e:
        print(f"❌ Gemini API test failed: {e}")
        return False

def test_hybrid_analysis_module():
    """Test the hybrid Face++ & Gemini analysis module"""
    print("\n🔬 Testing Hybrid Analysis Module...")
    try:
        from ai_analysis_faceplus import analyze_image
        
        # Create test image
        img = Image.new('RGB', (300, 400), color=(220, 180, 140))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        image_data = buf.getvalue()
        
        # Run analysis
        result = analyze_image(image_data)
        
        # Check result structure
        required_fields = ['condition', 'confidence', 'remedies']
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")
        
        print("✅ Hybrid analysis module working correctly")
        print(f"   Condition: {result.get('condition', 'Unknown')}")
        print(f"   Confidence: {result.get('confidence', 0):.1%}")
        print(f"   Face Detected: {result.get('face_detected', False)}")
        
        if 'gender_analysis' in result:
            gender = result['gender_analysis']
            print(f"   Gender: {gender.get('detected_gender', 'Unknown')}")
        
        if 'age_analysis' in result:
            age = result['age_analysis']
            print(f"   Age: {age.get('estimated_age', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hybrid analysis module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_flask_app_integration():
    """Test Flask app integration"""
    print("\n🌐 Testing Flask App Integration...")
    try:
        from app import app, AI_AVAILABLE
        
        if not AI_AVAILABLE:
            raise ValueError("AI analysis not available in Flask app")
        
        # Test with app context
        with app.test_client() as client:
            # Test main page
            response = client.get('/')
            if response.status_code != 200:
                raise ValueError(f"Main page failed: {response.status_code}")
            
            print("✅ Flask app integration working")
            print(f"   AI Available: {AI_AVAILABLE}")
            return True
            
    except Exception as e:
        print(f"❌ Flask app integration test failed: {e}")
        return False

def test_analysis_endpoint():
    """Test the analysis endpoint"""
    print("\n📊 Testing Analysis Endpoint...")
    try:
        from app import app
        
        # Create test image
        img = Image.new('RGB', (300, 400), color=(220, 180, 140))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        image_data = buf.getvalue()
        
        with app.test_client() as client:
            # Test file upload analysis
            response = client.post('/analyze', 
                data={'image': (io.BytesIO(image_data), 'test.jpg')},
                content_type='multipart/form-data'
            )
            
            if response.status_code == 200:
                result = response.get_json()
                if result.get('success'):
                    print("✅ Analysis endpoint working correctly")
                    analysis_result = result.get('result', {})
                    print(f"   Condition: {analysis_result.get('condition', 'Unknown')}")
                    print(f"   Confidence: {analysis_result.get('confidence', 0)}%")
                    return True
                else:
                    raise ValueError(f"Analysis failed: {result.get('error', 'Unknown error')}")
            else:
                raise ValueError(f"Analysis endpoint failed: {response.status_code}")
                
    except Exception as e:
        print(f"❌ Analysis endpoint test failed: {e}")
        return False

def test_chat_endpoint():
    """Test the chat endpoint with Gemini"""
    print("\n💬 Testing Chat Endpoint...")
    try:
        from app import app
        
        with app.test_client() as client:
            response = client.post('/api/chat',
                json={
                    'message': 'Hello! How can you help me with skin care?',
                    'history': []
                },
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.get_json()
                if result.get('success'):
                    print("✅ Chat endpoint working correctly")
                    print(f"   Model: {result.get('model', 'Unknown')}")
                    print(f"   Response: {result.get('response', '')[:50]}...")
                    return True
                else:
                    raise ValueError(f"Chat failed: {result.get('error', 'Unknown error')}")
            else:
                raise ValueError(f"Chat endpoint failed: {response.status_code}")
                
    except Exception as e:
        print(f"❌ Chat endpoint test failed: {e}")
        return False

def main():
    """Run complete integration test suite"""
    print("🌿 Heal Ayur - Complete Integration Test Suite")
    print("=" * 60)
    print("Testing Face++ API Key: 8f0FzOLKgxrCCai5NzAkiGp_D-5XRd-f")
    print("Testing Face++ API Secret: 8YbEz4LCUZ2-Furl5Oohw0RonwgV18CX")
    print("Testing Gemini API Key: AIzaSyDhsUnlg85rpkGoiJlKnJHNXAl_0FfIeh0")
    print("=" * 60)
    
    tests = [
        ("Face++ API Credentials", test_face_plus_api_credentials),
        ("Gemini API", test_gemini_api),
        ("Hybrid Analysis Module", test_hybrid_analysis_module),
        ("Flask App Integration", test_flask_app_integration),
        ("Analysis Endpoint", test_analysis_endpoint),
        ("Chat Endpoint", test_chat_endpoint)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📋 COMPLETE TEST SUMMARY:")
    print("=" * 40)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Face++ and Gemini integration is working perfectly!")
        print("\n🚀 Ready for deployment with:")
        print("   ✅ Face++ facial analysis and demographics")
        print("   ✅ Gemini AI skin condition analysis")
        print("   ✅ Intelligent chatbot with Gemini")
        print("   ✅ Enhanced fallback systems")
        print("   ✅ Complete Flask integration")
        return True
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
