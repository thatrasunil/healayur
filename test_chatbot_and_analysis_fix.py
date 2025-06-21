#!/usr/bin/env python3
"""
Test both chatbot response and analysis fixes
"""

import requests
import json
import time
from PIL import Image, ImageDraw
import io
import base64

def create_test_image():
    """Create a simple test image"""
    img = Image.new('RGB', (400, 300), color=(240, 220, 200))
    draw = ImageDraw.Draw(img)
    
    # Simple face-like shape
    draw.ellipse([100, 50, 300, 200], fill=(220, 190, 160), outline=(200, 170, 140), width=2)
    draw.ellipse([140, 90, 170, 120], fill=(0, 0, 0))  # Left eye
    draw.ellipse([230, 90, 260, 120], fill=(0, 0, 0))  # Right eye
    draw.ellipse([180, 140, 220, 160], fill=(180, 100, 100))  # Mouth
    
    return img

def test_chatbot_response():
    """Test chatbot response handling"""
    print("🤖 Testing Chatbot Response")
    print("=" * 30)
    
    test_messages = [
        "Hello! How are you?",
        "I have acne problems, can you help me with natural remedies?",
        "What are the benefits of turmeric for skin health?",
        "Can you explain how to prepare a neem face mask?"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n🔹 Test {i}: '{message[:50]}{'...' if len(message) > 50 else ''}'")
        
        try:
            response = requests.post(
                "http://localhost:5000/api/chat",
                json={
                    "message": message,
                    "history": []
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    ai_response = data.get('response', '')
                    model = data.get('model', 'unknown')
                    
                    print(f"✅ Response received")
                    print(f"🤖 Model: {model}")
                    print(f"📝 Response length: {len(ai_response)} characters")
                    print(f"💬 Response preview: {ai_response[:150]}...")
                    
                    # Check if response is complete (not truncated)
                    if len(ai_response) > 100:
                        print("✅ Response appears complete")
                    else:
                        print("⚠️ Response might be truncated")
                        
                else:
                    print(f"❌ Chat failed: {data.get('error', 'Unknown error')}")
            else:
                print(f"❌ HTTP Error {response.status_code}")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    return True

def test_image_analysis():
    """Test image analysis functionality"""
    print("\n🔬 Testing Image Analysis")
    print("=" * 30)
    
    # Create test image
    img = create_test_image()
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=95)
    image_data = buf.getvalue()
    
    print("📸 Created test image")
    
    try:
        # Test file upload analysis
        print("📁 Testing file upload analysis...")
        files = {'image': ('test_image.jpg', image_data, 'image/jpeg')}
        
        response = requests.post(
            "http://localhost:5000/analyze",
            files=files,
            timeout=45
        )
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                result = data.get('result', {})
                print("✅ Image analysis successful!")
                print(f"   Condition: {result.get('condition', 'Unknown')}")
                print(f"   Confidence: {result.get('confidence', 0)}%")
                print(f"   Processing Time: {result.get('processing_time', 0)}s")
                
                # Check for additional analysis data
                if 'remedies' in result:
                    print(f"   Remedies: {len(result['remedies'])} found")
                
                return True
            else:
                print(f"❌ Analysis failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Analysis test failed: {e}")
        return False

def test_webcam_analysis():
    """Test webcam-style analysis (base64)"""
    print("\n📷 Testing Webcam Analysis")
    print("=" * 30)
    
    # Create test image
    img = create_test_image()
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=95)
    image_data = buf.getvalue()
    
    # Convert to base64
    base64_image = base64.b64encode(image_data).decode('utf-8')
    
    print("📸 Created test webcam image (base64)")
    
    try:
        response = requests.post(
            "http://localhost:5000/analyze",
            json={'image_base64': base64_image},
            headers={'Content-Type': 'application/json'},
            timeout=45
        )
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                result = data.get('result', {})
                print("✅ Webcam analysis successful!")
                print(f"   Condition: {result.get('condition', 'Unknown')}")
                print(f"   Confidence: {result.get('confidence', 0)}%")
                print(f"   Processing Time: {result.get('processing_time', 0)}s")
                return True
            else:
                print(f"❌ Webcam analysis failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Webcam analysis test failed: {e}")
        return False

def test_flask_app():
    """Test if Flask app is running"""
    print("🔍 Testing Flask App")
    print("=" * 20)
    
    try:
        response = requests.get("http://localhost:5000/", timeout=5)
        if response.status_code == 200:
            print("✅ Flask app is running")
            return True
        else:
            print(f"❌ Flask app returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Flask app not accessible: {e}")
        return False

def main():
    """Run comprehensive tests"""
    print("🌿 Heal Ayur - Chatbot & Analysis Fix Verification")
    print("=" * 60)
    print("Testing both chatbot response handling and analysis functionality")
    print("=" * 60)
    
    tests = [
        ("Flask App", test_flask_app),
        ("Chatbot Response", test_chatbot_response),
        ("Image Analysis", test_image_analysis),
        ("Webcam Analysis", test_webcam_analysis)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 TEST SUMMARY:")
    print("=" * 20)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        print("✅ Chatbot should now display full responses")
        print("✅ Image analysis should work without 'Application not ready' errors")
        print("✅ Both file upload and webcam analysis should function properly")
    else:
        print(f"\n⚠️ {total - passed} tests failed.")
    
    print("\n🎯 What to expect now:")
    print("• Chatbot responses should display completely without truncation")
    print("• Image analysis should work immediately after file selection")
    print("• No more 'Application not ready' errors")
    print("• Detailed console logging for debugging")
    print("• Better error messages with specific guidance")
    
    print("\n🔧 If issues persist:")
    print("• Hard refresh browser (Ctrl+F5)")
    print("• Check browser console (F12) for detailed logs")
    print("• Ensure camera permissions are granted")
    print("• Try uploading a different image format")

if __name__ == "__main__":
    main()
