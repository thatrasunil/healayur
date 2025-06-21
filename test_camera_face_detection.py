#!/usr/bin/env python3
"""
Test camera face detection and diagnose "application is not ready" error
"""

import requests
import base64
import json
import time
from PIL import Image, ImageDraw
import io

def create_test_face_image():
    """Create a test face image for testing"""
    # Create a realistic face image
    img = Image.new('RGB', (640, 480), color=(240, 220, 200))
    draw = ImageDraw.Draw(img)
    
    # Face outline
    draw.ellipse([160, 120, 480, 360], fill=(220, 190, 160), outline=(200, 170, 140), width=2)
    
    # Eyes
    draw.ellipse([220, 180, 260, 210], fill=(255, 255, 255), outline=(0, 0, 0), width=1)
    draw.ellipse([230, 185, 250, 205], fill=(100, 50, 20), outline=(0, 0, 0), width=1)
    draw.ellipse([235, 190, 245, 200], fill=(0, 0, 0))
    
    draw.ellipse([380, 180, 420, 210], fill=(255, 255, 255), outline=(0, 0, 0), width=1)
    draw.ellipse([390, 185, 410, 205], fill=(100, 50, 20), outline=(0, 0, 0), width=1)
    draw.ellipse([395, 190, 405, 200], fill=(0, 0, 0))
    
    # Nose
    draw.polygon([(320, 220), (310, 260), (330, 260)], fill=(210, 180, 150))
    
    # Mouth
    draw.ellipse([290, 300, 350, 330], fill=(180, 100, 100))
    
    return img

def test_flask_app_status():
    """Test if Flask app is running and ready"""
    print("🔍 Testing Flask App Status")
    print("=" * 30)
    
    try:
        response = requests.get("http://localhost:5000/", timeout=5)
        if response.status_code == 200:
            print("✅ Flask app is running and accessible")
            return True
        else:
            print(f"❌ Flask app returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Flask app is not running")
        print("💡 Please start the Flask app: python app.py")
        return False
    except Exception as e:
        print(f"❌ Error connecting to Flask app: {e}")
        return False

def test_analysis_endpoint():
    """Test the analysis endpoint with a face image"""
    print("\n🧪 Testing Analysis Endpoint")
    print("=" * 30)
    
    try:
        # Create test face image
        img = create_test_face_image()
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=95)
        image_data = buf.getvalue()
        
        print("📸 Created test face image")
        
        # Test with file upload method
        print("🔬 Testing file upload analysis...")
        files = {'image': ('test_face.jpg', image_data, 'image/jpeg')}
        
        response = requests.post(
            "http://localhost:5000/analyze",
            files=files,
            timeout=30
        )
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                result = data.get('result', {})
                print("✅ File upload analysis successful!")
                print(f"   Condition: {result.get('condition', 'Unknown')}")
                print(f"   Confidence: {result.get('confidence', 0)}%")
                print(f"   Processing Time: {result.get('processing_time', 0)}s")
                
                # Check for face detection info
                if 'face_detected' in result:
                    print(f"   Face Detected: {result.get('face_detected', False)}")
                
                if 'gender_analysis' in result:
                    gender = result['gender_analysis']
                    print(f"   Gender: {gender.get('detected_gender', 'Unknown')}")
                
                if 'age_analysis' in result:
                    age = result['age_analysis']
                    print(f"   Age: {age.get('estimated_age', 'Unknown')}")
                
                return True
            else:
                print(f"❌ Analysis failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Analysis endpoint test failed: {e}")
        return False

def test_webcam_analysis():
    """Test webcam-style analysis (base64)"""
    print("\n📷 Testing Webcam Analysis")
    print("=" * 30)
    
    try:
        # Create test face image
        img = create_test_face_image()
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=95)
        image_data = buf.getvalue()
        
        # Convert to base64
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        print("📸 Created test webcam image (base64)")
        
        # Test webcam analysis
        response = requests.post(
            "http://localhost:5000/analyze",
            json={'image_base64': base64_image},
            headers={'Content-Type': 'application/json'},
            timeout=30
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
                
                # Check for face detection info
                if 'face_detected' in result:
                    print(f"   Face Detected: {result.get('face_detected', False)}")
                
                if 'faceplus_available' in result:
                    print(f"   Face++ Available: {result.get('faceplus_available', False)}")
                
                if 'fallback_analysis' in result:
                    print(f"   Using Fallback: {result.get('fallback_analysis', False)}")
                
                return True
            else:
                print(f"❌ Webcam analysis failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Webcam analysis test failed: {e}")
        return False

def test_face_plus_integration():
    """Test Face++ API integration specifically"""
    print("\n🤖 Testing Face++ Integration")
    print("=" * 30)
    
    try:
        from ai_analysis_faceplus import analyze_image
        
        # Create test face image
        img = create_test_face_image()
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=95)
        image_data = buf.getvalue()
        
        print("🔬 Running Face++ analysis...")
        result = analyze_image(image_data)
        
        print("✅ Face++ analysis completed!")
        print(f"   Condition: {result.get('condition', 'Unknown')}")
        print(f"   Confidence: {result.get('confidence', 0):.1%}")
        print(f"   Face Detected: {result.get('face_detected', False)}")
        
        if 'faceplus_available' in result:
            print(f"   Face++ Available: {result.get('faceplus_available', False)}")
        
        if 'fallback_analysis' in result:
            print(f"   Using Fallback: {result.get('fallback_analysis', False)}")
        
        if 'gender_analysis' in result:
            gender = result['gender_analysis']
            print(f"   Gender: {gender.get('detected_gender', 'Unknown')}")
            print(f"   Analysis Method: {gender.get('analysis_method', 'Unknown')}")
        
        if 'age_analysis' in result:
            age = result['age_analysis']
            print(f"   Age: {age.get('estimated_age', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Face++ integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_error_messages():
    """Check for common error messages"""
    print("\n🔍 Checking for Error Sources")
    print("=" * 30)
    
    # Check if "application is not ready" appears in any files
    files_to_check = [
        'app.py',
        'ai_analysis_faceplus.py',
        'static/js/main.js',
        'static/js/realtime.js',
        'templates/index.html'
    ]
    
    found_errors = []
    
    for filename in files_to_check:
        try:
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if 'application is not ready' in content.lower():
                    found_errors.append(filename)
                    print(f"❌ Found 'application is not ready' in {filename}")
        except Exception as e:
            print(f"⚠️ Could not check {filename}: {e}")
    
    if not found_errors:
        print("✅ No 'application is not ready' messages found in code")
    
    return len(found_errors) == 0

def main():
    """Run complete camera face detection diagnostic"""
    print("🌿 Heal Ayur - Camera Face Detection Diagnostic")
    print("=" * 55)
    print("This diagnostic will test camera functionality and face detection")
    print("to identify the source of 'application is not ready' error.")
    print("=" * 55)
    
    tests = [
        ("Flask App Status", test_flask_app_status),
        ("Analysis Endpoint", test_analysis_endpoint),
        ("Webcam Analysis", test_webcam_analysis),
        ("Face++ Integration", test_face_plus_integration),
        ("Error Message Check", check_error_messages)
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
    print("📋 DIAGNOSTIC SUMMARY:")
    print("=" * 25)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Camera and face detection should be working.")
        print("If you're still seeing 'application is not ready', it might be:")
        print("• Browser cache issue - try hard refresh (Ctrl+F5)")
        print("• JavaScript loading issue - check browser console")
        print("• Timing issue - wait a moment after starting camera")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Issues detected:")
        print("• Check Flask app is running: python app.py")
        print("• Verify Face++ API credentials are correct")
        print("• Check browser console for JavaScript errors")
        print("• Ensure camera permissions are granted")

if __name__ == "__main__":
    main()
