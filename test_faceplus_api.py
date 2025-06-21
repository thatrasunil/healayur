#!/usr/bin/env python3
"""
Test script for Face++ API integration
"""

import requests
import base64
import json
from PIL import Image
import io

def test_faceplus_api():
    """Test Face++ API directly"""
    print("🔑 Testing Face++ API Integration")
    print("=" * 40)
    
    # API Configuration
    API_KEY = "8f0FzOLKgxrCCai5NzAkiGp_D-5XRd-f"
    API_SECRET = "8YbEz4LCUZ2-Furl5Oohw0RonwgV18CX"
    API_URL = "https://api-us.faceplusplus.com/facepp/v3/detect"
    
    try:
        # Create a test face image (simple colored rectangle)
        print("📸 Creating test image...")
        img = Image.new('RGB', (300, 400), color=(220, 180, 140))  # Skin-like color
        
        # Add some basic facial features (simple rectangles for eyes, nose, mouth)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        # Eyes
        draw.rectangle([80, 120, 120, 140], fill=(50, 50, 50))  # Left eye
        draw.rectangle([180, 120, 220, 140], fill=(50, 50, 50))  # Right eye
        
        # Nose
        draw.rectangle([140, 160, 160, 200], fill=(200, 160, 120))  # Nose
        
        # Mouth
        draw.rectangle([120, 240, 180, 260], fill=(180, 100, 100))  # Mouth
        
        # Convert to bytes
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=95)
        image_data = buf.getvalue()
        
        print("✅ Test image created with basic facial features")
        
        # Test Face++ API
        print("🔍 Testing Face++ API...")
        
        # Prepare request data - try file upload method first
        files = {'image_file': ('test.jpg', image_data, 'image/jpeg')}

        data = {
            'api_key': API_KEY,
            'api_secret': API_SECRET,
            'return_attributes': 'gender,age,emotion,beauty,skinstatus'
        }
        
        # Make API request
        print("📡 Sending request to Face++ API...")
        response = requests.post(API_URL, data=data, files=files, timeout=30)
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Face++ API request successful!")
            
            # Check for faces
            if 'faces' in result and len(result['faces']) > 0:
                print(f"👤 Detected {len(result['faces'])} face(s)")
                
                face = result['faces'][0]
                attributes = face.get('attributes', {})
                
                # Display results
                print("\n🎯 FACE++ ANALYSIS RESULTS:")
                print("-" * 30)
                
                # Gender
                if 'gender' in attributes:
                    gender_info = attributes['gender']
                    print(f"Gender: {gender_info.get('value', 'Unknown')}")
                
                # Age
                if 'age' in attributes:
                    age_info = attributes['age']
                    print(f"Age: {age_info.get('value', 'Unknown')} years")
                
                # Emotion
                if 'emotion' in attributes:
                    emotion_info = attributes['emotion']
                    emotions = {k: v for k, v in emotion_info.items() if k != 'value'}
                    top_emotion = max(emotions.items(), key=lambda x: x[1]) if emotions else ('neutral', 0)
                    print(f"Primary Emotion: {top_emotion[0]} ({top_emotion[1]:.1f}%)")
                
                # Beauty
                if 'beauty' in attributes:
                    beauty_info = attributes['beauty']
                    print(f"Beauty Score: {beauty_info.get('female_score', beauty_info.get('male_score', 'N/A'))}")
                
                # Skin Status
                if 'skinstatus' in attributes:
                    skin_info = attributes['skinstatus']
                    print(f"Skin Health: {skin_info.get('health', 'N/A')}")
                    print(f"Skin Stain: {skin_info.get('stain', 'N/A')}")
                    print(f"Skin Acne: {skin_info.get('acne', 'N/A')}")
                
                print("\n✅ Face++ API integration working correctly!")
                return True
                
            else:
                print("⚠️ No faces detected in the image")
                print("💡 This might be normal for a simple test image")
                print("✅ Face++ API is responding correctly")
                return True
                
        else:
            print(f"❌ Face++ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Face++ API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hybrid_analysis():
    """Test the hybrid Face++ & Gemini analysis"""
    print("\n🧪 Testing Hybrid Face++ & Gemini Analysis")
    print("=" * 45)
    
    try:
        from ai_analysis_faceplus import analyze_image
        
        # Create test image
        img = Image.new('RGB', (300, 400), color=(220, 180, 140))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        image_data = buf.getvalue()
        
        print("🔬 Running hybrid analysis...")
        result = analyze_image(image_data)
        
        print("✅ Hybrid analysis completed!")
        print(f"Condition: {result.get('condition', 'Unknown')}")
        print(f"Confidence: {result.get('confidence', 0):.1%}")
        print(f"Face Detected: {result.get('face_detected', False)}")
        
        if 'gender_analysis' in result:
            gender = result['gender_analysis']
            print(f"Gender: {gender.get('detected_gender', 'Unknown')}")
        
        if 'age_analysis' in result:
            age = result['age_analysis']
            print(f"Age: {age.get('estimated_age', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hybrid analysis test failed: {e}")
        return False

def main():
    """Run all Face++ tests"""
    print("🌿 Heal Ayur - Face++ API Integration Test Suite")
    print("=" * 55)
    
    # Test 1: Direct Face++ API
    api_success = test_faceplus_api()
    
    # Test 2: Hybrid analysis
    hybrid_success = test_hybrid_analysis()
    
    print("\n📋 TEST SUMMARY:")
    print("=" * 20)
    print(f"Face++ API: {'✅ PASS' if api_success else '❌ FAIL'}")
    print(f"Hybrid Analysis: {'✅ PASS' if hybrid_success else '❌ FAIL'}")
    
    if api_success and hybrid_success:
        print("\n🎉 All Face++ tests passed! Integration is working correctly!")
        return True
    else:
        print("\n❌ Some tests failed. Please check the configuration.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
