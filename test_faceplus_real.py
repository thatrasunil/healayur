#!/usr/bin/env python3
"""
Test Face++ API with a real face image
"""

import requests
import base64
import json
from PIL import Image, ImageDraw
import io

def create_realistic_face_image():
    """Create a more realistic face-like image for testing"""
    # Create a larger image with better proportions
    img = Image.new('RGB', (400, 500), color=(240, 220, 200))  # Light skin tone
    draw = ImageDraw.Draw(img)
    
    # Face outline (oval)
    draw.ellipse([50, 80, 350, 420], fill=(220, 190, 160), outline=(200, 170, 140), width=2)
    
    # Eyes (more realistic)
    # Left eye
    draw.ellipse([120, 180, 180, 210], fill=(255, 255, 255), outline=(0, 0, 0), width=1)
    draw.ellipse([135, 185, 165, 205], fill=(100, 50, 20), outline=(0, 0, 0), width=1)  # Iris
    draw.ellipse([145, 190, 155, 200], fill=(0, 0, 0))  # Pupil
    
    # Right eye
    draw.ellipse([220, 180, 280, 210], fill=(255, 255, 255), outline=(0, 0, 0), width=1)
    draw.ellipse([235, 185, 265, 205], fill=(100, 50, 20), outline=(0, 0, 0), width=1)  # Iris
    draw.ellipse([245, 190, 255, 200], fill=(0, 0, 0))  # Pupil
    
    # Eyebrows
    draw.ellipse([110, 160, 190, 175], fill=(80, 50, 30))  # Left eyebrow
    draw.ellipse([210, 160, 290, 175], fill=(80, 50, 30))  # Right eyebrow
    
    # Nose
    draw.polygon([(200, 220), (190, 260), (210, 260)], fill=(210, 180, 150), outline=(190, 160, 130))
    
    # Mouth
    draw.ellipse([170, 300, 230, 330], fill=(180, 100, 100), outline=(160, 80, 80), width=1)
    
    # Hair
    draw.ellipse([40, 60, 360, 200], fill=(60, 40, 20))  # Hair top
    
    return img

def test_faceplus_with_real_image():
    """Test Face++ API with a realistic face image"""
    print("🔑 Testing Face++ API with Realistic Face Image")
    print("=" * 50)
    
    # API Configuration
    API_KEY = "8f0FzOLKgxrCCai5NzAkiGp_D-5XRd-f"
    API_SECRET = "8YbEz4LCUZ2-Furl5Oohw0RonwgV18CX"
    API_URL = "https://api-us.faceplusplus.com/facepp/v3/detect"
    
    try:
        # Create realistic face image
        print("🎨 Creating realistic face image...")
        img = create_realistic_face_image()
        
        # Convert to bytes
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=95)
        image_data = buf.getvalue()
        
        print("✅ Realistic face image created")
        
        # Test Face++ API with file upload
        print("🔍 Testing Face++ API with file upload...")
        
        files = {'image_file': ('face.jpg', image_data, 'image/jpeg')}
        data = {
            'api_key': API_KEY,
            'api_secret': API_SECRET,
            'return_attributes': 'gender,age,emotion,beauty,skinstatus'
        }
        
        response = requests.post(API_URL, data=data, files=files, timeout=30)
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Face++ API request successful!")
            
            # Print full response for debugging
            print(f"📋 Full Response: {json.dumps(result, indent=2)}")
            
            if 'faces' in result and len(result['faces']) > 0:
                print(f"👤 Detected {len(result['faces'])} face(s)")
                
                face = result['faces'][0]
                attributes = face.get('attributes', {})
                
                print("\n🎯 FACE++ ANALYSIS RESULTS:")
                print("-" * 30)
                
                # Display all available attributes
                for attr_name, attr_data in attributes.items():
                    print(f"{attr_name}: {attr_data}")
                
                print("\n✅ Face++ API integration working correctly!")
                return True
                
            else:
                print("⚠️ No faces detected in the realistic image")
                print("💡 The image might need more realistic features")
                return False
                
        else:
            print(f"❌ Face++ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Face++ API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_base64_method():
    """Test Face++ API with base64 encoding"""
    print("\n🔍 Testing Face++ API with Base64 encoding...")
    
    API_KEY = "8f0FzOLKgxrCCai5NzAkiGp_D-5XRd-f"
    API_SECRET = "8YbEz4LCUZ2-Furl5Oohw0RonwgV18CX"
    API_URL = "https://api-us.faceplusplus.com/facepp/v3/detect"
    
    try:
        # Create realistic face image
        img = create_realistic_face_image()
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=95)
        image_data = buf.getvalue()
        
        # Test with base64
        data = {
            'api_key': API_KEY,
            'api_secret': API_SECRET,
            'image_base64': base64.b64encode(image_data).decode('utf-8'),
            'return_attributes': 'gender,age,emotion,beauty,skinstatus'
        }
        
        response = requests.post(API_URL, data=data, timeout=30)
        
        print(f"📊 Base64 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Base64 method successful!")
            
            if 'faces' in result and len(result['faces']) > 0:
                print(f"👤 Base64 method detected {len(result['faces'])} face(s)")
                return True
            else:
                print("⚠️ Base64 method: No faces detected")
                return False
        else:
            print(f"❌ Base64 method failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Base64 test failed: {e}")
        return False

def main():
    """Run Face++ tests with realistic images"""
    print("🌿 Heal Ayur - Face++ API Real Image Test Suite")
    print("=" * 60)
    
    # Test 1: File upload method
    file_success = test_faceplus_with_real_image()
    
    # Test 2: Base64 method
    base64_success = test_base64_method()
    
    print("\n📋 TEST SUMMARY:")
    print("=" * 20)
    print(f"File Upload Method: {'✅ PASS' if file_success else '❌ FAIL'}")
    print(f"Base64 Method: {'✅ PASS' if base64_success else '❌ FAIL'}")
    
    if file_success or base64_success:
        print("\n🎉 Face++ API is working! At least one method succeeded.")
        return True
    else:
        print("\n❌ Both methods failed. Check API credentials or image format.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
