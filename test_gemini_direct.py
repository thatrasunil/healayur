"""
Test Gemini analysis directly
"""

import requests
import base64
from PIL import Image
import io

def test_gemini_analysis():
    """Test the Gemini analysis endpoint directly"""
    
    # Create a test image
    img = Image.new('RGB', (224, 224), color=(200, 150, 120))
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    
    # Test with file upload
    print("🧪 Testing Gemini analysis with file upload...")
    
    files = {'image': ('test.jpg', buf.getvalue(), 'image/jpeg')}
    
    try:
        response = requests.post('http://localhost:5000/analyze', files=files)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text[:500]}...")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ Analysis successful!")
                print(f"Condition: {result['result'].get('condition', 'Unknown')}")
                print(f"Confidence: {result['result'].get('confidence', 0)}%")
                if 'ai_analysis' in result['result']:
                    ai_data = result['result']['ai_analysis']
                    print(f"Gender: {ai_data.get('gender_analysis', {}).get('detected_gender', 'Unknown')}")
                    print(f"Age: {ai_data.get('age_analysis', {}).get('estimated_age', 'Unknown')}")
            else:
                print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_gemini_analysis()
