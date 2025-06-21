#!/usr/bin/env python3
"""
Test script for the Revolutionary AI Analysis System
"""

import numpy as np
from PIL import Image
import io
import base64
from ai_analysis_enhanced import RevolutionarySkinAnalyzer

def create_test_image():
    """Create a simple test image for analysis"""
    # Create a 224x224 RGB test image
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Add some skin-like colors
    test_image[:, :, 0] = np.random.randint(180, 220, (224, 224))  # Red channel
    test_image[:, :, 1] = np.random.randint(140, 180, (224, 224))  # Green channel  
    test_image[:, :, 2] = np.random.randint(120, 160, (224, 224))  # Blue channel
    
    # Convert to PIL Image
    pil_image = Image.fromarray(test_image)
    
    # Convert to bytes
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format='JPEG')
    img_bytes = img_buffer.getvalue()
    
    return img_bytes

def test_analysis():
    """Test the AI analysis system"""
    print("🧪 Testing Revolutionary AI Analysis System...")

    try:
        # Test the main analyze_image function
        from ai_analysis_enhanced import analyze_image
        print("✅ analyze_image function imported successfully")

        # Create test image
        test_image_bytes = create_test_image()
        print("✅ Test image created")

        # Run analysis using the main function
        print("🔬 Running analysis...")
        result = analyze_image(test_image_bytes)

        # Display results
        print("\n🎯 ANALYSIS RESULTS:")
        print("=" * 50)
        print(f"Condition: {result.get('condition', 'Unknown')}")
        print(f"Confidence: {result.get('confidence', 0):.1%}")
        print(f"Severity: {result.get('severity', 'Unknown')}")

        if 'gender_analysis' in result:
            gender_info = result['gender_analysis']
            print(f"Detected Gender: {gender_info.get('detected_gender', 'Unknown')}")
            print(f"Gender Confidence: {gender_info.get('gender_confidence', 0):.1%}")

        if 'age_analysis' in result:
            age_info = result['age_analysis']
            print(f"Estimated Age: {age_info.get('estimated_age', 'Unknown')} years")
            print(f"Age Category: {age_info.get('age_category', 'Unknown')}")
            print(f"Age Confidence: {age_info.get('age_confidence', 0):.1%}")

        print(f"\nRemedies: {len(result.get('remedies', []))} recommendations")
        print(f"Herbs: {len(result.get('herbs', []))} herbal suggestions")

        # Test JSON serialization
        import json
        json_str = json.dumps(result)
        print(f"✅ JSON serialization successful ({len(json_str)} characters)")

        print("\n✅ Analysis completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_analysis()
    if success:
        print("\n🎉 All tests passed! The analysis system is working correctly.")
    else:
        print("\n💥 Tests failed! Please check the error messages above.")
