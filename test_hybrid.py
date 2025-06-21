"""
Test the hybrid Face++ & Gemini analysis system
"""

from ai_analysis_faceplus import analyze_image
from PIL import Image
import io

def test_hybrid_analysis():
    """Test the hybrid analysis system"""
    print("🧪 Testing Hybrid Face++ & Gemini Analysis System...")
    
    try:
        # Create a test image
        img = Image.new('RGB', (224, 224), color=(200, 150, 120))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        buf.seek(0)
        
        print("✅ Test image created")
        
        # Run analysis
        print("🔬 Running hybrid analysis...")
        result = analyze_image(buf.getvalue())
        
        # Display results
        print("\n🎯 HYBRID ANALYSIS RESULTS:")
        print("=" * 50)
        print(f"Condition: {result.get('condition', 'Unknown')}")
        print(f"Confidence: {result.get('confidence', 0):.1%}")
        print(f"Severity: {result.get('severity', 'Unknown')}")
        print(f"Face Detected: {result.get('face_detected', False)}")
        
        if 'gender_analysis' in result:
            gender_info = result['gender_analysis']
            print(f"Detected Gender: {gender_info.get('detected_gender', 'Unknown')}")
            print(f"Gender Confidence: {gender_info.get('gender_confidence', 0):.1%}")
        
        if 'age_analysis' in result:
            age_info = result['age_analysis']
            print(f"Estimated Age: {age_info.get('estimated_age', 'Unknown')} years")
            print(f"Age Category: {age_info.get('age_category', 'Unknown')}")
        
        print(f"\nRemedies: {len(result.get('remedies', []))} recommendations")
        print(f"Herbs: {len(result.get('herbs', []))} herbal suggestions")
        
        print("\n✅ Hybrid analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Hybrid analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_hybrid_analysis()
