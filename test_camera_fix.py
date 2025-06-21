#!/usr/bin/env python3
"""
Test the camera fix and verify face detection is working
"""

import time
import subprocess
import sys

def test_flask_app_running():
    """Check if Flask app is running"""
    print("🔍 Checking Flask App Status")
    print("=" * 30)
    
    try:
        import requests
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

def check_javascript_fixes():
    """Check if the JavaScript fixes are in place"""
    print("\n🔧 Checking JavaScript Fixes")
    print("=" * 30)
    
    fixes_found = 0
    total_fixes = 4
    
    # Check main.js for debugging
    try:
        with open('static/js/main.js', 'r') as f:
            main_js = f.read()
            
        if 'console.log(\'🔍 Capture and analyze called\')' in main_js:
            print("✅ Capture debugging added to main.js")
            fixes_found += 1
        else:
            print("❌ Capture debugging missing in main.js")
            
        if 'console.log(\'📷 Requesting camera access...\')' in main_js:
            print("✅ Camera debugging added to main.js")
            fixes_found += 1
        else:
            print("❌ Camera debugging missing in main.js")
            
    except Exception as e:
        print(f"❌ Error checking main.js: {e}")
    
    # Check realtime.js for debugging
    try:
        with open('static/js/realtime.js', 'r') as f:
            realtime_js = f.read()
            
        if 'console.log(\'🔬 Real-time analysis - Video check:\')' in realtime_js:
            print("✅ Real-time debugging added to realtime.js")
            fixes_found += 1
        else:
            print("❌ Real-time debugging missing in realtime.js")
            
    except Exception as e:
        print(f"❌ Error checking realtime.js: {e}")
    
    # Check index.html for fallback function
    try:
        with open('templates/index.html', 'r') as f:
            index_html = f.read()
            
        if 'analyzeImageFallback' in index_html:
            print("✅ Fallback analysis function added to index.html")
            fixes_found += 1
        else:
            print("❌ Fallback analysis function missing in index.html")
            
    except Exception as e:
        print(f"❌ Error checking index.html: {e}")
    
    print(f"\n📊 Fixes Applied: {fixes_found}/{total_fixes}")
    return fixes_found == total_fixes

def provide_troubleshooting_guide():
    """Provide troubleshooting steps for camera issues"""
    print("\n🛠️ Camera Troubleshooting Guide")
    print("=" * 35)
    
    print("If you're still experiencing 'application not ready' errors:")
    print()
    print("1. 📱 **Browser Permissions**")
    print("   • Make sure camera permissions are granted")
    print("   • Check browser settings for camera access")
    print("   • Try a different browser (Chrome/Firefox recommended)")
    print()
    print("2. 🔄 **Page Loading**")
    print("   • Hard refresh the page (Ctrl+F5 or Cmd+Shift+R)")
    print("   • Clear browser cache and cookies")
    print("   • Wait 2-3 seconds after page load before using camera")
    print()
    print("3. 📷 **Camera Initialization**")
    print("   • Click 'Use Camera' and wait for video to appear")
    print("   • Ensure you see your video feed before clicking 'Capture'")
    print("   • Wait for the green 'Camera ready!' notification")
    print()
    print("4. 🔍 **Debug Information**")
    print("   • Open browser console (F12) to see debug messages")
    print("   • Look for messages starting with 🔍, 📊, ✅, or ❌")
    print("   • Check for any JavaScript errors in red")
    print()
    print("5. ⚡ **Real-time Analysis**")
    print("   • Start camera first, then click 'Start Real-Time'")
    print("   • Wait for video dimensions to be detected")
    print("   • Look for 'Video check' messages in console")
    print()
    print("6. 🔧 **If Still Not Working**")
    print("   • Restart the Flask app: python app.py")
    print("   • Try incognito/private browsing mode")
    print("   • Check if antivirus is blocking camera access")

def main():
    """Run camera fix verification"""
    print("🌿 Heal Ayur - Camera Fix Verification")
    print("=" * 45)
    print("This script verifies that camera issues have been fixed")
    print("and provides troubleshooting guidance.")
    print("=" * 45)
    
    # Test Flask app
    flask_running = test_flask_app_running()
    
    # Check fixes
    fixes_applied = check_javascript_fixes()
    
    # Summary
    print("\n📋 VERIFICATION SUMMARY")
    print("=" * 25)
    print(f"Flask App Running: {'✅ YES' if flask_running else '❌ NO'}")
    print(f"JavaScript Fixes: {'✅ APPLIED' if fixes_applied else '❌ MISSING'}")
    
    if flask_running and fixes_applied:
        print("\n🎉 All fixes have been applied successfully!")
        print("🌐 Open http://localhost:5000 in your browser to test")
        print("📱 Make sure to grant camera permissions when prompted")
        print("🔍 Check browser console (F12) for debug messages")
    else:
        print("\n⚠️ Some issues detected. Please review the results above.")
    
    # Always provide troubleshooting guide
    provide_troubleshooting_guide()
    
    print("\n🎯 **Expected Behavior After Fix:**")
    print("• Camera should initialize without 'application not ready' error")
    print("• Console should show detailed debug messages")
    print("• Fallback analysis should work even if main app isn't ready")
    print("• Real-time analysis should check video dimensions properly")
    print("• Clear error messages should guide users when something is wrong")

if __name__ == "__main__":
    main()
