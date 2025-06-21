import os
import sys

# Set environment variables for Vercel
os.environ['VERCEL_DEPLOYMENT'] = 'true'

try:
    # Try to import the simplified Vercel app first
    from app_vercel import app
    print("✅ Vercel-optimized app imported successfully")

except Exception as e1:
    print(f"⚠️ Vercel app failed: {e1}")
    try:
        # Fallback to main app
        from app import app
        print("✅ Main app imported successfully")

    except Exception as e2:
        print(f"❌ Main app failed: {e2}")
        # Create a minimal Flask app as final fallback
        from flask import Flask, jsonify

        app = Flask(__name__)

        @app.route('/')
        def hello():
            return """
            <html>
            <head><title>Heal Ayur</title></head>
            <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                <h1>🌿 Heal Ayur</h1>
                <p>AI-Powered Ancient Healing Platform</p>
                <p>The application is starting up... Please refresh in a moment.</p>
                <p style="color: #666;">If this message persists, the app may be loading dependencies.</p>
                <div style="margin-top: 30px;">
                    <a href="/health" style="color: #4CAF50;">Check Health Status</a>
                </div>
            </body>
            </html>
            """

        @app.route('/health')
        def health():
            return jsonify({
                "status": "fallback",
                "message": "Heal Ayur API is running in fallback mode",
                "errors": {
                    "vercel_app": str(e1),
                    "main_app": str(e2)
                }
            })

# Configure for Vercel
app.config['DEBUG'] = False
app.config['TESTING'] = False

# For direct execution
if __name__ == "__main__":
    app.run(debug=False)
