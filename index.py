import os
import sys

# Set environment variables for Vercel
os.environ['VERCEL_DEPLOYMENT'] = 'true'

try:
    # Import the Flask app
    from app import app

    # Configure for Vercel
    app.config['DEBUG'] = False
    app.config['TESTING'] = False

    print("✅ App imported successfully")

except Exception as e:
    print(f"❌ Error importing app: {e}")
    # Create a minimal Flask app as fallback
    from flask import Flask, jsonify

    app = Flask(__name__)

    @app.route('/')
    def hello():
        return """
        <html>
        <head><title>Heal Ayur</title></head>
        <body>
            <h1>🌿 Heal Ayur</h1>
            <p>AI-Powered Ancient Healing Platform</p>
            <p>The application is starting up... Please refresh in a moment.</p>
            <p>If this message persists, please check the deployment logs.</p>
        </body>
        </html>
        """

    @app.route('/health')
    def health():
        return jsonify({"status": "ok", "message": "Heal Ayur API is running"})

    @app.route('/api/status')
    def status():
        return jsonify({
            "status": "starting",
            "message": "Application is initializing",
            "error": str(e) if 'e' in locals() else "Unknown error"
        })

# For direct execution
if __name__ == "__main__":
    app.run(debug=False)
