# Vercel entry point for Flask app
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables for Vercel
os.environ['VERCEL_DEPLOYMENT'] = 'true'

try:
    # Import the Flask app
    from app import app

    # Configure for Vercel
    app.config['DEBUG'] = False
    app.config['TESTING'] = False

    # Disable file logging for Vercel
    import logging
    logging.basicConfig(level=logging.WARNING)

except Exception as e:
    print(f"Error importing app: {e}")
    # Create a minimal Flask app as fallback
    from flask import Flask
    app = Flask(__name__)

    @app.route('/')
    def hello():
        return "Heal Ayur is starting up... Please refresh in a moment."

    @app.route('/health')
    def health():
        return {"status": "ok", "message": "Heal Ayur API is running"}

# Export the app for Vercel
application = app

# For direct execution
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
