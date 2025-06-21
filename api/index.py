import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables for Vercel
os.environ['VERCEL_DEPLOYMENT'] = 'true'

# Import the Flask app
from app import app

# Configure for Vercel
app.config['DEBUG'] = False
app.config['TESTING'] = False

# For direct execution
if __name__ == "__main__":
    app.run(debug=False)
