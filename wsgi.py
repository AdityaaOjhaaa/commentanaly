import sys
import os

# Add your site directory to the python path
path = '/home/ojhaditya123/commentanaly'
if path not in sys.path:
    sys.path.append(path)

# Import your Flask app
from app import app as application

# Add environment variables
os.environ['YOUTUBE_API_KEY'] = 'AIzaSyAAVGD1a6hn8X_SAokeVRzGQSjtW-1s18A'  # Replace with your actual YouTube API key
os.environ['APIFY_API_TOKEN'] = 'apify_api_tSneM2HbMF5AlhDwIDVOm9h875aiio1RxpRb'  # Replace with your actual Apify token

# Optional: Configure logging
import logging
logging.basicConfig(stream=sys.stderr)
