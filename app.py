import os
import re
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, render_template_string
import googleapiclient.discovery
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
import plotly.express as px
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from apify_client import ApifyClient
import emoji
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from urllib.parse import urlparse
from datetime import datetime
import traceback
import sys

# Flask app configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Database configuration
if os.environ.get('RENDER'):
    # Use Render's persistent storage when deployed
    db_path = os.path.join('/data', 'users.db')
    os.makedirs('/data', exist_ok=True)
else:
    # Use local path for development
    db_path = 'users.db'

app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    join_date = db.Column(db.DateTime, default=datetime.utcnow)
    analyses = db.relationship('Analysis', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Analysis Model to store user's analysis history
class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    url = db.Column(db.String(500), nullable=False)
    platform = db.Column(db.String(50), nullable=False)
    total_comments = db.Column(db.Integer)
    positive_percent = db.Column(db.Float)
    negative_percent = db.Column(db.Float)
    neutral_percent = db.Column(db.Float)
    date_analyzed = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Your existing configuration
APIFY_API_TOKEN = "apify_api_tSneM2HbMF5AlhDwIDVOm9h875aiio1RxpRb"  # Replace with your Apify API token
apify_client = ApifyClient(APIFY_API_TOKEN)
API_KEY = "YOUR_YOUTUBE_API_KEY"  # Replace with your YouTube API key

# Your existing functions remain the same
# (extract_instagram_info, get_instagram_comments_apify, extract_video_id, etc.)
# Continue with your existing helper functions

def extract_instagram_info(url):
    """Extract post or reel ID from an Instagram URL."""
    post_pattern = r'(?:https?:\/\/)?(?:www\.)?instagram\.com(?:\/p\/|\/reel\/)([a-zA-Z0-9_-]+)(?:\/)?(\?.*)?$'
    reel_pattern = r'(?:https?:\/\/)?(?:www\.)?instagram\.com\/reels?\/([a-zA-Z0-9_-]+)(?:\/)?(\?.*)?$'

    post_match = re.match(post_pattern, url)
    if post_match:
        return post_match.group(1), 'instagram_post'

    reel_match = re.match(reel_pattern, url)
    if reel_match:
        return reel_match.group(1), 'instagram_reel'

    return None, None

# Your existing functions remain the same
def get_instagram_comments_apify(shortcode, max_comments=100):
    """Fetch comments from an Instagram post or reel using Apify."""
    try:
        run_input = {
            "directUrls": [f"https://www.instagram.com/p/{shortcode}/"],
            "resultsLimit": max_comments
        }
        run = apify_client.actor("apify/instagram-comment-scraper").call(run_input=run_input)
        comments = []
        for item in apify_client.dataset(run["defaultDatasetId"]).iterate_items():
            if "text" in item and item["text"]:
                comments.append(item["text"])
            if len(comments) >= max_comments:
                break
        return comments
    except Exception as e:
        print(f"Error fetching Instagram comments via Apify: {e}")
        return []
def extract_video_id(url):
    """Extract the video ID from a YouTube URL."""
    # Check if it's an Instagram URL first
    instagram_id, instagram_type = extract_instagram_info(url)
    if instagram_id:
        return instagram_id, instagram_type

    # Regular expressions to match various YouTube URL formats
    youtube_regex = (
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
    youtube_match = re.match(youtube_regex, url)
    if youtube_match:
        return youtube_match.group(6), 'youtube'

    return None, None


# Add this function after the extract_video_id function

def analyze_emoji_sentiment(text):
    """
    Analyze the sentiment of emojis in a text.
    If text contains only emojis, return sentiment directly without using VADER/transformers.
    """
    # Extract all emojis from the text
    emojis_list = [c for c in text if c in emoji.EMOJI_DATA]

    if not emojis_list:
        return 0  # Neutral if no emojis

    # Define positive and negative emoji sets
    positive_emoji_patterns = [
        # Smileys & positive emotions
        r'😀|😃|😄|😁|😆|😊|🙂|🙃|😉|😇|🥰|😍|🤩|☺️|😋|😸|😺|😻|😍',
        # Hearts
        r'❤️|🧡|💛|💚|💙|💜|🖤|💕|💞|💓|💗|💖|💘|💝|💟|♥️ |❤️',
        # Positive hand gestures
        r'👍|👏|🙌|🤝|👌|🤲|👐|🙏|🙌',
        # Other positive symbols
        r'🔥|✨|🌟|⭐|💯|🏆|🥇|🎉|🎊'
    ]

    negative_emoji_patterns = [
        # Negative emotions
        r'😠|😡|🤬|😤|😒|🙄|😑|😐|😕|☹️|🙁|😯|😦|😧|😮|😲|🥺|😢|😭|😱|😖|😣|😞|😓|😩|😫|🥱',
        # Negative gestures
        r'👎|🖕|✊|👊|🤛|🤜|💩',
        # Other negative symbols
        r'☠️|💀|👺|👹|👿|😈|🤮|🤢'
    ]

    # Combine all emojis in the text
    emoji_text = ''.join(emojis_list)

    # Check for positive and negative patterns
    positive_count = sum(len(re.findall(pattern, emoji_text)) for pattern in positive_emoji_patterns)
    negative_count = sum(len(re.findall(pattern, emoji_text)) for pattern in negative_emoji_patterns)

    # Check if the text consists only of emojis
    # Strip whitespace and check if all remaining characters are emojis
    stripped_text = text.strip()
    only_emojis = all(c in emoji.EMOJI_DATA or c.isspace() for c in stripped_text)

    if only_emojis:
        # If comment has only emojis, return direct sentiment result
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    else:
        # For mixed content (text + emojis), return the sentiment score to be combined with VADER/transformer
        if positive_count > negative_count:
            return min(0.5, 0.1 * positive_count)  # Cap at 0.5
        elif negative_count > positive_count:
            return max(-0.5, -0.1 * negative_count)  # Cap at -0.5
        else:
            return 0


# Function to get comments from a YouTube video
def get_youtube_comments(video_id, max_comments=100):
    """Fetch comments from a YouTube video using the YouTube Data API."""
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=API_KEY)
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        # Get comments for the video
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_comments - len(comments)),
            pageToken=next_page_token
        )

        try:
            response = request.execute()
        except Exception as e:
            print(f"Error fetching comments: {e}")
            break

        # Extract comments from the response
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        # Check if there are more comments to fetch
        next_page_token = response.get("nextPageToken")
        if not next_page_token or len(comments) >= max_comments:
            break

    return comments

# Initialize the transformer model (this will be done once)
def initialize_transformer_model():
    """Initialize the transformer model for sentiment analysis."""
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Global variables for the transformer model
transformer_tokenizer, transformer_model = None, None

# Function to analyze sentiment using transformer model
def analyze_transformer_sentiment(comment):
    """Analyze sentiment of a comment using a transformer model."""
    global transformer_tokenizer, transformer_model

    # Initialize the model if not already done
    if transformer_tokenizer is None or transformer_model is None:
        transformer_tokenizer, transformer_model = initialize_transformer_model()

    # Tokenize the comment
    inputs = transformer_tokenizer(comment, return_tensors="pt", truncation=True, max_length=512)

    # Get the model output
    with torch.no_grad():
        outputs = transformer_model(**inputs)

    # Get the predicted class (0: negative, 1: positive)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()

    # Get the confidence score
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    confidence = scores[0][predicted_class].item()

    # Map to sentiment categories
    if predicted_class == 1:
        category = "positive"
        score = confidence
    else:
        category = "negative"
        score = -confidence

    return category, score

# Function to perform combined sentiment analysis
# Replace your existing analyze_sentiment function with this:

def analyze_sentiment(comments):
    """Analyze sentiment of a list of comments using VADER, Transformer models, and emoji analysis."""
    analyzer = SentimentIntensityAnalyzer()
    results = []
    for comment in comments:
        # First check if comment contains only emojis
        stripped_comment = comment.strip()
        emoji_list = [c for c in comment if c in emoji.EMOJI_DATA]
        only_emojis = all(c in emoji.EMOJI_DATA or c.isspace() for c in stripped_comment) and len(emoji_list) > 0

        if only_emojis:
            # For emoji-only comments, use direct emoji sentiment analysis
            emoji_result = analyze_emoji_sentiment(comment)
            category = emoji_result  # Will be "positive", "negative", or "neutral"

            # Set placeholder scores for consistency in results
            combined_score = 1.0 if category == "positive" else (-1.0 if category == "negative" else 0.0)
            vader_compound = 0
            transformer_score = 0
            emoji_score = combined_score
        else:
            # VADER sentiment analysis
            vader_sentiment = analyzer.polarity_scores(comment)
            vader_compound = vader_sentiment["compound"]

            # Transformer sentiment analysis
            transformer_category, transformer_score = analyze_transformer_sentiment(comment)

            # Emoji sentiment analysis (will return numerical score for mixed content)
            emoji_score = analyze_emoji_sentiment(comment)

            # Combine the scores (weighted average)
            # Adjusting weights: VADER 35%, Transformer 50%, Emoji 15%
            combined_score = (0.35 * vader_compound) + (0.50 * transformer_score) + (0.15 * emoji_score)

            # Classify sentiment based on combined score
            if combined_score >= 0.05:
                category = "positive"
            elif combined_score <= -0.05:
                category = "negative"
            else:
                category = "neutral"

        # Add the result to the list
        emoji_count = len(emoji_list)

        results.append({
            "comment": comment,
            "sentiment": category,
            "score": combined_score,
            "vader_score": vader_compound,
            "transformer_score": transformer_score,
            "emoji_score": emoji_score,
            "emoji_count": emoji_count
        })

    return results

# Function to summarize sentiment analysis results
def summarize_sentiment(results):
    """Summarize sentiment analysis results."""
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "positive_percent": 0,
            "negative_percent": 0,
            "neutral_percent": 0
        }

    positive = sum(1 for r in results if r["sentiment"] == "positive")
    negative = sum(1 for r in results if r["sentiment"] == "negative")
    neutral = sum(1 for r in results if r["sentiment"] == "neutral")

    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "positive_percent": round((positive / total) * 100, 2),
        "negative_percent": round((negative / total) * 100, 2),
        "neutral_percent": round((neutral / total) * 100, 2)
    }
home_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CommentAnaly - YouTube Comment Sentiment Analyzer</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #ff6b6b;
            --secondary-color: #4ecdc4;
            --dark-color: #292f36;
            --light-color: #f7fff7;
            --accent-color: #ffe66d;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background-color: #f5f5f5;
            color: var(--dark-color);
            line-height: 1.6;
        }

        .container {
            width: 100%;
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 15px;
        }

        header {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 1rem 0;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }

        nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
        }

        .logo {
            font-size: 1.8rem;
            font-weight: bold;
            display: flex;
            align-items: center;
        }

        .logo i {
            margin-right: 10px;
            color: var(--accent-color);
        }

        .nav-links {
            list-style: none;
            display: flex;
        }

        .nav-links li {
            margin-left: 30px;
        }

        .nav-links a {
            color: white;
            text-decoration: none;
            font-weight: 500;
            padding: 5px 10px;
            border-radius: 5px;
            transition: all 0.3s ease;
        }

        .nav-links a:hover {
            background-color: rgba(255, 255, 255, 0.2);
        }

        .hero {
            background-image: url('https://i.pinimg.com/originals/6a/72/4b/6a724b9761a94d99a3510cab9150d579.gif');
            background-size: cover;
            background-position: center;
            height: 70vh;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            position: relative;
            color: white;
        }

        .hero::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.7);
        }

        .hero-content {
            position: relative;
            z-index: 1;
            max-width: 800px;
            padding: 2rem;
        }

        .hero h1 {
            font-size: 3.5rem;
            margin-bottom: 1rem;
            font-weight: 700;
        }

        .hero p {
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }

        .btn {
            display: inline-block;
            background-color: var(--primary-color);
            color: white;
            padding: 12px 24px;
            border-radius: 50px;
            text-decoration: none;
            font-weight: bold;
            transition: all 0.3s ease;
            border: none;
            cursor: pointer;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
            background-color: #ff5252;
        }

        .features {
            padding: 5rem 0;
            background-color: white;
        }

        .section-title {
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 3rem;
            position: relative;
            padding-bottom: 15px;
        }

        .section-title::after {
            content: "";
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 100px;
            height: 4px;
            background: linear-gradient(to right, var(--primary-color), var(--secondary-color));
            border-radius: 2px;
        }

        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin-top: 2rem;
        }

        .feature-card {
            background-color: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
            text-align: center;
        }

        .feature-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.1);
        }

        .feature-icon {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            width: 70px;
            height: 70px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 20px;
            font-size: 28px;
        }

        .feature-card h3 {
            font-size: 1.5rem;
            margin-bottom: 15px;
            color: var(--dark-color);
        }

        .feature-card p {
            color: #666;
        }

        .cta {
            background: linear-gradient(135deg, var(--secondary-color), var(--primary-color));
            padding: 5rem 0;
            text-align: center;
            color: white;
        }

        .cta h2 {
            font-size: 2.5rem;
            margin-bottom: 20px;
        }

        .cta p {
            font-size: 1.2rem;
            margin-bottom: 30px;
            max-width: 700px;
            margin-left: auto;
            margin-right: auto;
        }

        .cta .btn {
            background-color: white;
            color: var(--primary-color);
        }

        .cta .btn:hover {
            background-color: var(--light-color);
        }

        footer {
            background-color: var(--dark-color);
            color: white;
            padding: 3rem 0;
        }

        .footer-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 30px;
        }

        .footer-section h3 {
            font-size: 1.5rem;
            margin-bottom: 20px;
            position: relative;
            padding-bottom: 10px;
        }

        .footer-section h3::after {
            content: "";
            position: absolute;
            bottom: 0;
            left: 0;
            width: 50px;
            height: 2px;
            background-color: var(--primary-color);
        }

        .footer-section p {
            margin-bottom: 10px;
        }

        .footer-links {
            list-style: none;
        }

        .footer-links li {
            margin-bottom: 10px;
        }

        .footer-links a {
            color: #ccc;
            text-decoration: none;
            transition: all 0.3s ease;
        }

        .footer-links a:hover {
            color: var(--primary-color);
            padding-left: 5px;
        }

        .social-links {
            display: flex;
            gap: 15px;
            margin-top: 20px;
        }

        .social-links a {
            color: white;
            background-color: rgba(255, 255, 255, 0.1);
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            transition: all 0.3s ease;
        }

        .social-links a:hover {
            background-color: var(--primary-color);
            transform: translateY(-3px);
        }

        .copyright {
            text-align: center;
            padding-top: 30px;
            margin-top: 30px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 0.9rem;
        }

        @media (max-width: 768px) {
            .hero h1 {
                font-size: 2.5rem;
            }

            .section-title {
                font-size: 2rem;
            }

            .nav-links {
                display: none;
            }
        }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <nav>
                <div class="logo">
                    <i class="fas fa-comment-dots"></i>
                    <span>CommentAnaly</span>
                </div>
                <ul class="nav-links">
    <li><a href="{{ url_for('home') }}">Home</a></li>
    {% if current_user.is_authenticated %}
        <li><a href="{{ url_for('dashboard') }}">Dashboard</a></li>
        <li><a href="{{ url_for('analyze') }}">Analyze</a></li>
        <li><a href="{{ url_for('logout') }}">Logout</a></li>
    {% else %}
        <li><a href="{{ url_for('login') }}">Login</a></li>
        <li><a href="{{ url_for('signup') }}">Sign Up</a></li>
    {% endif %}
    <li><a href="{{ url_for('about') }}">About</a></li>
    <li><a href="{{ url_for('contact') }}">Contact</a></li>
</ul>
            </nav>
        </div>
    </header>

    <section class="hero">
    <div class="hero-content">
        <h1>Decode Your Social Media Audience Sentiment</h1>
        <p>CommentAnaly uses advanced AI to analyze YouTube and Instagram comments, providing powerful insights into your audience's reactions and engagement.</p>
        <a href="/analyze" class="btn">Get Started Now</a>
    </div>
</section>

    <section class="features">
        <div class="container">
            <h2 class="section-title">Powerful Features</h2>
            <div class="features-grid">
                <div class="feature-card">
                    <div class="feature-icon">
                        <i class="fas fa-chart-pie"></i>
                    </div>
                    <h3>Sentiment Analysis</h3>
                    <p>Our AI-powered tool categorizes comments as positive, negative, or neutral using VADER sentiment analysis.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">
                        <i class="fas fa-cube"></i>
                    </div>
                    <h3>3D Visualization</h3>
                    <p>Explore your data with interactive 3D pie charts that make sentiment trends easy to understand.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">
                        <i class="fas fa-tachometer-alt"></i>
                    </div>
                    <h3>Real-time Analysis</h3>
                    <p>Get instant results for any YouTube video by simply pasting the URL into our analyzer.</p>
                </div>
                <div class="feature-card">
    <div class="feature-icon">
        <i class="fas fa-globe"></i>
    </div>
    <h3>Multi-Platform Support</h3>
    <p>Analyze comments from both YouTube videos and Instagram posts/reels with the same powerful sentiment analysis engine.</p>
</div>
            </div>
        </div>
    </section>

    <section class="cta">
        <div class="container">
            <h2>Ready to Understand Your Audience?</h2>
            <p>Stop guessing how your audience feels. Get data-driven insights about your YouTube comments and make informed content decisions.</p>
            <a href="/analyze" class="btn">Analyze Comments Now</a>
        </div>
    </section>

    <footer>
        <div class="container">
            <div class="footer-container">
                <div class="footer-section">
                    <h3>CommentAnaly</h3>
                    <p>Advanced YouTube comment sentiment analysis tool powered by AI to help content creators understand their audience better.</p>
                    <div class="social-links">
                        <a href="#"><i class="fab fa-twitter"></i></a>
                        <a href="#"><i class="fab fa-facebook-f"></i></a>
                        <a href="#"><i class="fab fa-instagram"></i></a>
                        <a href="#"><i class="fab fa-linkedin-in"></i></a>
                    </div>
                </div>
                <div class="footer-section">
                    <h3>Quick Links</h3>
                    <ul class="footer-links">
                        <li><a href="/">Home</a></li>
                        <li><a href="/analyze">Analyze</a></li>
                        <li><a href="/about">About</a></li>
                        <li><a href="/contact">Contact</a></li>
                    </ul>
                </div>
                <div class="footer-section">
                    <h3>Contact Us</h3>
                    <p><i class="fas fa-envelope"></i> info@commentanaly.com</p>
                    <p><i class="fas fa-phone"></i> +1 (555) 123-4567</p>
                    <p><i class="fas fa-map-marker-alt"></i> 123 Analytics St, Data City</p>
                </div>
            </div>
            <div class="copyright">
                <p>&copy; 2025 CommentAnaly. All rights reserved.</p>
            </div>
        </div>
    </footer>
</body>
</html>
"""

# Analyze page
analyze_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analyze Comments - CommentAnaly</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        /* Your existing styles */
        :root {
            --primary-color: #ff6b6b;
            --secondary-color: #4ecdc4;
            --dark-color: #292f36;
            --light-color: #f7fff7;
            --accent-color: #ffe66d;
            --positive-color: #4CAF50;
            --neutral-color: #2196F3;
            --negative-color: #F44336;
        }
        /* Rest of your CSS styles */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background-color: #f5f5f5;
            color: var(--dark-color);
            line-height: 1.6;
        }

        .container {
            width: 100%;
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 15px;
        }

        header {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 1rem 0;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }

        nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
        }

        .logo {
            font-size: 1.8rem;
            font-weight: bold;
            display: flex;
            align-items: center;
        }

        .logo i {
            margin-right: 10px;
            color: var(--accent-color);
        }

        .nav-links {
            list-style: none;
            display: flex;
        }

        .nav-links li {
            margin-left: 30px;
        }

        .nav-links a {
            color: white;
            text-decoration: none;
            font-weight: 500;
            padding: 5px 10px;
            border-radius: 5px;
            transition: all 0.3s ease;
        }

        .nav-links a:hover {
            background-color: rgba(255, 255, 255, 0.2);
        }

        .analyze-section {
            padding: 4rem 0;
        }

        .section-title {
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 3rem;
            position: relative;
            padding-bottom: 15px;
        }

        .section-title::after {
            content: "";
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 100px;
            height: 4px;
            background: linear-gradient(to right, var(--primary-color), var(--secondary-color));
            border-radius: 2px;
        }

        .url-form {
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
        }

        .form-group {
            margin-bottom: 20px;
        }

        .form-group label {
            display: block;
            margin-bottom: 10px;
            font-weight: 600;
            color: var(--dark-color);
        }

        .form-group input {
            width: 100%;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            transition: all 0.3s ease;
        }

        .form-group input:focus {
            outline: none;
            border-color: var(--secondary-color);
            box-shadow: 0 0 0 2px rgba(78, 205, 196, 0.2);
        }

        .form-group select {
            width: 100%;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            background-color: white;
            transition: all 0.3s ease;
        }

        .form-group select:focus {
            outline: none;
            border-color: var(--secondary-color);
            box-shadow: 0 0 0 2px rgba(78, 205, 196, 0.2);
        }

        .btn {
            display: inline-block;
            background-color: var(--primary-color);
            color: white;
            padding: 15px 30px;
            border-radius: 50px;
            text-decoration: none;
            font-weight: bold;
            transition: all 0.3s ease;
            border: none;
            cursor: pointer;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            font-size: 16px;
            width: 100%;
        }

        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
            background-color: #ff5252;
        }

        .results-section {
            padding: 4rem 0;
            display: none;
        }

        .results-container {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
            padding: 30px;
            margin-top: 30px;
        }

        .summary-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .stat-card {
            background-color: #f9f9f9;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
        }

        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.05);
        }

        .stat-card h3 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }

        .stat-card p {
            color: #666;
            font-size: 1rem;
        }

        .chart-container {
            height: 500px;
            margin-bottom: 30px;
        }

        .comments-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 30px;
        }

        .comments-table th, .comments-table td {
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }

        .comments-table th {
            background-color: #f9f9f9;
            font-weight: 600;
        }

        .comments-table tr:hover {
            background-color: #f9f9f9;
        }

        .sentiment-badge {
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            color: white;
            display: inline-block;
        }

        .positive {
            background-color: var(--positive-color);
        }

        .neutral {
            background-color: var(--neutral-color);
        }

        .negative {
            background-color: var(--negative-color);
        }

        .loader {
            border: 4px solid #f3f3f3;
            border-top: 4px solid var(--primary-color);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
            display: none;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        footer {
            background-color: var(--dark-color);
            color: white;
            padding: 3rem 0;
            margin-top: 4rem;
        }

        .footer-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 30px;
        }

        .footer-section h3 {
            font-size: 1.5rem;
            margin-bottom: 20px;
            position: relative;
            padding-bottom: 10px;
        }

        .footer-section h3::after {
            content: "";
            position: absolute;
            bottom: 0;
            left: 0;
            width: 50px;
            height: 2px;
            background-color: var(--primary-color);
        }

        .footer-section p {
            margin-bottom: 10px;
        }

        .footer-links {
            list-style: none;
        }

        .footer-links li {
            margin-bottom: 10px;
        }

        .footer-links a {
            color: #ccc;
            text-decoration: none;
            transition: all 0.3s ease;
        }

        .footer-links a:hover {
            color: var(--primary-color);
            padding-left: 5px;
        }

        .social-links {
            display: flex;
            gap: 15px;
            margin-top: 20px;
        }

        .social-links a {
            color: white;
            background-color: rgba(255, 255, 255, 0.1);
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            transition: all 0.3s ease;
        }

        .social-links a:hover {
            background-color: var(--primary-color);
            transform: translateY(-3px);
        }

        .copyright {
            text-align: center;
            padding-top: 30px;
            margin-top: 30px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 0.9rem;
        }

        @media (max-width: 768px) {
            .section-title {
                font-size: 2rem;
            }

            .nav-links {
                display: none;
            }
        }

        /* Add these new styles for the AI model indicator */
        .ai-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            background-color: #9c27b0;
            color: white;
            font-size: 0.8rem;
            margin-left: 10px;
        }

        .score-details {
            font-size: 0.9rem;
            color: #666;
            margin-top: 5px;
        }

        .model-indicator {
            display: flex;
            justify-content: center;
            margin-bottom: 20px;
        }

        .model-badge {
            background-color: #e1f5fe;
            color: #0288d1;
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: 500;
            display: inline-flex;
            align-items: center;
        }

        .model-badge i {
            margin-right: 8px;
        }

        /* Add a spinner animation */
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        /* Add these CSS styles to your analyze_template's <style> section */
.emoji-badge {
    display: inline-block;
    padding: 3px 6px;
    border-radius: 12px;
    font-size: 0.8rem;
    margin-left: 8px;
    color: white;
}
.emoji-badge.positive {
    background-color: var(--positive-color);
}
.emoji-badge.negative {
    background-color: var(--negative-color);
}
.emoji-badge.neutral {
    background-color: var(--neutral-color);
}
    </style>
</head>
<body>
    <header>
        <div class="container">
            <nav>
                <div class="logo">
                    <i class="fas fa-comment-dots"></i>
                    <span>CommentAnaly</span>
                </div>
                <ul class="nav-links">
    <li><a href="{{ url_for('home') }}">Home</a></li>
    {% if current_user.is_authenticated %}
        <li><a href="{{ url_for('dashboard') }}">Dashboard</a></li>
        <li><a href="{{ url_for('analyze') }}">Analyze</a></li>
        <li><a href="{{ url_for('logout') }}">Logout</a></li>
    {% else %}
        <li><a href="{{ url_for('login') }}">Login</a></li>
        <li><a href="{{ url_for('signup') }}">Sign Up</a></li>
    {% endif %}
    <li><a href="{{ url_for('about') }}">About</a></li>
    <li><a href="{{ url_for('contact') }}">Contact</a></li>
</ul>
            </nav>
        </div>
    </header>
    <section class="analyze-section">
        <div class="container">
            <h2 class="section-title">YouTube Comment Analyzer</h2>
               <div class="url-form">
    <form id="analyzeForm" onsubmit="analyzeComments(event)">
        <div class="form-group">
            <label for="videoUrl">Enter YouTube or Instagram URL</label>
            <input type="text" id="videoUrl" name="videoUrl"
                   placeholder="Enter YouTube video, Instagram post or reel URL" required>
        </div>
        <div class="form-group">
            <label for="platform">Platform</label>
            <select id="platform" name="platform">
                <option value="auto" selected>Auto Detect</option>
                <option value="youtube">YouTube</option>
                <option value="instagram">Instagram</option>
            </select>
        </div>
        <div class="form-group">
            <label for="commentsCount">Number of Comments to Analyze</label>
            <select id="commentsCount" name="commentsCount">
                <option value="50">50 comments</option>
                <option value="100" selected>100 comments</option>
                <option value="200">200 comments</option>
                <option value="500">500 comments</option>
            </select>
        </div>
        <button type="submit" class="btn">Analyze Comments</button>
    </form>
    <div class="loader" id="loader"></div>
</div>
    </section>
    <section class="results-section" id="resultsSection">
        <div class="container">
            <h2 class="section-title">Analysis Results</h2>
            <div class="results-container">
                <div class="model-indicator">
                    <div class="model-badge">
                        <i class="fas fa-robot"></i> Hybrid AI Model (VADER + Transformer)
                    </div>
                </div>
                <div class="summary-stats" id="summaryStats">
                    <!-- Will be populated by JavaScript -->
                </div>
                <h3>Sentiment Distribution</h3>
                <div class="chart-container" id="sentimentChart">
                    <!-- Will be populated by JavaScript -->
                </div>
                <h3>Recent Comments Analysis</h3>
                <div class="table-container">
                    <table class="comments-table" id="commentsTable">
    <thead>
        <tr>
            <th>Comment</th>
            <th>Sentiment</th>
            <th>Combined Score</th>
        </tr>
    </thead>
    <tbody>
        <!-- Will be populated by JavaScript -->
    </tbody>
</table>
                </div>
            </div>
        </div>
    </section>
    <footer>
        <!-- Your existing footer content -->
    </footer>
    <script>
function analyzeComments(event) {
    event.preventDefault();
    
    // Show loader and hide results
    document.getElementById('loader').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';

    // Disable submit button and show spinner
    const submitButton = document.querySelector('#analyzeForm button');
    submitButton.disabled = true;
    submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';

    const videoUrl = document.getElementById('videoUrl').value;
    const commentsCount = document.getElementById('commentsCount').value;
    const platform = document.getElementById('platform').value;

    // Send the analysis request to the server
    fetch('/api/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            url: videoUrl,
            count: commentsCount,
            platform: platform
        }),
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => Promise.reject(err));
        }
        return response.json();
    })
    .then(data => {
        // Hide loader
        document.getElementById('loader').style.display = 'none';

        // Display results section
        document.getElementById('resultsSection').style.display = 'block';

        // Re-enable the submit button
        submitButton.disabled = false;
        submitButton.innerHTML = 'Analyze Comments';

        // Update platform indicator
        let platformIcon = data.platform === 'youtube' ? 'fa-youtube' : 'fa-instagram';
        let platformName = data.platform === 'youtube' ? 'YouTube' : 'Instagram';

        document.querySelector('.model-indicator').innerHTML = `
            <div class="model-badge">
                <i class="fab ${platformIcon}"></i> ${platformName} |
                <i class="fas fa-robot"></i> Hybrid AI Model (VADER + Transformer + Emoji)
            </div>
        `;

        // Update summary stats
        const summaryStats = document.getElementById('summaryStats');
        summaryStats.innerHTML = `
            <div class="stat-card">
                <h3>${data.summary.total}</h3>
                <p>Total Comments</p>
            </div>
            <div class="stat-card">
                <h3>${data.summary.positive_percent}%</h3>
                <p>Positive Comments</p>
            </div>
            <div class="stat-card">
                <h3>${data.summary.neutral_percent}%</h3>
                <p>Neutral Comments</p>
            </div>
            <div class="stat-card">
                <h3>${data.summary.negative_percent}%</h3>
                <p>Negative Comments</p>
            </div>
        `;

        // Create pie chart
        const chartData = [{
            type: 'pie',
            values: [data.summary.positive, data.summary.neutral, data.summary.negative],
            labels: ['Positive', 'Neutral', 'Negative'],
            textinfo: 'label+percent',
            textposition: 'outside',
            automargin: true,
            marker: {
                colors: ['#4CAF50', '#2196F3', '#F44336'],
                line: {
                    color: '#ffffff',
                    width: 2
                }
            },
            hole: 0.4
        }];
         const layout = {
            title: 'Comment Sentiment Distribution',
            height: 500,
            margin: { t: 30, l: 0, r: 0, b: 0 },
            showlegend: true,
            legend: {
                orientation: 'h',
                y: -0.1
            }
        };

        Plotly.newPlot('sentimentChart', chartData, layout, {responsive: true});

        // Update comments table
        const tableBody = document.querySelector('#commentsTable tbody');
        tableBody.innerHTML = '';
        data.results.slice(0, 20).forEach(result => {
            const row = document.createElement('tr');
            
            // Create emoji indicator
            let emojiIndicator = '';
            if (result.emoji_count > 0) {
                const emojiScoreClass = result.emoji_score > 0 ? 'positive' :
                                      (result.emoji_score < 0 ? 'negative' : 'neutral');
                emojiIndicator = `<span class="emoji-badge ${emojiScoreClass}">
                                 <i class="far fa-smile"></i> ${result.emoji_count}</span>`;
            }

            row.innerHTML = `
                <td>${result.comment.substring(0, 100)}${result.comment.length > 100 ? '...' : ''}
                    ${emojiIndicator}</td>
                <td><span class="sentiment-badge ${result.sentiment}">${result.sentiment}</span></td>
                <td>
                    ${result.score.toFixed(2)}
                    <div class="score-details">
                        VADER: ${result.vader_score.toFixed(2)} |
                        Transformer: ${result.transformer_score.toFixed(2)} |
                        Emoji: ${result.emoji_score.toFixed(2)}
                     </div>
                </td>
            `;
            tableBody.appendChild(row);
        });

        // Scroll to results
        document.getElementById('resultsSection').scrollIntoView({
            behavior: 'smooth'
        });
    })
    .catch(error => {
        // Hide loader
        document.getElementById('loader').style.display = 'none';
        
        // Re-enable submit button
        submitButton.disabled = false;
        submitButton.innerHTML = 'Analyze Comments';

        // Show error message
        alert(error.error || 'Error analyzing comments. Please try again.');
    });
}
</script>   

</body>
</html>
"""
# About page - Complete Template
about_template = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>About - CommentAnaly</title>
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">
<style>
:root {
    --primary-color: #ff6b6b;
    --secondary-color: #4ecdc4;
    --dark-color: #292f36;
    --light-color: #f7fff7;
    --accent-color: #ffe66d;
}
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
body {
    background-color: #f5f5f5;
    color: var(--dark-color);
    line-height: 1.6;
}
.container {
    width: 100%;
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 15px;
}
header {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    color: white;
    padding: 1rem 0;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}
nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
}
.logo {
    font-size: 1.8rem;
    font-weight: bold;
    display: flex;
    align-items: center;
}
.logo i {
    margin-right: 10px;
    color: var(--accent-color);
}
.nav-links {
    list-style: none;
    display: flex;
}
.nav-links li {
    margin-left: 30px;
}
.nav-links a {
    color: white;
    text-decoration: none;
    font-weight: 500;
    padding: 5px 10px;
    border-radius: 5px;
    transition: all 0.3s ease;
}
.nav-links a:hover {
    background-color: rgba(255, 255, 255, 0.2);
}
.page-header {
    background: linear-gradient(135deg, var(--secondary-color), var(--primary-color));
    color: white;
    padding: 3rem 0;
    text-align: center;
}
.page-header h1 {
    font-size: 3rem;
    margin-bottom: 1rem;
}
.page-header p {
    font-size: 1.2rem;
    max-width: 800px;
    margin: 0 auto;
}
.about-section {
    padding: 5rem 0;
    background-color: white;
}
.about-container {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 50px;
    align-items: center;
}
.about-content h2 {
    font-size: 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    padding-bottom: 15px;
}
.about-content h2::after {
    content: "";
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100px;
    height: 4px;
    background: linear-gradient(to right, var(--primary-color), var(--secondary-color));
    border-radius: 2px;
}
.about-content p {
    margin-bottom: 1.5rem;
    color: #555;
    font-size: 1.1rem;
}
.about-image img {
    width: 100%;
    border-radius: 10px;
    box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1);
}
.tech-section {
    padding: 5rem 0;
    background-color: #f9f9f9;
}
.section-title {
    text-align: center;
    font-size: 2.5rem;
    margin-bottom: 3rem;
    position: relative;
    padding-bottom: 15px;
}
.section-title::after {
    content: "";
    position: absolute;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 100px;
    height: 4px;
    background: linear-gradient(to right, var(--primary-color), var(--secondary-color));
    border-radius: 2px;
}
.tech-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 30px;
}
.tech-card {
    background-color: white;
    border-radius: 10px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
    transition: all 0.3s ease;
}
.tech-card:hover {
    transform: translateY(-10px);
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.1);
}
.tech-icon {
    font-size: 3rem;
    margin-bottom: 20px;
    color: var(--primary-color);
}
.tech-card h3 {
    font-size: 1.5rem;
    margin-bottom: 15px;
}
.tech-card p {
    color: #666;
}
.team-section {
    padding: 5rem 0;
    background-color: white;
}
.team-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 30px;
}
.team-member {
    background-color: #f9f9f9;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
    transition: all 0.3s ease;
}
.team-member:hover {
    transform: translateY(-10px);
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.1);
}
.member-img {
    width: 100%;
    height: 250px;
    object-fit: cover;
}
.member-info {
    padding: 20px;
    text-align: center;
}
.member-info h3 {
    font-size: 1.5rem;
    margin-bottom: 5px;
}
.member-info p {
    color: #666;
    margin-bottom: 15px;
}
.social-links {
    display: flex;
    justify-content: center;
    gap: 15px;
}
.social-links a {
    color: var(--dark-color);
    background-color: rgba(0, 0, 0, 0.05);
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    transition: all 0.3s ease;
}
.social-links a:hover {
    background-color: var(--primary-color);
    color: white;
    transform: translateY(-3px);
}
footer {
    background-color: var(--dark-color);
    color: white;
    padding: 3rem 0;
}
.footer-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 30px;
}
.footer-section h3 {
    font-size: 1.5rem;
    margin-bottom: 20px;
    position: relative;
    padding-bottom: 10px;
}
.footer-section h3::after {
    content: "";
    position: absolute;
    bottom: 0;
    left: 0;
    width: 50px;
    height: 2px;
    background-color: var(--primary-color);
}
.footer-section p {
    margin-bottom: 10px;
}
.footer-links {
    list-style: none;
}
.footer-links li {
    margin-bottom: 10px;
}
.footer-links a {
    color: #ccc;
    text-decoration: none;
    transition: all 0.3s ease;
}
.footer-links a:hover {
    color: var(--primary-color);
    padding-left: 5px;
}
.social-links-footer {
    display: flex;
    gap: 15px;
    margin-top: 20px;
}
.social-links-footer a {
    color: white;
    background-color: rgba(255, 255, 255, 0.1);
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    transition: all 0.3s ease;
}
.social-links-footer a:hover {
    background-color: var(--primary-color);
    transform: translateY(-3px);
}
.copyright {
    text-align: center;
    padding-top: 30px;
    margin-top: 30px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    font-size: 0.9rem;
}
@media (max-width: 768px) {
    .about-container {
        grid-template-columns: 1fr;
    }
    .about-image {
        order: -1;
    }
    .section-title {
        font-size: 2rem;
    }
    .nav-links {
        display: none;
    }
}
</style>
</head>
<body>
    <header>
        <div class="container">
            <nav>
                <div class="logo">
                    <i class="fas fa-comment-dots"></i>
                    <span>CommentAnaly</span>
                </div>
                <ul class="nav-links">
    <li><a href="{{ url_for('home') }}">Home</a></li>
    {% if current_user.is_authenticated %}
        <li><a href="{{ url_for('dashboard') }}">Dashboard</a></li>
        <li><a href="{{ url_for('analyze') }}">Analyze</a></li>
        <li><a href="{{ url_for('logout') }}">Logout</a></li>
    {% else %}
        <li><a href="{{ url_for('login') }}">Login</a></li>
        <li><a href="{{ url_for('signup') }}">Sign Up</a></li>
    {% endif %}
    <li><a href="{{ url_for('about') }}">About</a></li>
    <li><a href="{{ url_for('contact') }}">Contact</a></li>
</ul>
            </nav>
        </div>
    </header>

    <section class="page-header">
        <div class="container">
            <h1>About CommentAnaly</h1>
            <p>Learn more about our mission, technology, and the team behind this powerful YouTube comment analysis tool.</p>
        </div>
    </section>

    <section class="about-section">
        <div class="container">
            <div class="about-container">
                <div class="about-content">
                    <h2>Our Story</h2>
                    <p>CommentAnaly was created to help content creators and marketers understand audience sentiment in a data-driven way. In today's digital landscape, understanding how your audience feels about your content is crucial for success.</p>
                    <p>Founded in 2025, our mission is to provide powerful yet accessible AI-powered analytics that help creators make better content decisions based on audience feedback. We believe that understanding sentiment patterns can transform how creators engage with their audiences.</p>
                    <p>Our platform uses cutting-edge natural language processing technologies to analyze YouTube comments at scale, providing actionable insights that go beyond simple metrics.</p>
                </div>
                <div class="about-image">
                    <img src="/api/placeholder/600/400" alt="CommentAnaly Dashboard">
                </div>
            </div>
        </div>
    </section>

    <section class="tech-section">
        <div class="container">
            <h2 class="section-title">Our Technology</h2>
            <div class="tech-grid">
                <div class="tech-card">
                    <div class="tech-icon">
                        <i class="fas fa-brain"></i>
                    </div>
                    <h3>VADER Sentiment</h3>
                    <p>We use the VADER (Valence Aware Dictionary and sEntiment Reasoner) model, specifically designed for social media content analysis.</p>
                </div>
                <div class="tech-card">
                    <div class="tech-icon">
                        <i class="fas fa-chart-pie"></i>
                    </div>
                    <h3>Interactive Visualization</h3>
                    <p>Our custom 3D visualizations make it easy to understand sentiment patterns at a glance.</p>
                </div>
                <div class="tech-card">
                    <div class="tech-icon">
                        <i class="fas fa-rocket"></i>
                    </div>
                    <h3>Real-time Processing</h3>
                    <p>Get instant analysis of hundreds of comments with our optimized data processing pipeline.</p>
                </div>
                <div class="tech-card">
                    <div class="tech-icon">
                        <i class="fas fa-cloud"></i>
                    </div>
                    <h3>Cloud Architecture</h3>
                    <p>Built on modern cloud infrastructure to ensure reliability and scalability for all users.</p>
                </div>
            </div>
        </div>
    </section>

    <section class="team-section">
        <div class="container">
            <h2 class="section-title">Our Team</h2>
            <div class="team-grid">
                <div class="team-member">
                    <img class="member-img" src="/api/placeholder/400/300" alt="Team Member">
                    <div class="member-info">
                        <h3>Alex Johnson</h3>
                        <p>Founder & Data Scientist</p>
                        <div class="social-links">
                            <a href="#"><i class="fab fa-twitter"></i></a>
                            <a href="#"><i class="fab fa-linkedin-in"></i></a>
                            <a href="#"><i class="fab fa-github"></i></a>
                        </div>
                    </div>
                </div>
                <div class="team-member">
                    <img class="member-img" src="/api/placeholder/400/300" alt="Team Member">
                    <div class="member-info">
                        <h3>Sam Taylor</h3>
                        <p>Lead Developer</p>
                        <div class="social-links">
                            <a href="#"><i class="fab fa-twitter"></i></a>
                            <a href="#"><i class="fab fa-linkedin-in"></i></a>
                            <a href="#"><i class="fab fa-github"></i></a>
                        </div>
                    </div>
                </div>
                <div class="team-member">
                    <img class="member-img" src="/api/placeholder/400/300" alt="Team Member">
                    <div class="member-info">
                        <h3>Morgan Chen</h3>
                        <p>UI/UX Designer</p>
                        <div class="social-links">
                            <a href="#"><i class="fab fa-twitter"></i></a>
                            <a href="#"><i class="fab fa-linkedin-in"></i></a>
                            <a href="#"><i class="fab fa-dribbble"></i></a>
                        </div>
                    </div>
                </div>
                <div class="team-member">
                    <img class="member-img" src="/api/placeholder/400/300" alt="Team Member">
                    <div class="member-info">
                        <h3>Jordan Riley</h3>
                        <p>ML Engineer</p>
                        <div class="social-links">
                            <a href="#"><i class="fab fa-twitter"></i></a>
                            <a href="#"><i class="fab fa-linkedin-in"></i></a>
                            <a href="#"><i class="fab fa-github"></i></a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <footer>
        <div class="container">
            <div class="footer-container">
                <div class="footer-section">
                    <h3>CommentAnaly</h3>
                    <p>Advanced YouTube comment sentiment analysis tool powered by AI to help content creators understand their audience better.</p>
                    <div class="social-links-footer">
                        <a href="#"><i class="fab fa-twitter"></i></a>
                        <a href="#"><i class="fab fa-facebook-f"></i></a>
                        <a href="#"><i class="fab fa-instagram"></i></a>
                        <a href="#"><i class="fab fa-linkedin-in"></i></a>
                    </div>
                </div>
                <div class="footer-section">
                    <h3>Quick Links</h3>
                    <ul class="footer-links">
                        <li><a href="/">Home</a></li>
                        <li><a href="/analyze">Analyze</a></li>
                        <li><a href="/about">About</a></li>
                        <li><a href="/contact">Contact</a></li>
                    </ul>
                </div>
                <div class="footer-section">
                    <h3>Contact Us</h3>
                    <p><i class="fas fa-envelope"></i> info@commentanaly.com</p>
                    <p><i class="fas fa-phone"></i> +1 (555) 123-4567</p>
                    <p><i class="fas fa-map-marker-alt"></i> 123 Analytics St, Data City</p>
                </div>
            </div>
            <div class="copyright">
                <p>&copy; 2025 CommentAnaly. All rights reserved.</p>
            </div>
        </div>
    </footer>
</body>
</html>"""        


# Keep all your existing analysis functions the same
# Now let's add the new templates for login, signup, and dashboard

# Login template
login_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - CommentAnaly</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #ff6b6b;
            --secondary-color: #4ecdc4;
            --dark-color: #292f36;
            --light-color: #f7fff7;
            --accent-color: #ffe66d;
        }
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        body {
            background-color: #f5f5f5;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .auth-container {
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem;
        }
        .auth-box {
            background: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            width: 100%;
            max-width: 400px;
        }
        .auth-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .auth-header h1 {
            color: var(--dark-color);
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        .form-group {
            margin-bottom: 1.5rem;
        }
        .form-group label {
            display: block;
            margin-bottom: 0.5rem;
            color: var(--dark-color);
            font-weight: 500;
        }
        .form-group input {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 1rem;
        }
        .form-group input:focus {
            outline: none;
            border-color: var(--secondary-color);
            box-shadow: 0 0 0 2px rgba(78,205,196,0.1);
        }
        .btn {
            width: 100%;
            padding: 0.75rem;
            border: none;
            border-radius: 5px;
            background: var(--primary-color);
            color: white;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .btn:hover {
            background: #ff5252;
            transform: translateY(-1px);
        }
        .auth-footer {
            text-align: center;
            margin-top: 1.5rem;
        }
        .auth-footer a {
            color: var(--primary-color);
            text-decoration: none;
        }
        .auth-footer a:hover {
            text-decoration: underline;
        }
        .flash-messages {
            margin-bottom: 1rem;
        }
        .flash-message {
            padding: 0.75rem;
            border-radius: 5px;
            margin-bottom: 0.5rem;
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .flash-message.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
    </style>
</head>
<body>
    <div class="auth-container">
        <div class="auth-box">
            <div class="auth-header">
                <h1>Login to CommentAnaly</h1>
                <p>Welcome back! Please login to continue.</p>
            </div>
            {% with messages = get_flashed_messages() %}
                {% if messages %}
                    <div class="flash-messages">
                        {% for message in messages %}
                            <div class="flash-message">{{ message }}</div>
                        {% endfor %}
                    </div>
                {% endif %}
            {% endwith %}
            <form method="POST" action="{{ url_for('login') }}">
                <div class="form-group">
                    <label for="email">Email</label>
                    <input type="email" id="email" name="email" required>
                </div>
                <div class="form-group">
                    <label for="password">Password</label>
                    <input type="password" id="password" name="password" required>
                </div>
                <button type="submit" class="btn">Login</button>
            </form>
            <div class="auth-footer">
                <p>Don't have an account? <a href="{{ url_for('signup') }}">Sign up</a></p>
            </div>
        </div>
    </div>
</body>
</html>
"""

# Signup template
signup_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign Up - CommentAnaly</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">
    <style>
        /* Same CSS as login template */
        :root {
            --primary-color: #ff6b6b;
            --secondary-color: #4ecdc4;
            --dark-color: #292f36;
            --light-color: #f7fff7;
            --accent-color: #ffe66d;
        }
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        body {
            background-color: #f5f5f5;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .auth-container {
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem;
        }
        .auth-box {
            background: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            width: 100%;
            max-width: 400px;
        }
        .auth-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .auth-header h1 {
            color: var(--dark-color);
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        .form-group {
            margin-bottom: 1.5rem;
        }
        .form-group label {
            display: block;
            margin-bottom: 0.5rem;
            color: var(--dark-color);
            font-weight: 500;
        }
        .form-group input {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 1rem;
        }
        .form-group input:focus {
            outline: none;
            border-color: var(--secondary-color);
            box-shadow: 0 0 0 2px rgba(78,205,196,0.1);
        }
        .btn {
            width: 100%;
            padding: 0.75rem;
            border: none;
            border-radius: 5px;
            background: var(--primary-color);
            color: white;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .btn:hover {
            background: #ff5252;
            transform: translateY(-1px);
        }
        .auth-footer {
            text-align: center;
            margin-top: 1.5rem;
        }
        .auth-footer a {
            color: var(--primary-color);
            text-decoration: none;
        }
        .auth-footer a:hover {
            text-decoration: underline;
        }
        .flash-messages {
            margin-bottom: 1rem;
        }
        .flash-message {
            padding: 0.75rem;
            border-radius: 5px;
            margin-bottom: 0.5rem;
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
    </style>
</head>
<body>
    <div class="auth-container">
        <div class="auth-box">
            <div class="auth-header">
                <h1>Create Account</h1>
                <p>Join CommentAnaly to analyze your social media comments.</p>
            </div>
            {% with messages = get_flashed_messages() %}
                {% if messages %}
                    <div class="flash-messages">
                        {% for message in messages %}
                            <div class="flash-message">{{ message }}</div>
                        {% endfor %}
                    </div>
                {% endif %}
            {% endwith %}
            <form method="POST" action="{{ url_for('signup') }}">
                <div class="form-group">
                    <label for="name">Full Name</label>
                    <input type="text" id="name" name="name" required>
                </div>
                <div class="form-group">
                    <label for="email">Email</label>
                    <input type="email" id="email" name="email" required>
                </div>
                <div class="form-group">
                    <label for="password">Password</label>
                    <input type="password" id="password" name="password" required>
                </div>
                <button type="submit" class="btn">Sign Up</button>
            </form>
            <div class="auth-footer">
                <p>Already have an account? <a href="{{ url_for('login') }}">Login</a></p>
            </div>
        </div>
    </div>
</body>
</html>
"""
# Dashboard template
dashboard_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard - CommentAnaly</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {
            --primary-color: #ff6b6b;
            --secondary-color: #4ecdc4;
            --dark-color: #292f36;
            --light-color: #f7fff7;
            --accent-color: #ffe66d;
        }
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        body {
            background-color: #f5f5f5;
            color: var(--dark-color);
            line-height: 1.6;
        }
        .dashboard-header {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 2rem 0;
        }
        .container {
            width: 100%;
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 15px;
        }
        .dashboard-nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
        }
        .user-info {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .user-avatar {
            width: 50px;
            height: 50px;
            background: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            color: var(--primary-color);
        }
        .nav-links {
            display: flex;
            gap: 1rem;
        }
        .nav-links a {
            color: white;
            text-decoration: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            transition: all 0.3s ease;
        }
        .nav-links a:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        .dashboard-content {
            padding: 2rem 0;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        .stat-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stat-card h3 {
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
            color: var(--primary-color);
        }
        .stat-card p {
            color: #666;
        }
        .recent-analyses {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .recent-analyses h2 {
            margin-bottom: 1.5rem;
            color: var(--dark-color);
        }
        .analysis-table {
            width: 100%;
            border-collapse: collapse;
        }
        .analysis-table th,
        .analysis-table td {
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        .analysis-table th {
            background: #f9f9f9;
            font-weight: 600;
        }
        .analysis-table tr:hover {
            background: #f9f9f9;
        }
        .platform-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 500;
        }
        .youtube {
            background: #ff0000;
            color: white;
        }
        .instagram {
            background: #e1306c;
            color: white;
        }
        .sentiment-chart {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-top: 2rem;
        }
        .btn {
            display: inline-block;
            padding: 0.75rem 1.5rem;
            border-radius: 5px;
            text-decoration: none;
            color: white;
            background: var(--primary-color);
            transition: all 0.3s ease;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .logout-btn {
            background: transparent;
            border: 2px solid white;
        }
        .logout-btn:hover {
            background: rgba(255, 255, 255, 0.1);
        }
    </style>
</head>
<body>
    <header class="dashboard-header">
        <div class="container">
            <div class="dashboard-nav">
                <div class="user-info">
                    <div class="user-avatar">
                        <i class="fas fa-user"></i>
                    </div>
                    <div>
                        <h2>Welcome, {{ current_user.name }}!</h2>
                        <p>Member since {{ current_user.join_date.strftime('%B %Y') }}</p>
                    </div>
                </div>
                <div class="nav-links">
                    <a href="{{ url_for('analyze') }}" class="btn">New Analysis</a>
                    <a href="{{ url_for('logout') }}" class="btn logout-btn">Logout</a>
                </div>
            </div>
        </div>
    </header>

    <main class="dashboard-content">
        <div class="container">
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>{{ total_analyses }}</h3>
                    <p>Total Analyses</p>
                </div>
                <div class="stat-card">
                    <h3>{{ youtube_analyses }}</h3>
                    <p>YouTube Analyses</p>
                </div>
                <div class="stat-card">
                    <h3>{{ instagram_analyses }}</h3>
                    <p>Instagram Analyses</p>
                </div>
                <div class="stat-card">
                    <h3>{{ avg_sentiment }}%</h3>
                    <p>Average Positive Sentiment</p>
                </div>
            </div>

            <div class="recent-analyses">
                <h2>Recent Analyses</h2>
                <table class="analysis-table">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Platform</th>
                            <th>URL</th>
                            <th>Comments</th>
                            <th>Sentiment</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for analysis in recent_analyses %}
                        <tr>
                            <td>{{ analysis.date_analyzed.strftime('%Y-%m-%d %H:%M') }}</td>
                            <td>
                                <span class="platform-badge {{ analysis.platform.lower() }}">
                                    {{ analysis.platform }}
                                </span>
                            </td>
                            <td>
                                <a href="{{ analysis.url }}" target="_blank">
                                    {{ analysis.url[:50] }}...
                                </a>
                            </td>
                            <td>{{ analysis.total_comments }}</td>
                            <td>
                                <div>Positive: {{ "%.1f"|format(analysis.positive_percent) }}%</div>
                                <div>Neutral: {{ "%.1f"|format(analysis.neutral_percent) }}%</div>
                                <div>Negative: {{ "%.1f"|format(analysis.negative_percent) }}%</div>
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>

            <div class="sentiment-chart">
                <h2>Sentiment Trends</h2>
                <div id="sentimentTrend"></div>
            </div>
        </div>
    </main>

    <script>
        // Sample data for sentiment trend chart
        const dates = {{ dates|tojson }};
        const positiveData = {{ positive_trends|tojson }};
        const neutralData = {{ neutral_trends|tojson }};
        const negativeData = {{ negative_trends|tojson }};

        const trace1 = {
            x: dates,
            y: positiveData,
            name: 'Positive',
            type: 'scatter',
            line: {color: '#4CAF50'}
        };

        const trace2 = {
            x: dates,
            y: neutralData,
            name: 'Neutral',
            type: 'scatter',
            line: {color: '#2196F3'}
        };

        const trace3 = {
            x: dates,
            y: negativeData,
            name: 'Negative',
            type: 'scatter',
            line: {color: '#F44336'}
        };

        const layout = {
            title: 'Sentiment Analysis Trends',
            xaxis: {title: 'Date'},
            yaxis: {title: 'Percentage'},
            height: 400,
            margin: {t: 30}
        };

        Plotly.newPlot('sentimentTrend', [trace1, trace2, trace3], layout);
    </script>
</body>
</html>
"""

# Updated routes with dashboard
@app.route('/')
def home():
    """Render the home page."""
    return render_template_string(home_template)

@app.route('/analyze')
@login_required
def analyze():
    """Render the analyze page."""
    return render_template_string(analyze_template)

@app.route('/about')
def about():
    """Render the about page."""
    return render_template_string(about_template)

@app.route('/contact')
def contact():
    """Redirect to about page for now."""
    return render_template_string(about_template)

# Authentication routes
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists')
            return render_template_string(signup_template)
        
        new_user = User(name=name, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template_string(signup_template)

@app.route('/login', methods=['GET', 'POST'])
def login():
    print("Login route accessed", file=sys.stderr)
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        try:
            email = request.form.get('email')
            password = request.form.get('password')
            print(f"Login attempt for email: {email}", file=sys.stderr)
            
            user = User.query.filter_by(email=email).first()
            if user and user.check_password(password):
                login_user(user)
                print(f"Login successful for user: {email}", file=sys.stderr)
                return redirect(url_for('dashboard'))
            else:
                print(f"Login failed for user: {email}", file=sys.stderr)
                flash('Invalid email or password')
                return render_template_string(login_template)
        except Exception as e:
            print(f"Login error: {str(e)}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            flash('An error occurred. Please try again.')
            return render_template_string(login_template)
    
    return render_template_string(login_template)

@app.route('/logout')
def logout():
    logout_user()
    flash('Logged out successfully.')
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Render the user dashboard."""
    # Get user's analysis history
    analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.date_analyzed.desc()).all()
    
    # Calculate statistics
    total_analyses = len(analyses)
    youtube_analyses = len([a for a in analyses if a.platform == 'youtube'])
    instagram_analyses = len([a for a in analyses if 'instagram' in a.platform])
    
    # Calculate average positive sentiment
    if total_analyses > 0:
        avg_sentiment = sum(a.positive_percent for a in analyses) / total_analyses
    else:
        avg_sentiment = 0
    
    # Get recent analyses
    recent_analyses = analyses[:10]  # Last 10 analyses
    
    # Prepare sentiment trends data
    dates = [a.date_analyzed.strftime('%Y-%m-%d') for a in reversed(analyses[-30:])]  # Last 30 days
    positive_trends = [a.positive_percent for a in reversed(analyses[-30:])]
    neutral_trends = [a.neutral_percent for a in reversed(analyses[-30:])]
    negative_trends = [a.negative_percent for a in reversed(analyses[-30:])]
    
    return render_template_string(dashboard_template,
        total_analyses=total_analyses,
        youtube_analyses=youtube_analyses,
        instagram_analyses=instagram_analyses,
        avg_sentiment=round(avg_sentiment, 1),
        recent_analyses=recent_analyses,
        dates=dates,
        positive_trends=positive_trends,
        neutral_trends=neutral_trends,
        negative_trends=negative_trends
    )
@app.route('/api/analyze', methods=['GET', 'POST'])
@login_required
def api_analyze():
    """API endpoint to analyze comments from YouTube or Instagram."""
    if request.method != 'POST':
        return jsonify({"error": "Method not allowed"}), 405

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        url = data.get('url')
        count = int(data.get('count', 100))

        if not url:
            return jsonify({"error": "No URL provided"}), 400

        # Extract ID and platform from URL
        content_id, platform = extract_video_id(url)

        if not content_id:
            return jsonify({"error": "Invalid URL. Please enter a valid YouTube or Instagram URL"}), 400

        # Get comments based on platform
        if platform == 'youtube':
            comments = get_youtube_comments(content_id, max_comments=count)
            platform_name = "YouTube"
        elif platform in ['instagram_post', 'instagram_reel']:
            comments = get_instagram_comments_apify(content_id, max_comments=count)
            platform_name = "Instagram"
        else:
            return jsonify({"error": "Unsupported content type"}), 400

        # Skip analysis if no comments found
        if len(comments) == 0:
            return jsonify({
                "error": f"No comments found for this {platform_name} content, or comments are disabled."
            }), 404

        # Analyze sentiment of comments
        results = analyze_sentiment(comments)
        summary = summarize_sentiment(results)

        # Save the analysis to database
        analysis = Analysis(
            user_id=current_user.id,
            url=url,
            platform=platform_name,
            total_comments=summary['total'],
            positive_percent=summary['positive_percent'],
            negative_percent=summary['negative_percent'],
            neutral_percent=summary['neutral_percent']
        )
        db.session.add(analysis)
        db.session.commit()

        return jsonify({
            "content_id": content_id,
            "platform": platform,
            "results": results,
            "summary": summary
        })

    except Exception as e:
        print(f"Analysis error: {str(e)}")  # Add this for debugging
        return jsonify({"error": str(e)}), 500   
def init_db():
    with app.app_context():
        try:
            # Create all tables
            db.create_all()
            print("Database tables created successfully", file=sys.stderr)
            
            # Check if admin user exists
            admin = User.query.filter_by(email='admin@example.com').first()
            if not admin:
                admin = User(name='Admin', email='admin@example.com')
                admin.set_password('admin-password-change-me')
                db.session.add(admin)
                db.session.commit()
                print("Admin user created successfully", file=sys.stderr)
        except Exception as e:
            print(f"Database initialization error: {str(e)}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)             


# Modified main execution block for PythonAnywhere
if __name__ == "__main__":
    # Initialize database
    with app.app_context():
        init_db()
        
        # Check if admin user exists
        admin = User.query.filter_by(email='admin@example.com').first()
        if not admin:
            admin = User(name='Admin', email='admin@example.com')
            admin.set_password('admin-password-change-me')
            db.session.add(admin)
            db.session.commit()
    
    # Initialize transformer model
    print("Initializing the transformer model...")
    transformer_tokenizer, transformer_model = initialize_transformer_model()
    print("Transformer model initialized successfully!")
    
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 10000))
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port) 
