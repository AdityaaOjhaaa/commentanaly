# CommentAnaly - Social Media Comment Sentiment Analyzer

A powerful web application that analyzes sentiment in YouTube and Instagram comments using advanced AI models.

## Features

- YouTube and Instagram comment analysis
- Multi-model sentiment analysis (VADER + Transformer + Emoji)
- User authentication and history tracking
- Interactive visualizations
- Responsive design

## Deployment on PythonAnywhere

1. Create a PythonAnywhere account at https://www.pythonanywhere.com

2. Upload files:
   - Upload all files to `/home/yourusername/commentanaly/`

3. Set up virtual environment:
   ```bash
   cd commentanaly
   mkvirtualenv --python=/usr/bin/python3.11 commentanaly-env
   pip install -r requirements.txt
   ```

4. Configure web app:
   - Add new web app
   - Choose "Manual configuration"
   - Python 3.11
   - Set source code: `/home/yourusername/commentanaly/app.py`
   - Set virtualenv: `/home/yourusername/.virtualenvs/commentanaly-env`

5. Update WSGI file:
   - Edit `/var/www/yourusername_pythonanywhere_com_wsgi.py`
   - Replace content with `wsgi.py` from this repository
   - Update paths and API keys

6. Initialize database:
   ```python
   from app import app, db
   with app.app_context():
       db.create_all()
   ```

7. Create admin user:
   ```python
   from app import User
   with app.app_context():
       admin = User(name='Admin', email='admin@example.com')
       admin.set_password('your-secure-password')
       db.session.add(admin)
       db.session.commit()
   ```

## Required Environment Variables

- `YOUTUBE_API_KEY`: Your YouTube Data API key
- `APIFY_API_TOKEN`: Your Apify API token

## Local Development

1. Clone the repository
2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Set environment variables
4. Run the application:
   ```bash
   python app.py
   ```

## License

MIT License - Feel free to use and modify for your own projects. 
