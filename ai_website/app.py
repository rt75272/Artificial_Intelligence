#!/usr/bin/env python3
"""AI Website.

A Flask based personal portfolio website showcasing artificial intelligence
projects, expertise, and interactive demonstrations.
"""
import os
import secrets
from flask import Flask, render_template, request
from werkzeug.exceptions import NotFound, InternalServerError

# Initialize Flask application.
app = Flask(__name__)

# Application configuration.
class Config:
    """Application configuration class."""
    # Security configuration.
    SECRET_KEY = os.environ.get('SECRET_KEY') or secrets.token_hex(16)
    # Development settings.
    DEBUG = os.environ.get('FLASK_ENV') == 'development'
    # Server configuration.
    HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
    PORT = int(os.environ.get('FLASK_PORT', 5000))

# Apply configuration to Flask app.
app.config.from_object(Config)

@app.route('/')
def home():
    """Render the home page with hero section and AI expertise overview.
    
    Returns:
        str: Rendered HTML template for the home page.
    """
    return render_template('home.html')

@app.route('/about')
def about():
    """Render the about page with personal background and mission.
    
    Returns:
        str: Rendered HTML template for the about page.
    """
    return render_template('about.html')

@app.route('/demos')
def demos():
    """Render the AI demos page for future interactive demonstrations.
    
    Returns:
        str: Rendered HTML template for the demos page.
    """
    return render_template('demos.html')

@app.errorhandler(404)
def page_not_found(error):
    """Handle 404 Not Found errors with custom error page.
    
    Args:
        error: The 404 error object.
        
    Returns:
        tuple: Rendered 404 template and HTTP status code.
    """
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(error):
    """Handle 500 Internal Server errors gracefully.
    
    Args:
        error: The 500 error object.
        
    Returns:
        tuple: Error message and HTTP status code.
    """
    return render_template('404.html'), 500

@app.context_processor
def inject_global_vars():
    """Inject global variables into all templates.
    
    Returns:
        dict: Dictionary of global template variables.
    """
    return {
        'current_year': 2025,
        'site_name': 'Ryan Thompson - AI Portfolio'
    }

def create_app():
    """Application factory pattern for creating Flask app instances.
    
    Returns:
        Flask: Configured Flask application instance.
    """
    return app

if __name__ == '__main__':
    # Run the application in development mode.
    # Note: Use a production WSGI server like Gunicorn for production deployment.
    app.run(
        debug=Config.DEBUG,
        host=Config.HOST,
        port=Config.PORT,
        threaded=True  # Enable threading for better performance.
    )