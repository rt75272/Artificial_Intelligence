#!/usr/bin/env python3
"""AI Website.

A Flask based personal portfolio website showcasing artificial intelligence
projects, expertise, and interactive demonstrations.
"""
import base64
import io
import os
import secrets
import logging
import traceback
import pickle
from typing import Optional
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import NotFound, InternalServerError
from PIL import Image

# Configure application logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)

# Global model variables for handwriting recognition
digit_model = None
scaler = None

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

def load_sklearn_model():
    """Load the pre-trained scikit-learn model and scaler from disk.
    
    Returns:
        tuple: (model, scaler) if successful, (None, None) if failed
    """
    try:
        logger.info("🔧 Loading scikit-learn digit recognition model...")
        model_path = 'sklearn_digit_model.pkl'
        scaler_path = 'sklearn_scaler.pkl'
        
        # Check if model file exists
        if not os.path.exists(model_path):
            logger.error(f"❌ Sklearn model file {model_path} not found!")
            return None, None
        
        # Load the trained model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load the data scaler
        scaler = None
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        
        logger.info(f"✅ Scikit-learn model loaded successfully")
        return model, scaler
        
    except Exception as e:
        logger.error(f"❌ Failed to load sklearn model: {e}")
        return None, None

def enhanced_preprocess(image_data: str) -> Optional[np.ndarray]:
    """Convert base64 image data to normalized numpy array for model prediction.
    
    Args:
        image_data: Base64 encoded image string from HTML5 canvas
        
    Returns:
        numpy.ndarray: Processed 784-element array ready for model input,
                      or None if processing fails
    """
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image data
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Find the bounding box of drawn content to crop efficiently
        bbox = image.getbbox()
        if bbox:
            left, top, right, bottom = bbox
            width = right - left
            height = bottom - top
            
            # Add padding (20% of larger dimension) to preserve aspect ratio
            padding = int(0.2 * max(width, height))
            left = max(0, left - padding)
            top = max(0, top - padding)
            right = min(image.width, right + padding)
            bottom = min(image.height, bottom + padding)
            image = image.crop((left, top, right, bottom))
        
        # Resize to MNIST standard 28x28 pixels
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array for mathematical operations
        img_array = np.array(image, dtype=np.float32)
        
        # Invert colors if needed (MNIST expects white digits on black background)
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
        
        # Normalize pixel values to [0, 1] range
        img_array = img_array / 255.0
        
        # Flatten to 784-element vector for scikit-learn input
        img_array = img_array.reshape(1, 784)
        
        return img_array
        
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        return None

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

@app.route('/references')
def references():
    """Render the references page with AI/ML write-ups and code examples.
    
    Returns:
        str: Rendered HTML template for the references page.
    """
    return render_template('references.html')

@app.route('/demos/handwriting')
def handwriting_demo():
    """Render the handwriting recognition demo page.
    
    Returns:
        str: Rendered HTML template for the handwriting recognition demo.
    """
    return render_template('handwriting.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict digit from canvas drawing using the trained model.
    
    Expected JSON input:
        {
            "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
        }
    
    Returns:
        JSON response with prediction results:
        {
            "prediction": 7,
            "confidence": "98.2%", 
            "probabilities": [0.001, 0.002, ..., 0.982, ...]
        }
    """
    try:
        # Verify model is loaded
        if digit_model is None:
            logger.error("❌ No trained model available")
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Extract image data from request
        data = request.json
        image_data = data.get('image')
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Preprocess the image for model input
        processed_image = enhanced_preprocess(image_data)
        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        # Apply data scaling if scaler is available
        if scaler is not None:
            processed_image = scaler.transform(processed_image)
        
        # Make prediction using the trained model
        predicted_digit = int(digit_model.predict(processed_image)[0])
        probabilities = digit_model.predict_proba(processed_image)[0]
        confidence = float(np.max(probabilities))
        
        logger.info(f"🎯 Prediction: {predicted_digit} (confidence: {confidence:.3f})")
        
        # Return prediction results as JSON
        return jsonify({
            'prediction': predicted_digit,
            'confidence': f"{confidence * 100:.1f}%",
            'probabilities': probabilities.tolist()
        })
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Prediction failed'}), 500

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

def initialize_models():
    """Initialize the digit recognition model and scaler."""
    global digit_model, scaler
    try:
        logger.info("🚀 Initializing handwriting recognition model...")
        digit_model, scaler = load_sklearn_model()
        if digit_model is not None:
            logger.info("✅ Handwriting model loaded successfully")
        else:
            logger.warning("⚠️ Handwriting model not available - predictions will not work")
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")

# Initialize models when module is loaded
initialize_models()

if __name__ == '__main__':
    # Run the application in development mode.
    # Note: Use a production WSGI server like Gunicorn for production deployment.
    app.run(
        debug=Config.DEBUG,
        host=Config.HOST,
        port=Config.PORT,
        threaded=True  # Enable threading for better performance.
    )