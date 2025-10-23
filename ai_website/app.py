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
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Configure application logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask application.
app = Flask(__name__)

# Global model variables for handwriting recognition.
digit_model = None
scaler = None
house_model = None
house_scaler = None
house_feature_names = ["sqft", "bedrooms", "bathrooms", "year_built"]

# Application configuration.
class Config:
    """Application configuration class."""
    # Security configuration.
    SECRET_KEY = os.environ.get('SECRET_KEY') or secrets.token_hex(16)
    # Development settings.
    DEBUG = os.environ.get('FLASK_ENV') == 'development'
    # Server configuration.
    HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
    # Prefer Render's PORT if present, then FLASK_PORT, then 5000.
    PORT = int(os.environ.get('PORT') or os.environ.get('FLASK_PORT', 5000))

# Apply configuration to Flask app.
app.config.from_object(Config)

def load_sklearn_model():
    """Load the pre-trained scikit-learn model and scaler from disk.
    Returns:
        tuple: (model, scaler) if successful, (None, None) if failed.
    """
    try:
        logger.info("🔧 Loading scikit-learn digit recognition model...")
        model_path = 'sklearn_digit_model.pkl'
        scaler_path = 'sklearn_scaler.pkl'
        # Check if model file exists.
        if not os.path.exists(model_path):
            logger.error(f"❌ Sklearn model file {model_path} not found!")
            return None, None
        # Load the trained model.
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        # Load the data scaler.
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
        image_data: Base64 encoded image string from HTML5 canvas.
    Returns:
        numpy.ndarray: Processed 784-element array ready for model input, or None if processing fails.
    """
    try:
        # Remove data URL prefix if present.
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        # Decode base64 image data.
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        # Convert to grayscale if needed.
        if image.mode != 'L':
            image = image.convert('L')
        # Find the bounding box of drawn content to crop efficiently.
        bbox = image.getbbox()
        if bbox:
            left, top, right, bottom = bbox
            width = right - left
            height = bottom - top
            # Add padding (20% of larger dimension) to preserve aspect ratio.
            padding = int(0.2 * max(width, height))
            left = max(0, left - padding)
            top = max(0, top - padding)
            right = min(image.width, right + padding)
            bottom = min(image.height, bottom + padding)
            image = image.crop((left, top, right, bottom))
        # Resize to MNIST standard 28x28 pixels.
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        # Convert to numpy array for mathematical operations.
        img_array = np.array(image, dtype=np.float32)
        # Invert colors if needed (MNIST expects white digits on black background).
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
        # Normalize pixel values to [0, 1] range.
        img_array = img_array / 255.0
        # Flatten to 784-element vector for scikit-learn input.
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

@app.route('/demos/linear-regression')
def linear_regression_demo():
    """Render the interactive linear regression demo page.
    Returns:
        str: Rendered HTML template for the linear regression demo.
    """
    return render_template('linear_regression_demo.html')

@app.route('/demos/housing-prices')
def housing_prices_demo():
    """Render the housing price prediction demo page.
    Returns:
        str: Rendered HTML template for the housing prices demo.
    """
    return render_template('housing_prices.html')

@app.route('/demos/kmeans')
def kmeans_demo():
    """Render the interactive K-Means clustering demo page.
    Returns:
        str: Rendered HTML template for the K-Means demo.
    """
    return render_template('kmeans_demo.html')

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
        }.
    """
    try:
        # Verify model is loaded.
        if digit_model is None:
            logger.error("❌ No trained model available")
            return jsonify({'error': 'Model not loaded'}), 500
        # Extract image data from request.
        data = request.json
        image_data = data.get('image')
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        # Preprocess the image for model input.
        processed_image = enhanced_preprocess(image_data)
        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 400
        # Apply data scaling if scaler is available.
        if scaler is not None:
            processed_image = scaler.transform(processed_image)
        # Make prediction using the trained model.
        predicted_digit = int(digit_model.predict(processed_image)[0])
        probabilities = digit_model.predict_proba(processed_image)[0]
        confidence = float(np.max(probabilities))
        logger.info(f"🎯 Prediction: {predicted_digit} (confidence: {confidence:.3f})")
        # Return prediction results as JSON.
        return jsonify({
            'prediction': predicted_digit,
            'confidence': f"{confidence * 100:.1f}%",
            'probabilities': probabilities.tolist()
        })
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/api/house-price', methods=['POST'])
def predict_house_price():
    """Predict housing price from provided features.
    Expected JSON input:
        {"sqft": 1800, "bedrooms": 3, "bathrooms": 2, "year_built": 2005}.
    Returns:
        JSON response with predicted price and echo of inputs.
    """
    try:
        if house_model is None or house_scaler is None:
            return jsonify({'error': 'Housing model not available.'}), 500
        data = request.get_json(silent=True) or {}
        try:
            sqft = float(data.get('sqft', 0))
            bedrooms = float(data.get('bedrooms', 0))
            bathrooms = float(data.get('bathrooms', 0))
            year_built = float(data.get('year_built', 0))
        except Exception:
            return jsonify({'error': 'Invalid input values.'}), 400
        X = np.array([[sqft, bedrooms, bathrooms, year_built]], dtype=float)
        Xs = house_scaler.transform(X)
        pred = float(house_model.predict(Xs)[0])
        pred_clamped = max(10000.0, pred)
        return jsonify({'prediction': round(pred_clamped, 2), 'inputs': {'sqft': sqft, 'bedrooms': bedrooms, 'bathrooms': bathrooms, 'year_built': year_built}})
    except Exception as e:
        logger.error(f"❌ House price prediction error: {e}")
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Prediction failed.'}), 500

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
    """Application factory; returns the Flask app instance."""
    return app

def initialize_models():
    """Initialize models used by the website (handwriting, housing)."""
    global digit_model, scaler, house_model, house_scaler
    try:
        # Initialize handwriting recognition model.
        logger.info("🚀 Initializing handwriting recognition model...")
        digit_model, scaler = load_sklearn_model()
        if digit_model is not None:
            logger.info("✅ Handwriting model loaded successfully")
        else:
            logger.warning("⚠️ Handwriting model not available - predictions will not work")
        # Initialize housing price prediction model from synthetic data.
        logger.info("🏠 Initializing housing price prediction model...")
        rng = np.random.default_rng(42)
        n_samples = 400
        sqft = rng.uniform(600, 3500, n_samples)
        bedrooms = rng.integers(1, 6, n_samples)
        bathrooms = rng.integers(1, 4, n_samples) + rng.choice([0, 0.5], n_samples)
        year_built = rng.integers(1960, 2025, n_samples)
        base = 50000
        price = base + 180 * sqft + 12000 * bedrooms + 15000 * bathrooms - 400 * (2025 - year_built) + rng.normal(0, 30000, n_samples)
        X = np.column_stack([sqft, bedrooms, bathrooms, year_built])
        house_scaler = StandardScaler()
        Xs = house_scaler.fit_transform(X)
        house_model = LinearRegression()
        house_model.fit(Xs, price)
        logger.info("✅ Housing model initialized in-memory with synthetic data.")
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")

# Initialize models when module is loaded.
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