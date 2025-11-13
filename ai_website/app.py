#!/usr/bin/env python3
"""Personal Portfolio Website.

A Flask-based personal website showcasing projects and technical expertise.
"""

import base64
import io
import logging
import os
import pickle
import secrets
import traceback
from typing import Optional

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Configure application logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask application.
app = Flask(__name__)

# Global model variables.
digit_model = None
scaler = None
house_model = None
house_scaler = None
house_feature_names = ["sqft", "bedrooms", "bathrooms", "year_built"]

# Application configuration.
class Config:
    """Application configuration settings."""
    # Security configuration.
    SECRET_KEY = os.environ.get('SECRET_KEY') or secrets.token_hex(16)
    # Development settings.
    DEBUG = os.environ.get('FLASK_ENV') == 'development'
    # Server configuration.
    HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
    # Use Render's PORT if present, otherwise use FLASK_PORT or default 5000.
    PORT = int(os.environ.get('PORT') or os.environ.get('FLASK_PORT', 5000))

# Apply configuration to Flask app.
app.config.from_object(Config)

def load_sklearn_model():
    """Load the pre-trained scikit-learn model and scaler from disk.

    Returns:
        tuple: (model, scaler) if successful, (None, None) if failed.
    """
    try:
        logger.info("Loading digit recognition model...")
        # Get absolute path to app directory.
        app_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(app_dir, 'sklearn_digit_model.pkl')
        scaler_path = os.path.join(app_dir, 'sklearn_scaler.pkl')
        # Check if model file exists.
        if not os.path.exists(model_path):
            logger.error(f"Model file {model_path} not found!")
            return None, None
        # Load the trained model.
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        # Load the data scaler.
        scaler = None
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        logger.info("Model loaded successfully")
        return model, scaler
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None

def enhanced_preprocess(image_data: str) -> Optional[np.ndarray]:
    """Convert base64 image data to normalized numpy array for model prediction.

    Args:
        image_data: Base64 encoded image string from HTML5 canvas.

    Returns:
        numpy.ndarray: Processed 784-element array ready for model input, 
                      or None if processing fails.
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
        # Find the bounding box of drawn content.
        bbox = image.getbbox()
        if bbox:
            left, top, right, bottom = bbox
            width = right - left
            height = bottom - top
            # Add padding to preserve aspect ratio.
            padding = int(0.2 * max(width, height))
            left = max(0, left - padding)
            top = max(0, top - padding)
            right = min(image.width, right + padding)
            bottom = min(image.height, bottom + padding)
            image = image.crop((left, top, right, bottom))
        # Resize to MNIST standard 28x28 pixels.
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        # Convert to numpy array.
        img_array = np.array(image, dtype=np.float32)
        # Invert colors if needed (MNIST expects white on black).
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
        # Normalize pixel values to [0, 1] range.
        img_array = img_array / 255.0
        # Flatten to 784-element vector.
        img_array = img_array.reshape(1, 784)
        return img_array
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return None

@app.route('/')
def home():
    """Render the home page.

    Returns:
        str: Rendered HTML template for the home page.
    """
    return render_template('home.html')

@app.route('/about')
def about():
    """Render the about page.

    Returns:
        str: Rendered HTML template for the about page.
    """
    return render_template('about.html')

@app.route('/projects')
def projects():
    """Render the projects page with demos and references.

    Returns:
        str: Rendered HTML template for the projects page.
    """
    return render_template('projects.html')

@app.route('/demos/handwriting')
def handwriting_demo():
    """Render the handwriting recognition demo page.

    Returns:
        str: Rendered HTML template for the handwriting demo.
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
    """Render the K-Means clustering demo page.

    Returns:
        str: Rendered HTML template for the K-Means demo.
    """
    return render_template('kmeans_demo.html')

@app.route('/demos/decision-tree')
def decision_tree_demo():
    """Render the decision tree classifier demo page.

    Returns:
        str: Rendered HTML template for the decision tree demo.
    """
    return render_template('decision_tree_demo.html')

@app.route('/demos/neural-network')
def neural_network_demo():
    """Render the neural network visualization demo page.

    Returns:
        str: Rendered HTML template for the neural network demo.
    """
    return render_template('neural_network_demo.html')

@app.route('/api/neural-network', methods=['POST'])
def predict_pattern():
    """Process drawn pattern and return neuron activations.

    Expected JSON input:
        {"pixels": [[0.1, 0.5, ...], ...]}.

    Returns:
        JSON response with layer activations and prediction.
    """
    try:
        data = request.get_json(silent=True) or {}
        pixels = data.get('pixels', [])
        if not pixels:
            return jsonify({'error': 'No pixel data provided'}), 400
        # Convert to numpy array.
        pixel_array = np.array(pixels)
        from PIL import Image as PILImage, ImageOps
        img = PILImage.fromarray(pixel_array.astype('uint8'))
        # Invert if needed so digit is white on black background.
        img_array = np.array(img)
        if np.mean(img_array) > 127:
            img = ImageOps.invert(img)
        # Find bounding box of drawn content.
        img_array = np.array(img)
        rows = np.any(img_array > 30, axis=1)
        cols = np.any(img_array > 30, axis=0)
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            # Crop to content with padding.
            pad = 10
            rmin = max(0, rmin - pad)
            rmax = min(img_array.shape[0], rmax + pad)
            cmin = max(0, cmin - pad)
            cmax = min(img_array.shape[1], cmax + pad)
            img = img.crop((cmin, rmin, cmax, rmax))
        # Resize to 20x20 maintaining aspect ratio.
        img.thumbnail((20, 20), PILImage.Resampling.LANCZOS)
        # Create 28x28 image with digit centered.
        final_img = PILImage.new('L', (28, 28), 0)
        offset_x = (28 - img.width) // 2
        offset_y = (28 - img.height) // 2
        final_img.paste(img, (offset_x, offset_y))
        # Convert to array and normalize.
        img_array = np.array(final_img)
        flat_pixels = img_array.flatten() / 255.0
        # Use trained model for actual prediction if available.
        if digit_model is not None:
            # Apply scaler if available.
            if scaler is not None:
                flat_pixels = scaler.transform([flat_pixels])[0]
            # Get prediction from trained model.
            prediction = digit_model.predict([flat_pixels])[0]
            # Get probability estimates if available.
            if hasattr(digit_model, 'predict_proba'):
                probabilities = digit_model.predict_proba([flat_pixels])[0]
            else:
                # Create one-hot style probabilities.
                probabilities = np.zeros(10)
                probabilities[prediction] = 1.0
        else:
            # Fallback if no model available.
            logger.warning("No trained model available, using random prediction")
            prediction = 0
            probabilities = np.random.dirichlet(np.ones(10))
        # Generate layer activations for visualization.
        input_size = len(flat_pixels)
        hidden1_size = 64
        hidden2_size = 32
        # Create synthetic layer activations based on input.
        np.random.seed(int(flat_pixels.sum() * 1000) % 2**32)
        w1 = np.random.randn(input_size, hidden1_size) * 0.1
        w2 = np.random.randn(hidden1_size, hidden2_size) * 0.1
        # Forward pass for visualization.
        h1 = np.maximum(0, np.dot(flat_pixels, w1))
        h2 = np.maximum(0, np.dot(h1, w2))
        # Sample neurons for visualization.
        layer1_activations = h1[:16].tolist()
        layer2_activations = h2[:16].tolist()
        output_activations = probabilities.tolist()
        return jsonify({
            'layer1': layer1_activations,
            'layer2': layer2_activations,
            'output': output_activations,
            'prediction': int(prediction)
        })
    except Exception as e:
        logger.error(f"Neural network prediction error: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/api/decision-tree', methods=['POST'])
def train_decision_tree():
    """Train a decision tree on provided data points.

    Expected JSON input:
        {"points": [{"x": 0.5, "y": 0.3, "class": 0}, ...]}.

    Returns:
        JSON response with decision boundary grid.
    """
    try:
        from sklearn.tree import DecisionTreeClassifier
        data = request.get_json(silent=True) or {}
        points = data.get('points', [])
        if len(points) < 2:
            return jsonify({'error': 'Need at least 2 points'}), 400
        # Extract features and labels.
        X = [[p['x'], p['y']] for p in points]
        y = [p['class'] for p in points]
        # Train decision tree.
        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
        clf.fit(X, y)
        # Generate prediction grid.
        grid_size = 30
        x_min, x_max = 0, 1
        y_min, y_max = 0, 1
        xx = np.linspace(x_min, x_max, grid_size)
        yy = np.linspace(y_min, y_max, grid_size)
        grid = []
        for yi in yy:
            row = []
            for xi in xx:
                pred = int(clf.predict([[xi, yi]])[0])
                row.append(pred)
            grid.append(row)
        return jsonify({'grid': grid, 'grid_size': grid_size})
    except Exception as e:
        logger.error(f"Decision tree error: {e}")
        return jsonify({'error': 'Training failed'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Predict digit from canvas drawing using the trained model.

    Expected JSON input:
        {"image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."}.

    Returns:
        JSON response with prediction results:
        {"prediction": 7, "confidence": "98.2%", "probabilities": [...]}.
    """
    try:
        # Verify model is loaded.
        if digit_model is None:
            logger.error("No trained model available")
            return jsonify({'error': 'Model not loaded'}), 500
        # Extract image data from request.
        data = request.json
        image_data = data.get('image')
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        # Preprocess the image.
        processed_image = enhanced_preprocess(image_data)
        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 400
        # Apply data scaling if scaler is available.
        if scaler is not None:
            processed_image = scaler.transform(processed_image)
        # Make prediction.
        predicted_digit = int(digit_model.predict(processed_image)[0])
        probabilities = digit_model.predict_proba(processed_image)[0]
        confidence = float(np.max(probabilities))
        logger.info(f"Prediction: {predicted_digit} (confidence: {confidence:.3f})")
        # Return prediction results.
        return jsonify({
            'prediction': predicted_digit,
            'confidence': f"{confidence * 100:.1f}%",
            'probabilities': probabilities.tolist()
        })
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/api/house-price', methods=['POST'])
def predict_house_price():
    """Predict housing price from provided features.

    Expected JSON input:
        {"sqft": 1800, "bedrooms": 3, "bathrooms": 2, "year_built": 2005}.

    Returns:
        JSON response with predicted price and inputs.
    """
    try:
        if house_model is None or house_scaler is None:
            return jsonify({'error': 'Housing model not available.'}), 500
        data = request.get_json(silent=True) or {}
        # Parse input values.
        try:
            sqft = float(data.get('sqft', 0))
            bedrooms = float(data.get('bedrooms', 0))
            bathrooms = float(data.get('bathrooms', 0))
            year_built = float(data.get('year_built', 0))
        except Exception:
            return jsonify({'error': 'Invalid input values.'}), 400
        # Prepare features and make prediction.
        X = np.array([[sqft, bedrooms, bathrooms, year_built]], dtype=float)
        Xs = house_scaler.transform(X)
        pred = float(house_model.predict(Xs)[0])
        pred_clamped = max(10000.0, pred)
        return jsonify({
            'prediction': round(pred_clamped, 2),
            'inputs': {
                'sqft': sqft,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'year_built': year_built
            }
        })
    except Exception as e:
        logger.error(f"House price prediction error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Prediction failed.'}), 500

@app.errorhandler(404)
def page_not_found(error):
    """Handle 404 Not Found errors.

    Args:
        error: The 404 error object.

    Returns:
        tuple: Rendered 404 template and HTTP status code.
    """
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(error):
    """Handle 500 Internal Server errors.

    Args:
        error: The 500 error object.

    Returns:
        tuple: Error template and HTTP status code.
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
        'site_name': 'Ryan Thompson'
    }

def create_app():
    """Application factory.

    Returns:
        Flask: The Flask app instance.
    """
    return app

def initialize_models():
    """Initialize machine learning models used by the website."""
    global digit_model, scaler, house_model, house_scaler
    try:
        # Initialize handwriting recognition model.
        logger.info("Initializing handwriting recognition model...")
        digit_model, scaler = load_sklearn_model()
        if digit_model is not None:
            logger.info("Handwriting model loaded successfully")
        else:
            logger.warning("Handwriting model not available")
        # Initialize housing price prediction model from synthetic data.
        logger.info("Initializing housing price prediction model...")
        rng = np.random.default_rng(42)
        n_samples = 400
        # Generate synthetic housing data.
        sqft = rng.uniform(600, 3500, n_samples)
        bedrooms = rng.integers(1, 6, n_samples)
        bathrooms = rng.integers(1, 4, n_samples) + rng.choice([0, 0.5], n_samples)
        year_built = rng.integers(1960, 2025, n_samples)
        # Calculate prices based on features.
        base = 50000
        price = (base + 180 * sqft + 12000 * bedrooms + 15000 * bathrooms
                 - 400 * (2025 - year_built) + rng.normal(0, 30000, n_samples))
        # Train model.
        X = np.column_stack([sqft, bedrooms, bathrooms, year_built])
        house_scaler = StandardScaler()
        Xs = house_scaler.fit_transform(X)
        house_model = LinearRegression()
        house_model.fit(Xs, price)
        logger.info("Housing model initialized successfully")
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

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