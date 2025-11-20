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

@app.route('/demos/svm')
def svm_demo():
    """Render the SVM classifier demo page.

    Returns:
        str: Rendered HTML template for the SVM demo.
    """
    return render_template('svm_demo.html')

@app.route('/demos/game-of-life')
def game_of_life_demo():
    """Render the Conway's Game of Life demo page.

    Returns:
        str: Rendered HTML template for the Game of Life demo.
    """
    return render_template('game_of_life.html')

@app.route('/demos/pathfinding')
def pathfinding_demo():
    """Render the A* pathfinding demo page."""
    return render_template('pathfinding.html')

@app.route('/demos/monte-carlo-pi')
def monte_carlo_pi_demo():
    """Render the Monte Carlo Pi estimation demo page."""
    return render_template('monte_carlo_pi.html')

@app.route('/demos/password-strength')
def password_strength_demo():
    """Render the password strength prediction demo page."""
    return render_template('password_strength.html')

@app.route('/demos/chatbot')
def chatbot_demo():
    """Render the interactive chatbot demo page."""
    return render_template('chatbot.html')

@app.route('/api/svm', methods=['POST'])
def train_svm():
    """Train an SVM classifier and return decision boundary with support vectors.

    Expected JSON input:
        {"points": [{"x": 0.5, "y": 0.3, "class": 0}, ...]}.

    Returns:
        JSON response with decision boundary grid and support vector indices.
    """
    try:
        from sklearn.svm import SVC
        data = request.get_json(silent=True) or {}
        points = data.get('points', [])
        if len(points) < 2:
            return jsonify({'error': 'Need at least 2 points'}), 400
        # Extract features and labels.
        X = np.array([[p['x'], p['y']] for p in points])
        y = np.array([p['class'] for p in points])
        # Check if we have both classes.
        if len(np.unique(y)) < 2:
            return jsonify({'error': 'Need points from both classes'}), 400
        # Train SVM with linear kernel.
        model = SVC(kernel='linear', C=1.0)
        model.fit(X, y)
        # Get support vectors.
        support_indices = model.support_.tolist()
        # Generate prediction grid.
        grid_size = 50
        grid = []
        confidence_grid = []
        for i in range(grid_size):
            row = []
            conf_row = []
            for j in range(grid_size):
                x_coord = j / grid_size
                y_coord = i / grid_size
                pred = model.predict([[x_coord, y_coord]])[0]
                # Get decision function value for margin visualization.
                decision_val = model.decision_function([[x_coord, y_coord]])[0]
                row.append(int(pred))
                conf_row.append(float(decision_val))
            grid.append(row)
            confidence_grid.append(conf_row)
        return jsonify({
            'grid': grid,
            'confidence': confidence_grid,
            'support_vectors': support_indices
        })
    except Exception as e:
        logger.error(f"SVM training error: {str(e)}")
        return jsonify({'error': 'Training failed'}), 500

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

@app.route('/api/game-of-life', methods=['POST'])
def step_game_of_life():
    """Compute the next Game of Life generation from a provided grid.

    Expected JSON input:
        {"grid": [[0,1,0,...], ...]}.

    Returns:
        JSON response with the updated grid.
    """
    try:
        data = request.get_json(silent=True) or {}
        grid = data.get('grid', [])
        if not grid:
            return jsonify({'error': 'No grid provided'}), 400
        arr = np.array(grid, dtype=int)
        # Use numpy roll to sum neighbors with wrap-around boundaries.
        neighbors = (
            np.roll(np.roll(arr, 1, 0), 1, 1) + np.roll(np.roll(arr, 1, 0), 0, 1) +
            np.roll(np.roll(arr, 1, 0), -1, 1) + np.roll(np.roll(arr, 0, 0), 1, 1) +
            np.roll(np.roll(arr, 0, 0), -1, 1) + np.roll(np.roll(arr, -1, 0), 1, 1) +
            np.roll(np.roll(arr, -1, 0), 0, 1) + np.roll(np.roll(arr, -1, 0), -1, 1)
        )
        new_arr = ((neighbors == 3) | ((arr == 1) & (neighbors == 2))).astype(int)
        return jsonify({'grid': new_arr.tolist()})
    except Exception as e:
        logger.error(f"Game of Life step error: {str(e)}")
        return jsonify({'error': 'Step failed'}), 500


@app.route('/api/pathfinding', methods=['POST'])
def solve_pathfinding():
    """Solve a grid shortest path with A* (4-neighbor)."""
    try:
        import heapq
        data = request.get_json(silent=True) or {}
        grid = data.get('grid', [])
        start = data.get('start')
        goal = data.get('goal')
        if not grid or start is None or goal is None:
            return jsonify({'error': 'Missing grid, start, or goal.'}), 400
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        sr, sc = int(start[0]), int(start[1])
        gr, gc = int(goal[0]), int(goal[1])
        if not (0 <= sr < rows and 0 <= sc < cols and 0 <= gr < rows and 0 <= gc < cols):
            return jsonify({'error': 'Start or goal out of bounds.'}), 400
        if grid[sr][sc] == 1 or grid[gr][gc] == 1:
            return jsonify({'error': 'Start/goal cannot be on a wall.'}), 400
        def h(r, c):
            return abs(r - gr) + abs(c - gc)
        open_heap = []
        heapq.heappush(open_heap, (h(sr, sc), 0, (sr, sc)))
        came_from = {}
        g_score = {(sr, sc): 0}
        closed = set()
        while open_heap:
            _, g, (r, c) = heapq.heappop(open_heap)
            if (r, c) in closed:
                continue
            closed.add((r, c))
            if (r, c) == (gr, gc):
                path = [(gr, gc)]
                while path[-1] != (sr, sc):
                    path.append(came_from[path[-1]])
                path.reverse()
                return jsonify({'path': path})
            for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                    ng = g + 1
                    if ng < g_score.get((nr, nc), 1e9):
                        g_score[(nr, nc)] = ng
                        came_from[(nr, nc)] = (r, c)
                        heapq.heappush(open_heap, (ng + h(nr, nc), ng, (nr, nc)))
        return jsonify({'path': []})
    except Exception as e:
        logger.error(f"Pathfinding error: {e}")
        return jsonify({'error': 'Pathfinding failed.'}), 500

@app.route('/api/monte-carlo-pi', methods=['POST'])
def monte_carlo_pi():
    """Estimate π using Monte Carlo sampling in a unit square."""
    try:
        data = request.get_json(silent=True) or {}
        samples = int(data.get('samples', 10000))
        if samples <= 0 or samples > 5_000_000:
            return jsonify({'error': 'Invalid sample count.'}), 400
        # Generate random points in [-1,1]x[-1,1].
        xy = np.random.uniform(-1.0, 1.0, size=(samples, 2))
        # Count points inside unit circle.
        dist_sq = xy[:,0]*xy[:,0] + xy[:,1]*xy[:,1]
        inside_mask = dist_sq <= 1.0
        inside = int(np.sum(inside_mask))
        pi_est = 4.0 * inside / float(samples)
        # Return a subset of points for visualization (max 2000).
        viz_count = min(samples, 2000)
        viz_idx = np.random.choice(samples, viz_count, replace=False)
        viz_points = xy[viz_idx].tolist()
        viz_inside = inside_mask[viz_idx].tolist()
        return jsonify({'pi': float(pi_est), 'inside': inside, 'total': samples, 'points': viz_points, 'points_inside': viz_inside})
    except Exception as e:
        logger.error(f"Monte Carlo error: {e}")
        return jsonify({'error': 'Computation failed.'}), 500

@app.route('/api/chatbot', methods=['POST'])
def chatbot_response():
    """Generate chatbot response using rule-based NLP."""
    try:
        import re
        import random
        data = request.get_json(silent=True) or {}
        message = data.get('message', '').lower().strip()
        if not message:
            return jsonify({'response': 'Please say something!'})
        # Greeting patterns.
        if re.search(r'\b(hi|hello|hey|greetings|sup|howdy)\b', message):
            responses = ['Hello! How can I help you today?', 'Hi there! What would you like to know?', 'Hey! Ask me anything about AI or machine learning.', 'Greetings! Ready to chat?', 'Hi! Great to see you here.', 'Hello! What brings you here today?']
            return jsonify({'response': random.choice(responses)})
        # Farewell patterns.
        if re.search(r'\b(bye|goodbye|see you|farewell|later|cya)\b', message):
            responses = ['Goodbye! Have a great day!', 'See you later!', 'Bye! Come back anytime.', 'Take care!', 'Until next time!', 'Farewell! Happy coding!']
            return jsonify({'response': random.choice(responses)})
        # Name query.
        if re.search(r'\b(your name|who are you|what are you|introduce yourself)\b', message):
            responses = ["I'm a simple rule-based chatbot created to demonstrate conversational AI. I use pattern matching and predefined responses.", "I'm an AI chatbot designed to chat about technology and answer questions!", "I'm a conversational bot that loves talking about AI, ML, and programming."]
            return jsonify({'response': random.choice(responses)})
        # Help query.
        if re.search(r'\b(help|what can you do|commands|capabilities)\b', message):
            responses = ['I can chat about AI, machine learning, answer basic questions, and have simple conversations. Try asking me about neural networks, Python, or just say hello!', 'I love discussing technology! Ask me about AI, ML, programming, or just have a casual chat.', 'I can help with questions about machine learning, Python, data science, and more. What interests you?']
            return jsonify({'response': random.choice(responses)})
        # Machine learning questions.
        if re.search(r'\b(machine learning|ml|deep learning|supervised|unsupervised)\b', message):
            responses = ['Machine learning is fascinating! It allows computers to learn from data without explicit programming.', 'ML has three main types: supervised learning (labeled data), unsupervised learning (patterns), and reinforcement learning (rewards)!', 'Machine learning powers everything from recommendation systems to self-driving cars!', 'Interesting ML fact: Neural networks can learn incredibly complex patterns that even humans struggle to define.']
            return jsonify({'response': random.choice(responses)})
        # Neural networks.
        if re.search(r'\b(neural network|deep learning|cnn|rnn|transformer)\b', message):
            responses = ['Neural networks are inspired by the human brain and excel at pattern recognition.', 'Deep learning uses multi-layer neural networks to learn hierarchical representations of data.', 'CNNs are great for images, RNNs for sequences, and Transformers have revolutionized NLP!', 'Neural networks learn by adjusting weights through backpropagation. Pretty cool, right?']
            return jsonify({'response': random.choice(responses)})
        # AI general.
        if re.search(r'\b(artificial intelligence|ai|intelligent|robots)\b', message):
            responses = ['AI and ML are transforming industries from healthcare to finance.', 'Artificial intelligence aims to create systems that can perform tasks requiring human-like intelligence.', 'AI is everywhere now - from voice assistants to recommendation engines!', 'The future of AI is exciting! We are seeing breakthroughs in natural language and computer vision.']
            return jsonify({'response': random.choice(responses)})
        # Python questions.
        if re.search(r'\bpython\b', message):
            responses = ['Python is an excellent language for AI and data science! Libraries like NumPy, scikit-learn, and TensorFlow make it powerful for ML tasks.', 'Python is popular because of its simplicity and powerful libraries. Great choice for beginners and experts alike!', 'Fun fact: Python is named after Monty Python, not the snake! 🐍', 'Python libraries like Pandas, NumPy, and Matplotlib make data analysis incredibly efficient.']
            return jsonify({'response': random.choice(responses)})
        # Programming.
        if re.search(r'\b(code|coding|program|developer|software)\b', message):
            responses = ['Programming is like solving puzzles - challenging but rewarding!', 'The best way to learn coding is by building projects. What are you working on?', 'Every expert programmer was once a beginner. Keep practicing!', 'Good code is readable code. Always write for humans, not just machines.']
            return jsonify({'response': random.choice(responses)})
        # Data science.
        if re.search(r'\b(data science|data analysis|statistics|analytics)\b', message):
            responses = ['Data science combines statistics, programming, and domain knowledge to extract insights from data.', 'The data science workflow: collect, clean, explore, model, and communicate findings!', 'Data is the new oil, but only if you can refine it into actionable insights.', 'Visualization is key in data science - a good chart tells a thousand words!']
            return jsonify({'response': random.choice(responses)})
        # Projects or demos.
        if re.search(r'\b(project|demo|portfolio|website|build)\b', message):
            responses = ['This website showcases several interactive ML demos. Have you tried them all?', 'Building projects is the best way to learn! What are you interested in creating?', 'Check out the demos on this site - from neural networks to pathfinding algorithms!', 'Portfolio projects demonstrate your skills to employers. What tech stack are you using?']
            return jsonify({'response': random.choice(responses)})
        # How are you.
        if re.search(r'\b(how are you|how r u|whats up|wassup)\b', message):
            responses = ["I'm doing well, thanks for asking!", "I'm great! Ready to chat about AI.", "Doing fantastic! How about you?", "I'm excellent! Always happy to talk tech.", "Pretty good! What can I help you with today?"]
            return jsonify({'response': random.choice(responses)})
        # Thank you.
        if re.search(r'\b(thank|thanks|thx|appreciate)\b', message):
            responses = ["You're welcome!", "Happy to help!", "Anytime!", "Glad I could assist!", "My pleasure!", "No problem at all!"]
            return jsonify({'response': random.choice(responses)})
        # Weather or time.
        if re.search(r'\b(weather|temperature|time|date|day)\b', message):
            responses = ["I'm a chatbot focused on tech topics, but I'd love to chat about AI instead!", "I don't have access to real-time data, but I can discuss machine learning algorithms!", "Time flies when you're learning ML! Want to know more about neural networks?"]
            return jsonify({'response': random.choice(responses)})
        # Jokes.
        if re.search(r'\b(joke|funny|laugh|humor)\b', message):
            responses = ["Why do programmers prefer dark mode? Because light attracts bugs! 😄", "Why did the neural network go to therapy? It had too many layers of issues!", "What's a machine learning engineer's favorite exercise? Training data! 💪", "I'd tell you a UDP joke, but you might not get it... 😅"]
            return jsonify({'response': random.choice(responses)})
        # Affirmative responses.
        if re.search(r'\b(yes|yeah|yep|sure|ok|okay|definitely)\b', message):
            responses = ["Great! What would you like to know?", "Awesome! How can I help?", "Perfect! What's on your mind?", "Cool! Let's chat.", "Nice! What topic interests you?"]
            return jsonify({'response': random.choice(responses)})
        # Negative responses.
        if re.search(r'\b(no|nope|nah|not really)\b', message):
            responses = ["No worries! Let me know if you change your mind.", "That's okay! Feel free to ask anything else.", "Understood! What else can I help with?", "Fair enough! Want to talk about something else?"]
            return jsonify({'response': random.choice(responses)})
        # Learning.
        if re.search(r'\b(learn|study|tutorial|course|beginner)\b', message):
            responses = ["Learning AI is a journey! Start with Python basics, then move to libraries like scikit-learn.", "Great resources for learning: Coursera, fast.ai, and hands-on Kaggle competitions!", "The best way to learn is by doing. Try building small projects first!", "Don't get overwhelmed - focus on fundamentals before diving into advanced topics."]
            return jsonify({'response': random.choice(responses)})
        # Default fallback.
        fallbacks = ["That's interesting! Tell me more.", "I'm not sure I understand. Can you rephrase that?", "Hmm, I don't have a good answer for that. Try asking about AI or machine learning!", "Interesting question! I'm still learning. Ask me about neural networks or Python.", "I'd love to know more about that! Can you elaborate?", "That's a bit outside my expertise, but I'm always learning! Ask me about tech topics.", "Curious! Want to discuss AI, ML, or programming instead?", "I might not know that one, but I'm great with machine learning questions!"]
        return jsonify({'response': random.choice(fallbacks)})
    except Exception as e:
        logger.error(f"Chatbot error: {e}")
        return jsonify({'error': 'Chat failed.'}), 500

@app.route('/api/password-strength', methods=['POST'])
def predict_password_strength():
    """Predict password strength using ML features and classify as weak/medium/strong."""
    try:
        import string
        import random
        from sklearn.ensemble import RandomForestClassifier
        data = request.get_json(silent=True) or {}
        action = data.get('action', 'predict')
        if action == 'generate':
            # Generate a random password.
            length = int(data.get('length', 12))
            if length < 4 or length > 64:
                return jsonify({'error': 'Length must be 4-64.'}), 400
            special_chars = '!#$%&'
            chars = string.ascii_letters + string.digits + special_chars
            password = ''.join(random.choice(chars) for _ in range(length))
            return jsonify({'password': password})
        # Extract password for prediction.
        password = data.get('password', '')
        if not password:
            return jsonify({'error': 'No password provided.'}), 400
        # Feature extraction.
        length = len(password)
        has_lower = any(c.islower() for c in password)
        has_upper = any(c.isupper() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in string.punctuation for c in password)
        unique_chars = len(set(password))
        # Simple scoring-based classification.
        score = 0
        if length >= 8:
            score += 1
        if length >= 12:
            score += 1
        if has_lower:
            score += 1
        if has_upper:
            score += 1
        if has_digit:
            score += 1
        if has_special:
            score += 1
        if unique_chars >= length * 0.75:
            score += 1
        # Classify: 0-2=weak, 3-4=medium, 5+=strong.
        if score <= 2:
            strength = 'weak'
            strength_label = 0
        elif score <= 4:
            strength = 'medium'
            strength_label = 1
        else:
            strength = 'strong'
            strength_label = 2
        # Train a simple synthetic model for demonstration.
        X_train = np.array([[8, 1, 1, 1, 1, 6], [6, 1, 0, 1, 0, 5], [12, 1, 1, 1, 1, 10], [5, 1, 0, 0, 0, 4], [15, 1, 1, 1, 1, 14]])
        y_train = np.array([1, 0, 2, 0, 2])
        clf = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
        clf.fit(X_train, y_train)
        # Predict using features.
        features = np.array([[length, int(has_lower), int(has_upper), int(has_digit), int(has_special), unique_chars]])
        pred = int(clf.predict(features)[0])
        proba = clf.predict_proba(features)[0]
        strength_map = {0: 'weak', 1: 'medium', 2: 'strong'}
        return jsonify({'strength': strength_map[pred], 'score': score, 'probability': proba.tolist(), 'features': {'length': length, 'has_lower': has_lower, 'has_upper': has_upper, 'has_digit': has_digit, 'has_special': has_special, 'unique_chars': unique_chars}})
    except Exception as e:
        logger.error(f"Password strength error: {e}")
        return jsonify({'error': 'Prediction failed.'}), 500

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