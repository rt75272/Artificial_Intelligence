# Cat vs Dog Demo (Standalone)

This folder contains the Cat vs Dog image classification demo that was removed from the main website.

Contents:
- templates/cat_dog_demo.html — the demo page (Jinja2 template)
- train_ultra_model.py — training script for the ultra model
- cat_dog_ultra_model.h5 — trained model weights
- cat_dog_ultra_metadata.json — model metadata
- static/images/examples/ — example images used by the demo

Notes:
- The HTML template extends the website's `base.html`. If you want to run it standalone, either copy the base layout into this folder or remove the `{% extends %}` and embed a minimal HTML shell.
- The original Flask endpoints (`/demos/cat-dog` and `/classify-image`) were removed from `ai_website/app.py`. If you want a standalone Flask app for this demo, create a small Flask `app.py` here that serves the template and implements the `POST /classify-image` endpoint (you can reuse the logic from the removed code in `ai_website/app.py` history).

Quick start idea:
1. Create a minimal Flask app in this folder to serve `cat_dog_demo.html` and handle `/classify-image`.
2. Run `FLASK_APP=app.py flask run` to test locally.

