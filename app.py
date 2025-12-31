from flask import Flask, jsonify, request, render_template, send_from_directory, url_for
from flask_cors import CORS
import os
import sys
import joblib
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
import json

# Initialize Flask app with template and static folders
app = Flask(__name__,
            static_folder=os.path.join('app', 'static'),
            template_folder=os.path.join('app', 'templates'))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Enable CORS for all routes
CORS(app, resources={
    r"/api/*": {"origins": "*"}
})

# Configuration
app.config['UPLOAD_FOLDER'] = os.path.join('app', 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Import routes after app is created to avoid circular imports
from app.routes import main as main_blueprint
app.register_blueprint(main_blueprint)

# This is required for gunicorn
application = app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"Starting server on port {port}...")
    logger.info(f"Model loaded: {hasattr(main_blueprint, 'model') and main_blueprint.model is not None}")
    app.run(host='0.0.0.0', port=port, debug=False)