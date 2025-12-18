import os
import sys
from flask import Flask, jsonify

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def create_app():
    # Create and configure the app
    app = Flask(__name__, 
                instance_relative_config=True,
                template_folder='templates',
                static_folder='static')
    
    # Ensure the instance folder exists
    os.makedirs(app.instance_path, exist_ok=True)
    
    # Set a secret key for the application
    app.config['SECRET_KEY'] = 'dev'  # Change this to a random secret key in production
    
    # Configure upload folder
    app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Configure models folder
    app.config['MODELS_FOLDER'] = os.path.join(os.path.dirname(app.root_path), 'app', 'models')
    os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)
    
    # Register blueprints
    try:
        from app.routes import main as main_blueprint
        app.register_blueprint(main_blueprint, url_prefix='/')
    except ImportError as e:
        print(f"Error importing routes: {e}")
        # Register a simple route if the main blueprint fails to load
        @app.route('/')
        def index():
            return 'Heart Disease Prediction API is running. The routes could not be loaded.'
    
    # Add error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({
            'status': 'error',
            'message': 'Resource not found',
            'error': str(error)
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            'status': 'error',
            'message': 'An internal server error occurred',
            'error': str(error)
        }), 500

    @app.errorhandler(Exception)
    def handle_exception(error):
        return jsonify({
            'status': 'error',
            'message': 'An unexpected error occurred',
            'error': str(error)
        }), 500
    
    return app
