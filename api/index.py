from app import app
from flask import jsonify, request

# Vercel requires a callable application
api = app

# Add CORS headers for Vercel
@api.after_request
after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Health check endpoint for Vercel
@api.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Heart Disease Prediction API is running on Vercel',
        'environment': 'production'
    })

# This is needed for Vercel to properly handle the function
def handler(req, context):
    return api(req.environ, lambda x, y: None)
