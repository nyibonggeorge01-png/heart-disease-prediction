from app import create_app
import os
# Create the Flask application
application = create_app()
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    application.run(host='0.0.0.0', port=port, debug=False)