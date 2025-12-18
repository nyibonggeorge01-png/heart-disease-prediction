# Heart Disease Prediction Web Application

This is a web application that predicts the likelihood of heart disease based on various health parameters using machine learning.

## Features

- User-friendly web interface for inputting health parameters
- Machine learning model trained on heart disease dataset
- Real-time prediction with confidence score
- Responsive design that works on desktop and mobile devices

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the data**
   - Ensure you have the `heart_disease_data.xlsx` file in the project root directory.

5. **Train the model**
   ```bash
   python train_and_save_model.py
   ```
   This will train the model and save it to `app/models/heart_disease_model.pkl`.

## Running the Application

1. **Start the Flask development server**
   ```bash
   python run.py
   ```

2. **Access the application**
   Open a web browser and navigate to:
   ```
   http://localhost:8000
   ```

## Project Structure

```
.
├── app/
│   ├── __init__.py         # Flask application factory
│   ├── routes.py           # Application routes and views
│   ├── models/             # Directory for trained models
│   ├── static/             # Static files (CSS, JS, images)
│   │   └── uploads/        # Directory for file uploads
│   └── templates/          # HTML templates
│       └── index.html      # Main application page
├── heart_disease_data.xlsx # Dataset
├── train_and_save_model.py # Script to train and save the model
├── requirements.txt        # Python dependencies
└── run.py                 # Application entry point
```

## API Endpoints

- `GET /` - Home page with the prediction form
- `POST /predict` - API endpoint for making predictions
- `GET /api/health` - Health check endpoint

## Model Information

The application uses a Random Forest Classifier with the following features:

- **Numeric Features**:
  - Age (years)
  - Cholesterol (mg/dl)
  - Cigarettes per day
  - Blood sugar (mg/dl)

- **Categorical Features**:
  - Sex
  - Family history of heart disease
  - Chest pain type

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset: [Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) (modified for educational purposes)
