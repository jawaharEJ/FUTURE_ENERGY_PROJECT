# Future Energy Consumption Predictor

A Flask-based web application that predicts future energy consumption using machine learning. Features user authentication, admin dashboard, chatbot, and data visualization.

## Features

- **User Registration & Login**: Secure user accounts with session management.
- **Energy Prediction**: Uses Linear Regression to predict consumption based on year, month, population, and industrial growth.
- **Data Visualization**: Generates charts showing historical data and predictions using Matplotlib.
- **Admin Dashboard**: View user statistics and manage data.
- **Chatbot**: Interactive bot answering questions about the app.
- **Responsive UI**: Professional interface built with Bootstrap.

## Technologies Used

- **Backend**: Flask (Python)
- **ML**: scikit-learn (Linear Regression)
- **Frontend**: HTML, CSS, Bootstrap, JavaScript
- **Data**: Pandas, Matplotlib
- **Storage**: JSON for users (can be upgraded to database)

## Installation

1. Clone the repository.
2. Create a virtual environment: `python -m venv .venv`
3. Activate it: `.venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Run the app: `python app.py`
6. Open http://127.0.0.1:5000/register to register, then login.

## Usage

- Register a new account or login.
- Enter prediction parameters and click Predict.
- View the chart and result.
- Access chatbot for help.
- Admins can view dashboard at /admin.

## Project Structure

- `app.py`: Main Flask application
- `templates/`: HTML templates
- `static/`: CSS and generated images
- `energy_data.csv`: Sample data
- `users.json`: User data storage
- `requirements.txt`: Dependencies

## Future Enhancements

- Database integration
- API endpoints
- Real-time features
- Advanced ML models
- Deployment

## License

This project is for educational purposes.