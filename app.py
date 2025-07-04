# Importing packages
import pandas as pd
from src.components.create_custom_data import CreateCustomData
from Flask import Flask, request, render_template, jsonify

# Creating the Flask app
app = Flask(__name__)

# Creating the home page
@app.route('/')
def index():
    '''
    This function creates the home page for the web application.
    '''
    return render_template('index.html')

# Creating a function to predict the datapoint
@app.route('/predict.html', methods=['GET', 'POST'])
def predict_datapoint():
    '''
    This function will display the prediction landing page if the method is "GET".
    If the method is "POST", the function will run the prediction.
    '''
    # Display the prediction landing page if the method is "GET"
    if request.method == 'GET':
        return render_template('predict.html')
    else:
        # If the method is "POST", then return the prediction
        data = CreateCustomData(
            x = int(request.form.get('x')),
            y = int(request.form.get('y')),
            direction = str(request.form.get('direction'))
        )
        
        # Creating a dataframe from the user entered data
        df = data.create_dataframe()
    