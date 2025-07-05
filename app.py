# Importing packages
import pandas as pd
from src.components.create_custom_data import CreateCustomData
from src.components.make_predictions import MakePredictions
from flask import Flask, request, render_template, jsonify

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
    # If the method is "POST", then run the prediction
    elif request.method == 'POST':
        # Capturing the data entered by the user on the web app
        data = CreateCustomData(
            x = int(request.form.get('x')),
            y = int(request.form.get('y')),
            direction = str(request.form.get('direction'))
        )
        
        # Creating a dataframe from the user entered data
        df = data.create_dataframe()
        
        # Instantiating the prediction class and making the prediction
        prediction = MakePredictions()
        preds = prediction.predict(df)
        
        # Returning the prediction to the web app
        return render_template('predict.html', results=preds, pred_df=df)

# Creating a function to return the prediction as an API call
def fetch_prediction_api():
    '''
    This function will take the data entered by the user and then 
    return the prediction as an API call.
    ==================================================================================
    ---------------------
    Returns:
    ---------------------
    predictions : json - This is the prediction in a JSON format.
    ===================================================================================
    '''
    if request.method == 'POST':
        # Capturing the data entered by the user on the web app
        data = CreateCustomData(
            x = int(request.form.get('x')),
            y = int(request.form.get('y')),
            direction = str(request.form.get('direction'))
        )
        
        # Creating a dataframe from the user entered data
        df = data.create_dataframe()
        
        # Instantiating the prediction class and making the prediction
        prediction = MakePredictions()
        preds = prediction.predict(df)
        
        # Creating a dictionary for the preds
        preds_dict = {
            'prediction': preds
        }
        
        return jsonify(preds_dict)
    
# Running the Flask app
if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"Failed to run the Flask app: {e}")
    
    