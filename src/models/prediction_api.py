
# import
from flask import Flask, request
import pandas as pd
import numpy as np
import json
from sklearn.externals import joblib
from pathlib import Path

# create the flask app
app = Flask(__name__)

# create file path
project_dir = Path.home() / 'Python' / 'Kaggle' / 'titanic_survival'
model_file_path = project_dir / 'models' / 'poly_svc_cp_model.pkl'

# load model
model = joblib.load(model_file_path)

# use the Python decorator to create the API route with one endpoint ('/api') that will be used for POST requests
@app.route('/api', methods = ['POST'])

# the prediciton method that will be invoked internally once the API is invoked
def make_prediction():
    # read json object and conter to json string
    data = json.dumps(request.get_json(force = True))
    
    # create pandas dataframe from json string
    df = pd.read_json(data)
    
    # extract PassengerIds
    passenger_ids = df['PassengerId']
    
    # actual 'Survived' values
    # in practice, we wouldn't have this, but we'll include it for testing purposes since train is labeled
    actuals = df['Survived']
    
    # extract feature columns
    X = df.drop(['PassengerId', 'Survived'], axis = 1)
    
    # make predictions
    predictions = model.predict(X)
    
    # create respnse data fram
    df_response = pd.DataFrame({'PassengerId': passenger_ids,
                                'Predicted' : predictions,
                                'Actual' : actuals})
    
    # return json response object
    return df_response.to_json()
    
if __name__ == '__main__':
    # can pick any unused port
    # debug = True means if you have any problems in the API call, then you get a detailed stack trace
    # good for development process, probably set to False in the production environment
    app.run(port = 10001, debug = True)