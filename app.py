from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('heartmodel.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from form
        input_data = [float(x) for x in request.form.values()]
        
        # Convert data to numpy array
        input_data_as_numpy_array = np.asarray(input_data)
        
        # Reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data_reshaped)
        
        # Return result
        if prediction[0] == 0:
            result = 'The Person does not have a Heart Disease'
        else:
            result = 'The Person has Heart Disease'
        
        return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
