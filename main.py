import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, url_for

# Set the Agg backend before importing pyplot
import matplotlib
matplotlib.use('Agg')

# Creating object of class Flask
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


def plot_prediction_probabilities(probabilities, labels):
    plt.figure(figsize=(8, 6))
    plt.bar(labels, probabilities, color=['green', 'red'])
    plt.ylabel('Probability')
    plt.title('Prediction Probabilities')
    plt.ylim([0, 1])
    plt.savefig('static/prediction_probabilities.png')
    plt.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    input_features = [
        request.form.get('age'),
        request.form.get('gender'),
        request.form.get('cp'),
        request.form.get('restbps'),
        request.form.get('chol'),
        request.form.get('fbs'),
        request.form.get('restecg'),
        request.form.get('maxHeartRate'),
        request.form.get('exang'),
        request.form.get('oldpeak'),
        request.form.get('slope'),
        request.form.get('majorVessel'),
        request.form.get('glucose'),
        request.form.get('kcm'),
        request.form.get('troponin')
    ]

    # Convert input features to a NumPy array
    input_features_array = [float(feature) for feature in input_features]

    # Reshape the array to match the expected input shape for your model
    input_features_array = np.array(input_features_array).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_features_array)
    prediction_probabilities = model.predict_proba(input_features_array)[0]

    # Plot the prediction probabilities
    plot_prediction_probabilities(prediction_probabilities, ['No Heart Disease', 'Heart Disease'])

    # Determine prediction result and color
    if prediction == 0:
        pri = "No Heart Disease"
        color = "green"
    else:
        pri = "Heart Disease"
        color = "red"

    # Render the result
    return render_template('index.html', 
                           prediction_text=f'<span style="color: {color};">{pri}</span>',
                           prediction_image=url_for('static', filename='prediction_probabilities.png'))

if __name__ == '__main__':
    app.run(debug=True)
