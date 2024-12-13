import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
df = pd.read_csv("Mumbai House Prices.csv")
pipe = pickle.load(open("DCmodel1.pkl", "rb"))

# Create label encoders for categorical columns
type_encoder = LabelEncoder()
locality_encoder = LabelEncoder()
region_encoder = LabelEncoder()
status_encoder = LabelEncoder()
age_encoder = LabelEncoder()

# Fit encoders on the dataset
df['type_encoded'] = type_encoder.fit_transform(df['type'])
df['locality_encoded'] = locality_encoder.fit_transform(df['locality'])
df['region_encoded'] = region_encoder.fit_transform(df['region'])
df['status_encoded'] = status_encoder.fit_transform(df['status'])
df['age_encoded'] = age_encoder.fit_transform(df['age'])

@app.route('/')
def index():
    regions = sorted(df['region'].unique())
    localities = sorted(df['locality'].unique())
    return render_template('index.html', regions=regions, localities=localities)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        region = request.form.get('region')
        locality = request.form.get('locality')
        bhk = int(request.form.get('bhk'))
        Type = request.form.get('type')
        area = float(request.form.get('area'))
        status = request.form.get('status')
        age = request.form.get('age')

        # Encode categorical inputs
        type_encoded = type_encoder.transform([Type])[0]
        locality_encoded = locality_encoder.transform([locality])[0]
        region_encoded = region_encoder.transform([region])[0]
        status_encoded = status_encoder.transform([status])[0]
        age_encoded = age_encoder.transform([age])[0]

        # Create input DataFrame
        input_data = pd.DataFrame([[bhk, type_encoded, locality_encoded, area, region_encoded, status_encoded, age_encoded]],
                                  columns=['bhk', 'type', 'locality', 'area', 'region', 'status', 'age'])

        # Make prediction
        prediction = pipe.predict(input_data)[0] * 1e5
        return str(np.round(prediction, 2))

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
