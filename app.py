from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipline import PredictPipeline, CustomData

application = Flask(__name__)

app = application





@app.route('/', methods=['POST','GET'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=request.form.get('reading_score'),
            writing_score=request.form.get('writing_score')
        )

        predict_pipeline = PredictPipeline()
        preds = predict_pipeline.predict(data.get_data_as_data_frame())

        return render_template('home.html', prediction=preds)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
