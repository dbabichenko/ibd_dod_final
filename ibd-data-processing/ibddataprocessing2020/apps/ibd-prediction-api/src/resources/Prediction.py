from flask import request
from flask_restful import Resource

import os
import numpy as np


class Prediction(Resource):
    GLOBAL_ASSETS_PATH = os.path.join(__file__, 'assets')

    def __init__(self, **kwargs):
        self.prediction_service = kwargs['prediction_service']

    def get(self):
        return self.prediction_service.get_patient_features_format()

    def post(self):
        body_dict = request.get_json(force=True)

        # read info out of the request
        patient_data = np.array(body_dict['patientData'])

        # support either a 1-dim array for a single patient's prediction, or 2dim for multiple patients
        if patient_data.ndim == 1:
            patient_data = [patient_data]

        diagnosis = body_dict['diagnosis'].upper()
        ci_level = .95 if 'ciLevel' not in body_dict else body_dict['ciLevel']

        # create a predictor with the right parameters
        return self.prediction_service.predict(
            patient_data,
            diagnosis=diagnosis,
            ci_level=ci_level
        )
