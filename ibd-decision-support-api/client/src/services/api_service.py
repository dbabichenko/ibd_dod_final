import os
import requests


class ApiService:
    base_url__ = os.getenv('API_URL_ROOT')

    def get_prediction(self, diagnosis, patient_data):
        body = {
            'diagnosis': diagnosis,
            'patientData': patient_data
        }

        r = requests.post(f'{self.base_url__}/predict', json=body)
        return r.json()
