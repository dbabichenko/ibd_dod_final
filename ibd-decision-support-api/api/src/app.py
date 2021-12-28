import os
from flask import Flask
from flask_cors import CORS
from flask_restful import Resource, Api
from resources.Prediction import Prediction
from resources.HelloWorld import HelloWorld
from services.PredictionService import PredictionService

app = Flask(__name__)
cors = CORS(app)
api = Api(app)

api.add_resource(HelloWorld, '/')
api.add_resource(Prediction, '/predict',
                 resource_class_kwargs={'prediction_service': PredictionService(os.path.join(os.getcwd(), './assets'))})

if __name__ == '__main__':
    app.run(debug=True)
