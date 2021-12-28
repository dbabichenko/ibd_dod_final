import os
from flask import Flask, redirect, request, render_template, url_for
from flask_bootstrap import Bootstrap
from dotenv import load_dotenv
from services.api_service import ApiService
from services.form_parser_service import FormParserService

# load env variables from dotenv
load_dotenv()

app = Flask(__name__)
bootstrap = Bootstrap(app)
api = ApiService()
form_parser = FormParserService()


@app.route('/', methods=['GET'])
def index():
    return redirect(url_for('diagnosis_uc'))


@app.route('/diagnosis/ulcerative-colitis', methods=['GET'])
def diagnosis_uc():
    return render_template('ulcerative-colitis.html')


@app.route('/diagnosis/ulcerative-colitis', methods=['POST'])
def post_diagnosis_uc():
    parsed_form = form_parser.parseUcForm(request.form)
    response = api.get_prediction('uc', parsed_form)
    return render_template('ulcerative-colitis.html', predictions=response[0]['predictions'])


@app.route('/diagnosis/crohns', methods=['GET'])
def diagnosis_cd():
    return render_template('crohns.html')


@app.route('/diagnosis/crohns', methods=['POST'])
def post_diagnosis_cd():
    parsed_form = form_parser.parseCdForm(request.form)
    response = api.get_prediction('cd', parsed_form)
    return render_template('crohns.html', predictions=response[0]['predictions'])


if __name__ == "__main__":
    host_ip = os.getenv('APP_HOST_IP', '0.0.0.0')
    host_port = os.getenv('APP_HOST_PORT', 80)
    app.run(host=host_ip, port=host_port)
