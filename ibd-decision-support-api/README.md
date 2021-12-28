# University of Pittsburgh's IBD Decision Support API

This repository contains an API developed by a team at [University of Pittsburgh](https://pitt.edu) in a collaboration between the School of Computing and Information, the School of Medicine, and the School of Public Health. Based on properties of a patient, including medical history, behavioral history, and demographic information, it can generate predictions about whether a patient will experience high medical charges, hospitalization, and prescription of systemic steroids in the next year. An example client to consume the API is also provided, though it can be consumed by any REST-aware client.

## Deployment

Both the client and API applications are simple Python 3 / Flask apps that can be deployed using a variety of web servers. We hosted them on Ubuntu and RHEL servers using NGINX (see a tutorial for configuring an NGINX Python app [here](https://dev.to/brandonwallace/deploy-flask-the-easy-way-with-gunicorn-and-nginx-jgc)), but any environment capable of serving Python will work.

Both the client and API contain requirements.txt files that describe their pip dependencies. When deploying these, it is strongly recommended that you [create a virtual environment](https://docs.python.org/3/library/venv.html) for each. Once you've activated the virtual environment for each app, you can install its dependencies from within the app's subdirectory by running `pip install -r requirements.txt`.

## Running locally

To run either app locally, simply run its `app.py` (e.g. `python ./api/src/app.py` from the project root). Note that Python 3.6+ is required.

## About the apps

### The API

#### Overview

The API is a RESTful web service that generates predictions about a patient given a number of their history, demographic information, and treatment history. While we call it from our client Flask app, as a REST-based service, it can be consumed by any client capable of issuing HTTP requests, including browsers, mobile apps, desktop apps, and more.

The API has two endpoints at `GET /predict` and `POST /predict`. (There is a simple "Hello, world"-style endpoint at `GET /` that you can use to easily ensure your deployment is good to go.)

#### The POST /predict endpoint

This endpoint accepts the information required to make a prediction as a JSON payload and returns JSON that contains predictions about the patient's likelihood to experience high charges (of over $100k), to be prescribed systemic steroids, and to be hospitalized, all of which are scoped over the next year of their care.

##### Input

| Field | Type | Description |
| ---- | --- | ----------- |
| patientData | 1 or 2-dimensional array of numbers | An array that contains numerical values that describe the patient's demographics and treatment history. If the array is one-dimensional, a prediction will be generated assuming a single patient. If multiple patients are described using two dimensions, the prediction results will contain a prediction for each patient. |
| diagnosis | either `'UC'` or `'CD'` | Whether the patient's underlying IBD cause is Crohn's Disease (CD) or Ulcerative Colitis (UC) |
| ciLevel* | number from 0 to 1 | Describes the width of the confidence interval used in generating the prediction.

*ciLevel is optional and will default to `0.95` if unspecified.

Note that `patientData` is a numerical array and does not accept named parameters for the patient's properties. These must be specified in a fixed order. This order is described below, but it can also be supplied by the API using the `GET /predict` endpoint which returns a textual representation of this format. Note that predictions for ulcerative colitis require different information than predictions for Crohn's disease.

###### patientData format

All values are numeric. Many are boolean values expressed as 0s or 1s (1 is positive), but some are continuous values. Additionally, because the prediction model is a simple regression, it contains some one-hot-encoded features that span multiple ordinal positions.

**For Crohn's disease patients**

*NOTE:*
* Only the values prior to `LabHemoGlobinGroup0` (i.e. the first 20 ordinal positions) are required to make a prediction, but passing the remaining values may improve prediction quality.
* If you pass fewer values than the maximum possible (in this case, 34), the API will automatically fill the remaining values with `null`.

| Feature(s) | Ordinal position(s) | Type | Description |
| -- | -- | -- | ----------- |
| unemployed | 0 | boolean | Is the patient unemployed? |
| notMarried | 1 | boolean | Is the patient single? |
| age50 | 2 | boolean | Is the patient 50 years of age or over? |
| tobaccoEver | 3 | boolean | Has the patient ever used tobacco? |
| alcohol2yes | 4 | boolean | Does the patient currently drink alcohol? |
| psyche | 5 | boolean | Does the patient have any relevant psychiatric comorbidities? |
| durationGroup0-15, durationGroup15-25, durationGroup25+ | 6-8 | boolean | How long has the patient been diagnosed with Crohn's? |
|  Discharge.summary_ave3 | 9 | float | In the past three years, how many hospitalizations has the patient had each year on average? |
| ER.Report_bin | 10 | boolean | In the past year, has the patient been admitted to an ER? | 
| TelephoneGroup0, TelephoneGroup0-10, TelephoneGroup10+ | 11-13 | boolean | In the past year, how many phone calls has the patient had with the doctor's office? |
| Immunomodulators_bin | 14 | boolean | In the past year, has the patient been treated with immunomodulators? |
| Narcotics_bin3 | 15 | boolean | In the past three years, has the patient been treated with narcotics? |
| Systemic_steroids_bin | 16 | boolean | In the past year, has the patient been treated with systemic steroids? |
| SystemicSteroids3Group0, SystemicSteroids3Group0-2, SystemicSteroids3Group2+ | 17-19 | boolean | Of the past three years, in how many was the patient treated with systemic steroids? | 
| LabHemoglobinGroup0, LabHemoglobinGroup(0, 2], LabHemoglobinGroup[2, | 20-22 | boolean | Of the past three years, in how many did the patient have an abnormal hemoglobin value? |
| LabEOS0, LabEOS1, LabEOS2+ | 23-25 | boolean | Of the past three years, in how many did the patient have an abnormal EOS blood test? |
| LabMonocytes_ave3 | 26 | float | In the past three years, how many abnormal monotcytes tests did the patient have on average each year? |
| LabMonocytes_bin3 | 27 | boolean | In the past three years, did the patient have an abnormal monocytes test? |
| LabMonocytesNoAbnormalTests>2 | 28 | boolean | In the past three years, were the patients monocytes values abnormal more than twice?
| Lab_EOS_bin3 | 29 | boolean | *Same as LabMonocytes_bin3 above, repeat the value*
| Step2Albumin | 30 | boolean | Has the patient had abnormal albumin test in the last year? |
| Step2EOS | 31 | boolean | Has the patient had abnormal EOS test in the last year? |
| Step2Hemoglobin | 32 | boolean | Has the patient had abnormal hemoglobin test in the last year? |
| Step2Monocytes | 33 | boolean | Has the patient had abnormal monocytes test in the last year? |

**For ulcerative colitis patients**

*NOTE:*
* Only the values prior to `lab_albumin_bin3` (i.e. the first 20 ordinal positions) are required to make a prediction, but passing the remaining values may improve prediction quality.
* If you pass fewer values than the maximum possible (in this case, 29), the API will automatically fill the remaining values with `null`.


| Feature(s) | Ordinal position(s) | Type | Description |
| -- | -- | -- | ----------- |
| age | 0 | integer | How old is the patient? |
| unemployed | 1 | boolean | Is the patient unemployed? |
| tobaccoEver | 2 | boolean | Has the patient ever used tobacco? |
| family_hx | 3 | boolean | Does that the patient's family have a history of ulcerative colitis? |
| duration | 4 | int | How long, in years, has the patient been diagnosed with ulcerative colitis? |
| duration_cat_5-25, duration_cat_25+ | 5-6 | boolean | How long, in years, has the patient been diagnosed with ulcerative colitis? |
| Discharge.Summary_bin | 7 | boolean | Has the patient been hospitalized in the last year? |
| Discharge.Summary_bin | 8 | boolean | Has the patient been hospitalized in the last 3 years? |
| ER.Report_bin | 9 | boolean | Has the patient been hospitalized in the last year? |
| office_visit_bin | 10 | boolean | In the past year, has the patient had more than one office visit? |
| Telephone_cat_1-10, Telephone_cat_10+ | 11-12 | boolean | How many phone calls to the doctor's office has the patient had? |
| colectomy_bin3 | 13 | boolean | In the past 3 years, has the patient had a colectomy? |
| rx_immunomodulators_bin | 14 | boolean | In the past year, has the patient been treated with immunomodulators? |
| rx_narcotics_bin | 15 | boolean | In the past 3 years, has the patient been treated with narcotics? |
| rx_systemic_steroids_bin | 16 | boolean | In the past year, has the patient been treated with systemic steroids? |
| rx_systemic_steroids_years_0-1,rx_systemic_steroids_years_1-2, rx_systemic_steroids_years_2+  | 17-19 | boolean | In the past year, has the patient been treated with immunomodulators? |
| lab_albumin_bin3 | 20 | boolean | In the past three years, has the patient had an abnormal albumin test? |
| lab_hemoglobin_bin3 | 21 | boolean | In the past three years, has the patient had an abnormal hemoglobin test? |
| lab_eos_bin3 | 22 | boolean | In the past three years, has the patient had an abnormal EOS blood test? |
| lab_monocytes_bin3 | 23 | boolean | In the past three years, has the patient had an abnormal monocytes test? |
| lab_monocytes_years | 24 | integer | Of the past three years, in how many did the patient have an abnormal monocytes test? |
| step2_albumin_bin | 25 | boolean | Has the patient had abnormal albumin test in the last year? |
| step2_eos_bin | 26 | boolean | Has the patient had abnormal EOS test in the last year? |
| step2_hemoglobin_bin | 27 | boolean | Has the patient had abnormal hemoglobin test in the last year? |
| step2_monocytes_bin | 28 | boolean | Has the patient had abnormal monocytes test in the last year? |

##### Output

| Field | Type | Description |
| ---- | --- | ----------- |
| original_data | 1 or 2-dimensional array of numbers | The patient data you specified when making the call. This is provided to assist in debugging. |
| diagnosis | either `'CD'` or `'UC'` | The patient diagnosis you specified when making the call. This is provided to assist in debugging. |
| predictions | object or array of objects | Contains the predictions generated by the API from the patient data in the request. If the patient data was one-dimensional, this will be an object with three predictions for a single patient (hospitalization, high charges, and systemic steroids prescription). If the patient data was two-dimensional, this will be an array of objects with an entry for each patient. |

Each object returned in `predictions` will have three properties: `discharge`, `charge`, and `steroids`. `discharge` refers to the patient's likelihood of being hospitalized, `charge` refers to the patient's likelihood of accruing charges of over $100k, and `steroids` refers to the patient's likelihood of being prescribed systemic steroids. Each prediction object has four properties that describe its likelihood:

| Field | Type | Description |
| ---- | --- | ----------- |
| phat | number 0-1 | The probability of the specified outcome. For example, a patient with a `phat` of .0743 in their `steroids` prediction has a 7.43% chance to be prescribed systemic steroids in the next year. |
| lb | number 0-1 | The lower bound of the prediction as dictated by the confidence interval width (by default, 0.95). |
| ub | number 0-1 | The upper bound of the prediction as dictated by the confidence interval width (by default, 0.95). |
| ci | number 0-1 | The confidence interval width used in generating the prediction (by default, 0.95). |

##### Sample calls

The root directory of the API contains a collection of Postman requests in the file `postman-api-requests.json`. These can be used to get you started testing the API, whether deployed or local to your machine.

### The client

The client is a relatively straightforward Flask app. It has two pages - one to make predictions for patients with Crohn's, and another to make them for patients with ulcerative colitis. Just visit the appropriate page, enter the requested information, click "Predict", and off you go.

#### Configuring the client

The client makes use of a `.env` file in its root (i.e. in the `client` folder) to know where to look for the API. When you run the client locally, rename the `client/rename-to.env` file to `.env` and enter your configuration information. For dev, it'll likely look something like this:

```
API_URL_ROOT=http://localhost:5000
APP_HOST_IP=127.0.0.1
APP_HOST_PORT=5001
```

When you deploy, make a copy of this file and change it to the desired values to accommodate your server configuration.

#### A note about formatting values for the API

Because the API is based on a multi-feature regression model, some of its required parameters can be inferred from a single user-entered value. For example, the Crohn's disease model has three one-hot-encoded features describing the number of phone calls have been recorded for the patient  in the past year (`TelephoneGroup0`, `TelephoneGroup0-10`, and `TelephoneGroup10+`). 

The client app contains code to encode these features. Future efforts should consider moving this logic into the API to simplify the process of composing a request to it.