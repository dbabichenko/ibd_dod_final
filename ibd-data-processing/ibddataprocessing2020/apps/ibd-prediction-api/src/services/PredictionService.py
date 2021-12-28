import math
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from .models.ModelAssets import ModelAssets


class PredictionService:
    GLOBAL_COLUMN_INDICES_CD = {
        'unemployed': 0,
        'notMarried': 1,
        'age50': 2,
        'tobaccoEver': 3,
        'alcohol2yes': 4,
        'psyche': 5,
        'durationGroup0-15': 6,
        'durationGroup15-25': 7,
        'durationGroup25+': 8,
        'Discharge.Summary_ave3': 9,
        'ER.Report_bin': 10,
        'TelephoneGroup0': 11,
        'TelephoneGroup0-10': 12,
        'TelephoneGroup10+': 13,
        'Immunomodulators_bin': 14,
        'narcotics_bin3': 15,
        'Systemic_steroids_bin': 16,
        'SystemicSteroids3Group0': 17,
        'SystemicSteroids3Group0-2': 18,
        'SystemicSteroids3Group2+': 19,
        'LabHemoglobinGroup0': 20,
        'LabHemoglobinGroup(0-2]': 21,
        'LabHemoglobinGroup[2,': 22,
        'LabEOS0': 23,
        'LabEOS1': 24,
        'LabEOS2+': 25,
        'LabMonocytes_ave3': 26,
        'LabMonocytes_bin3': 27,
        'LabMonocytesNoAbnormalTests>2': 28,
        'Lab_EOS_bin3': 29,
        'Step2Albumin': 30,
        'Step2EOS': 31,
        'Step2Hemoglobin': 32,
        'Step2Monocytes': 33
    }

    GLOBAL_COLUMN_INDICES_UC = {
        'age': 0,
        'unemployed': 1,
        'tobaccoEver': 2,
        'family_hx': 3,
        'duration': 4,
        'duration_cat_5-25': 5,
        'duration_cat_25+': 6,
        'Discharge.Summary_bin': 7,
        'Discharge.Summary_bin3': 8,
        'ER.Report_bin': 9,
        'office_visit_bin': 10,
        'Telephone_cat_1-10': 11,
        'Telephone_cat_10+': 12,
        'colectomy_bin3': 13,
        'rx_immunomodulators_bin': 14,
        'rx_narcotics_bin3': 15,
        'rx_systemic_steroids_bin': 16,
        'rx_systemic_steroids_years_0-1': 17,
        'rx_systemic_steroids_years_1-2': 18,
        'rx_systemic_steroids_years_2+': 19,
        'lab_albumin_bin3': 20,
        'lab_hemoglobin_bin3': 21,
        'lab_eos_bin3': 22,
        'lab_monocytes_bin3': 23,
        'lab_monocytes_years': 24,
        'step2_albumin_bin': 25,
        'step2_eos_bin': 26,
        'step2_hemoglobin_bin': 27,
        'step2_monocytes_bin': 28,
    }

    GLOBAL_MODEL_FEATURES = {
        'CD': {
            'discharge': {
                'labs': [
                    'unemployed',
                    'notMarried',
                    'Discharge.Summary_ave3',
                    'ER.Report_bin',
                    'Immunomodulators_bin',
                    'LabMonocytes_bin3',
                    'LabHemoglobinGroup[2,'
                ],
                'nolabs': [
                    'unemployed',
                    'notMarried',
                    'Discharge.Summary_ave3',
                    'ER.Report_bin',
                    'alcohol2yes',
                    'Immunomodulators_bin',
                    'narcotics_bin3',
                    'psyche',
                    'unemployed:psyche'
                ]
            },
            'charge': {
                'labs': [
                    'unemployed',
                    'tobaccoEver',
                    'Discharge.Summary_ave3',
                    'ER.Report_bin',
                    'Immunomodulators_bin',
                    'narcotics_bin3',
                    'Systemic_steroids_bin',
                    'LabHemoglobinGroup(0-2]',
                    'LabHemoglobinGroup[2,',
                    'Lab_EOS_bin3',
                    'LabMonocytes_ave3',
                    'Immunomodulators_bin:narcotics_bin3',
                    'narcotics_bin3:Systemic_steroids_bin'
                ],
                'nolabs': [
                    'unemployed',
                    'tobaccoEver',
                    'Discharge.Summary_ave3',
                    'ER.Report_bin',
                    'Immunomodulators_bin',
                    'narcotics_bin3',
                    'Systemic_steroids_bin',
                    'Immunomodulators_bin:narcotics_bin3',
                    'narcotics_bin3:Systemic_steroids_bin'
                ]
            },
            'steroids': {
                'labs': [
                    'age50',
                    'durationGroup15-25',
                    'durationGroup25+',
                    'SystemicSteroids3Group0-2',
                    'SystemicSteroids3Group2+',
                    'narcotics_bin3',
                    'Immunomodulators_bin',
                    'ER.Report_bin',
                    'TelephoneGroup0-10',
                    'TelephoneGroup10+',
                    'LabMonocytesNoAbnormalTests>2',
                    'LabEOS2+',
                    'age50:Immunomodulators_bin',
                    'narcotics_bin3:Immunomodulators_bin'
                ],
                'nolabs': [
                    'age50',
                    'durationGroup15-25',
                    'durationGroup25+',
                    'SystemicSteroids3Group0-2',
                    'SystemicSteroids3Group2+',
                    'narcotics_bin3',
                    'Immunomodulators_bin',
                    'ER.Report_bin',
                    'TelephoneGroup0-10',
                    'TelephoneGroup10+',
                    'age50:Immunomodulators_bin',
                    'SystemicSteroids3Group0-2:narcotics_bin3',
                    'SystemicSteroids3Group2+:narcotics_bin3',
                    'narcotics_bin3:Immunomodulators_bin'
                ]
            }
        },
        'UC': {
            'discharge': {
                'labs': [
                    'age',
                    'duration_cat_5-25',
                    'duration_cat_25+',
                    'Discharge.Summary_bin',
                    'ER.Report_bin',
                    'Telephone_cat_10+',
                    'rx_narcotics_bin3',
                    'rx_immunomodulators_bin',
                    'lab_hemoglobin_bin3',
                    'age:duration_cat_5-25',
                    'age:duration_cat_25+',
                    'Discharge.Summary_bin:ER.Report_bin',
                ],
                'nolabs': [
                    'age',
                    'duration_cat_5-25',
                    'duration_cat_25+',
                    'Discharge.Summary_bin',
                    'ER.Report_bin',
                    'Telephone_cat_10+',
                    'rx_narcotics_bin3',
                    'rx_immunomodulators_bin',
                    'age:duration_cat_5-25',
                    'age:duration_cat_25+',
                    'Discharge.Summary_bin:ER.Report_bin',
                ],
            },
            'charge': {
                'labs': [
                    'unemployed',
                    'tobaccoEver',
                    'family_hx',
                    'duration',
                    'colectomy_bin3',
                    'ER.Report_bin',
                    'office_visit_bin',
                    'Telephone_cat_10+',
                    'lab_eos_bin3',
                    'lab_albumin_bin3',
                    'lab_eos_bin3:lab_albumin_bin3',
                ],
                'nolabs': [
                    'unemployed',
                    'tobaccoEver',
                    'family_hx',
                    'colectomy_bin3',
                    'ER.Report_bin',
                    'office_visit_bin',
                    'Telephone_cat_10+',
                    'colectomy_bin3:ER.Report_bin',
                ],
            },
            'steroids': {
                'labs': [
                    'rx_systemic_steroids_years_0-1',
                    'rx_systemic_steroids_years_1-2',
                    'rx_systemic_steroids_years_2+',
                    'Discharge.Summary_bin3',
                    'office_visit_bin',
                    'Telephone_cat_1-10',
                    'Telephone_cat_10+',
                    'lab_eos_bin3',
                ],
                'nolabs': [
                    'rx_systemic_steroids_years_0-1',
                    'rx_systemic_steroids_years_1-2',
                    'rx_systemic_steroids_years_2+',
                    'Discharge.Summary_bin3',
                    'office_visit_bin',
                    'Telephone_cat_1-10',
                    'Telephone_cat_10+',
                ]
            }
        }
    }

    asset_path = None

    def __init__(self, asset_path):
        self.asset_path = asset_path

    def __get_column_indices_dict(self, diagnosis):
        return self.GLOBAL_COLUMN_INDICES_CD if diagnosis == 'CD' else self.GLOBAL_COLUMN_INDICES_UC

    def __get_patient_has_labs(self, patient, diagnosis):
        # patient data can either contain data for labs or not - shortcutting this with a single check
        has_labs = False

        if diagnosis == 'CD' and patient[self.GLOBAL_COLUMN_INDICES_CD['LabMonocytes_bin3']] != None:
            has_labs = True
        elif diagnosis == 'UC':
            has_labs = True

        return has_labs

    def __get_patient_features(self, patient, diagnosis, prediction_type, has_labs):
        patient_features = []
        column_indices = self.__get_column_indices_dict(diagnosis)

        # can't do a list comprehension here because we have to check for interaction terms (e.g. "term1:term2")
        for feature in self.GLOBAL_MODEL_FEATURES[diagnosis][prediction_type]['labs' if has_labs else 'nolabs']:
            if ':' not in feature:
                patient_features.append(patient[column_indices[feature]])
            else:
                # assuming only two-way interactions, if that's even what they're called
                term_indices = [column_indices[term] for term in feature.split(":")]
                interaction_features = [patient[term_index] for term_index in term_indices]
                patient_features.append(np.prod(interaction_features))

        # the model requires an intercept term of 1, but i don't want the caller to have to pass it so i just insert it here
        patient_features = np.insert(patient_features, 0, 1)

        return patient_features

    def __resolve_model_assets(self, diagnosis='CD', prediction_type='discharge', has_labs=False):
        step = 1
        path_start = os.path.join(self.asset_path, diagnosis)
        path_end = f"_{prediction_type}{'_nolab' if not has_labs else ''}_step{step}.csv"

        # coefficients are easy, just need values
        coefficients = pd.read_csv(os.path.join(path_start, f"Coefficients{path_end}")).values

        covariance_matrix = pd.read_csv(os.path.join(path_start, f"Covariance matrix{path_end}"))
        # the covariance matrix has a column that relates the coefficients but we don't need it here
        covariance_matrix = covariance_matrix.drop(columns=['Unnamed: 0'])
        # and we don't need the frame, just the values
        covariance_matrix = covariance_matrix.values

        return ModelAssets(
            coefficients=coefficients,
            covariance_matrix=covariance_matrix
        )

    def __predict_single(self, patient, model_assets, ci_level):
        # for clarity (?)
        beta = model_assets.coefficients
        v_beta = model_assets.covariance_matrix

        linear = np.matmul(patient, beta.transpose())
        phat = 1 / (1 + math.exp(-linear))

        vx_beta = np.matmul(np.matmul(patient, v_beta), patient.transpose())
        lb_linear = linear - norm.ppf(q=0.5 + ci_level / 2) * math.sqrt(vx_beta)
        ub_linear = linear + norm.ppf(q=0.5 + ci_level / 2) * math.sqrt(vx_beta)
        lb = 1 / (1 + math.exp(-lb_linear))
        ub = 1 / (1 + math.exp(-ub_linear))

        return {
            'phat': phat,
            'lb': lb,
            'ub': ub,
            'ci': ci_level
        }

    def get_patient_features_format(self):
        return {
            'cd': self.GLOBAL_COLUMN_INDICES_CD,
            'uc': self.GLOBAL_COLUMN_INDICES_UC
        }

    def predict(self, patient_data, diagnosis='CD', ci_level=.95):
        results = []

        column_indices = self.__get_column_indices_dict(diagnosis)
        prediction_types = ['discharge', 'charge', 'steroids']

        for patient in patient_data:
            # convert to list for convenience methods and serialization
            patient = patient.tolist()
            # each patient gets a dict containing predictions for each of the three prediction types
            patient_predictions = {}

            # if the patient vector is less than the total number of possible columns, reshape to the expected size for convenience (defaulting to null)
            while len(patient) < len(column_indices):
                patient.append(None)

            # we need to know if the patient has lab data, because that affects which model we use
            patient_has_labs = self.__get_patient_has_labs(patient, diagnosis)

            for prediction_type in prediction_types:
                model_assets = self.__resolve_model_assets(diagnosis, prediction_type, has_labs=patient_has_labs)
                patient_features = self.__get_patient_features(patient, diagnosis, prediction_type, patient_has_labs)
                patient_predictions[prediction_type] = self.__predict_single(patient_features, model_assets, ci_level)

            results.append({
                'original_data': [value for value in patient if value is not None],
                'diagnosis': diagnosis,
                'predictions': patient_predictions
            })

        return results
