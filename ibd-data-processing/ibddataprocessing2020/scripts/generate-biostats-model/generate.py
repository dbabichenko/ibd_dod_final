import math
import numpy as np
import os
import pandas as pd
from scipy.stats import norm


def predict(inputs, beta, v_beta, ci_level=0.95):
    linear = np.matmul(inputs, beta.transpose())
    phat = 1 / (1 + math.exp(-linear))

    vx_beta = np.matmul(np.matmul(inputs, v_beta), inputs.transpose())
    lb_linear = linear - norm.ppf(q=0.5 + ci_level / 2) * math.sqrt(vx_beta)
    ub_linear = linear + norm.ppf(q=0.5 + ci_level / 2) * math.sqrt(vx_beta)
    lb = 1 / (1 + math.exp(-lb_linear))
    ub = 1 / (1 + math.exp(-ub_linear))

    return {
        'phat': phat,
        'lb': lb,
        'ub': ub,
    }


def predict_row(x):
    prediction = predict(x.values[1:], beta, v_beta)
    x['my_phat'] = prediction['phat']
    x['my_lb'] = prediction['lb']
    x['my_ub'] = prediction['ub']

    return x


assets_path = os.path.join(os.getcwd(), "./ibddataprocessing2020/scripts/generate-biostats-model/assets")
beta = pd.read_csv(f"{assets_path}/Coefficients.csv").values

# the covariance matrix has a column that relates the coefficients but we don't need it here
v_beta = pd.read_csv(f"{assets_path}/Covariance matrix.csv")
v_beta = v_beta.drop(columns=['Unnamed: 0']).values

df_validate = pd.read_csv(f"{assets_path}/Validation.csv")
df_validate.set_index('psuedoID')
df_predict = df_validate.drop(columns=["phat", "LB", "UB"])
df_predict = df_predict.apply(predict_row, axis="columns")

df_validate = pd.merge(df_validate, df_predict[["my_phat", "my_lb", "my_ub"]], left_index=True, right_index=True)
df_validate['phat_delta'] = df_validate['phat'] - df_validate['my_phat']
df_validate['lb_delta'] = df_validate['LB'] - df_validate['my_lb']
df_validate['ub_delta'] = df_validate['UB'] - df_validate['my_ub']

df_error = df_validate.query("phat_delta > 1e-15 | lb_delta > 1e-15 | ub_delta > 1e-15")
print("Error count", len(df_error))

df_validate.to_csv(f"{assets_path}/done.csv")
