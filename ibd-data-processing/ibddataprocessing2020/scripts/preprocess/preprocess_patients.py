import datetime
from sklearn.preprocessing import LabelEncoder


def get_age_bucket(age):
    if age < 18:
        return 0
    if age < 35:
        return 1
    if age < 51:
        return 2
    if age < 70:
        return 3
    return 4


def preprocess_patients(df_patients):
    # create age
    df_patients["AGE"] = df_patients["BIRTH_YEAR"].subtract(datetime.datetime.now().year).abs()
    df_patients["AGE LE"] = df_patients["AGE"].apply(get_age_bucket)

    # standardize gender attributes
    df_patients["GENDER"] = df_patients["GENDER"].str[:1].str.lower()

    # marital statuses condensed to married/single/unknown (missing is unknown)
    df_patients["MARITAL STATUS"].fillna("Unknown", inplace=True)
    df_patients["MARITAL STATUS"] = df_patients["MARITAL STATUS"].replace(
        ["Divorced", "Widowed", "Legally Separated", "Significant Other"], "Single")
    df_patients["MARITAL STATUS"] = df_patients["MARITAL STATUS"].replace("", "Unknown")
    df_patients["MARITAL STATUS"] = df_patients["MARITAL STATUS"].replace(
        ["Committed relationship", "Married"], "Married")

    # employment statuses condensed to employed/not employed/student/unknown (missing is unknown)
    df_patients["EMPLOYMENT_STATUS"].fillna("Unknown", inplace=True)
    df_patients["EMPLOYMENT_STATUS"].replace(["Full Time", "Part Time", "Self Employed"], "Employed")
    df_patients["EMPLOYMENT_STATUS"].replace(["Retired", "Not Employed"], "Not Employed")
    df_patients["EMPLOYMENT_STATUS"].replace(["Student - Full Time", "Student - Part Time"], "Student")

    # label encode other relevant columns
    df_patients["GENDER"] = df_patients["GENDER"].astype("category").cat.codes
    df_patients["MARITAL STATUS LE"] = df_patients["MARITAL STATUS"].astype("category").cat.codes
    df_patients["EMPLOYMENT_STATUS_LE"] = df_patients["EMPLOYMENT_STATUS"].astype("category").cat.codes
    df_patients["RACE"] = df_patients["RACE"].astype("category").cat.codes
    df_patients["ETHNIC_GROUP"] = df_patients["ETHNIC_GROUP"].astype("category").cat.codes
    df_patients["IS_ALIVE"] = df_patients["IS_ALIVE"].astype("category").cat.codes

    # drop unnecessary columns
    df_patients = df_patients.drop(columns=[
        "AGE",
        "BIRTH_YEAR",
        "DATA_SOURCE",
        "EMPLOYMENT_STATUS",
        "MARITAL STATUS"
    ])

    return df_patients
