from os import path
import os
import pandas as pd
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from import_allergies import import_allergies
from import_encounters import import_encounters
from import_labs import import_labs
from import_meds import import_meds
from import_problems import import_problems
from import_utils import truncate_table, delete_where

# ========================================================================================
# GLOBAL VARIABLES
# ========================================================================================
# so i can develop one import at a time
do = [
    'allergies',
    'encounters',
    'labs',
    'meds',
    'problems'
]
root_dir = path.join(os.getcwd(), 'ibddataprocessing2020\\scripts\\openemr-import\\data')
patient_ids = [2265, 2863]

# ========================================================================================
# Prep sql alchemy
# ========================================================================================
Base = automap_base()
engine_url = f"mysql+pymysql://{os.getenv('IBD_DOD_MYSQL_USER')}:{os.getenv('IBD_DOD_MYSQL_PASSWORD')}@134.209.169.96/openemr"
engine = create_engine(engine_url)

# reflect the tables
Base.prepare(engine, reflect=True)

# ========================================================================================
# ENCOUNTERS
# ========================================================================================
if 'encounters' in do:
    print("Importing encounters...")
    truncate_table(engine, 'form_encounter')
    df_enc = import_encounters(path.join(root_dir, 'encounters_merged.xlsx'), patient_ids)
    df_enc.to_sql('form_encounter', engine, index=False, if_exists='append')

# ========================================================================================
# LABS
# ========================================================================================
if 'labs' in do:
    print("Importing labs...")
    df_labs = import_labs(path.join(root_dir), 'labs_merged.xlsx', patient_ids)
    print(df_labs)

# ========================================================================================
# MEDICATIONS
# ========================================================================================
if 'meds' in do:
    print("Importing meds...")
    delete_where(engine, 'lists', 'type = "medication"')
    df_meds = import_meds(path.join(root_dir, 'meds_merged.xlsx'), patient_ids)
    df_meds.to_sql('lists', engine, index=False, if_exists='append')

# ========================================================================================
# PROBLEMS
# ========================================================================================
if 'problems' in do:
    print("Importing problems...")
    delete_where(engine, 'lists', 'type = "medical_problem"')
    df_probs = import_problems(path.join(root_dir, 'problem_list_merged.xlsx'), patient_ids)
    df_probs.to_sql('lists', engine, index=False, if_exists='append')

# ========================================================================================
# ALLERGIES
# ========================================================================================
if 'allergies' in do:
    print("Importing allergies...")
    delete_where(engine, 'lists', 'type = "allergy"')
    df_allergies = import_allergies(path.join(root_dir, 'allergies.xlsx'), patient_ids)
    df_allergies.to_sql('lists', engine, index=False, if_exists='append')
