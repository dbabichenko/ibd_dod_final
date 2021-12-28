class Med:
    def __init__(self, generic_name, med_name_list, group_name):
        self.med_generic_name = generic_name
        self.med_names = med_name_list
        self.group = group_name


meds_dict = {}

# Anti TNF
med = Med('ADALIMUMAB', [], 'ANTI TNF')
meds_dict[med.med_generic_name] = med
med = Med('CERTOLIZUMAB PEGOL', [], 'ANTI TNF')
meds_dict[med.med_generic_name] = med
med = Med('ETANERCEPT', [], 'ANTI TNF')
meds_dict[med.med_generic_name] = med
med = Med('GOLIMUMAB', [], 'ANTI TNF')
meds_dict[med.med_generic_name] = med
med = Med('infliximab', [], 'ANTI TNF')
meds_dict[med.med_generic_name] = med


# ANTI INTEGRIN
med_names = [
    'ENTYVIO 300 MG INTRAVENOUS SOLUTION',
    'ENTYVIO IV',
    'NATALIZUMAB 300 MG/15 ML INTRAVENOUS SOLUTION',
    'TYSABRI 300 MG/15 ML INTRAVENOUS SOLUTION',
    'TYSABRI IV',
    'VEDOLIZUMAB (ENTYVIO) INFUSION',
    'VEDOLIZUMAB 300 MG INTRAVENOUS SOLUTION',
    'VEDOLIZUMAB IV'
]

med = Med('NATALIZUMAB', med_names, 'ANTI INTEGRIN')
meds_dict[med.med_generic_name] = med
med = Med('VEDOLIZUMAB', med_names, 'ANTI INTEGRIN')
meds_dict[med.med_generic_name] = med


# ANTI IL12
med_names = [
    'STELARA 45 MG/0.5 ML SUBCUTANEOUS SYRINGE',
    'STELARA SUBQ',
    'USTEKINUMAB (STELARA) INFUSION',
    'USTEKINUMAB 45 MG/0.5 ML SUBCUTANEOUS SOLUTION',
    'USTEKINUMAB 45 MG/0.5 ML SUBCUTANEOUS SYRINGE',
    'USTEKINUMAB 90 MG/ML SUBCUTANEOUS SYRINGE'
]

med = Med('USTEKINUMAB', med_names, 'ANTI IL12')
meds_dict[med.med_generic_name] = med

# 5 ASA
med = Med('MESALAMINE', [], '5 ASA')
meds_dict[med.med_generic_name] = med
med = Med('MESALAMINE W/CLEANSING WIPES', [], '5 ASA')
meds_dict[med.med_generic_name] = med
med = Med('SULFASALAZINE', [], '5 ASA')
meds_dict[med.med_generic_name] = med

# Immunomodulators
med = Med('MERCAPTOPURINE', [], 'Immunomodulators')
meds_dict[med.med_generic_name] = med
med = Med('SIMPLE_GENERIC_NAME', [], 'Immunomodulators')
meds_dict[med.med_generic_name] = med
med = Med('METHOTREXATE', [], 'Immunomodulators')
meds_dict[med.med_generic_name] = med
med = Med('METHOTREXATE SODIUM', [], 'Immunomodulators')
meds_dict[med.med_generic_name] = med
med = Med('METHOTREXATE SODIUM/PF', [], 'Immunomodulators')
meds_dict[med.med_generic_name] = med
med = Med('METHOTREXATE/PF', [], 'Immunomodulators')
meds_dict[med.med_generic_name] = med
med = Med('SIMPLE_GENERIC_NAME', [], 'Immunomodulators')
meds_dict[med.med_generic_name] = med
med = Med('AZATHIOPRINE', [], 'Immunomodulators')
meds_dict[med.med_generic_name] = med
med = Med('SIMPLE_GENERIC_NAME', [], 'Immunomodulators')
meds_dict[med.med_generic_name] = med
med = Med('LEFLUNOMIDE', [], 'Immunomodulators')
meds_dict[med.med_generic_name] = med

# Systemic steroids
med = Med('DEXAMETHASONE', [], 'Systemic steroids')
meds_dict[med.med_generic_name] = med
med = Med('DEXAMETHASONE SOD PHOSPHATE', [], 'Systemic steroids')
meds_dict[med.med_generic_name] = med
med = Med('HYDROCORTISONE', [], 'Systemic steroids')
meds_dict[med.med_generic_name] = med
med = Med('HYDROCORTISONE SOD PHOSPHATE', [], 'Systemic steroids')
meds_dict[med.med_generic_name] = med
med = Med('HYDROCORTISONE SOD SUCC/PF', [], 'Systemic steroids')
meds_dict[med.med_generic_name] = med
med = Med('HYDROCORTISONE SOD SUCCINATE', [], 'Systemic steroids')
meds_dict[med.med_generic_name] = med
med = Med('METHYLPREDNISOLONE', [], 'Systemic steroids')
meds_dict[med.med_generic_name] = med
med = Med('METHYLPREDNISOLONE ACETATE', [], 'Systemic steroids')
meds_dict[med.med_generic_name] = med
med = Med('METHYLPREDNISOLONE SOD SUCC', [], 'Systemic steroids')
meds_dict[med.med_generic_name] = med
med = Med('METHYLPREDNISOLONE SOD SUCC/PF', [], 'Systemic steroids')
meds_dict[med.med_generic_name] = med
med = Med('PREDNISOLONE', [], 'Systemic steroids')
meds_dict[med.med_generic_name] = med
med = Med('PREDNISOLONE SOD PHOSPHATE', [], 'Systemic steroids')
meds_dict[med.med_generic_name] = med
med = Med('PREDNISONE', [], 'Systemic steroids')
meds_dict[med.med_generic_name] = med


# Vitamin D
med = Med('CALCIUM CARBONATE/VITAMIN D3', [], 'Vitamin D')
meds_dict[med.med_generic_name] = med
med = Med('CHOLECALCIFEROL (VITAMIN D3)', [], 'Vitamin D')
meds_dict[med.med_generic_name] = med
med = Med('ERGOCALCIFEROL (VITAMIN D2)', [], 'Vitamin D')
meds_dict[med.med_generic_name] = med
med = Med('SIMPLE_GENERIC_NAME', [], 'Vitamin D')
meds_dict[med.med_generic_name] = med
