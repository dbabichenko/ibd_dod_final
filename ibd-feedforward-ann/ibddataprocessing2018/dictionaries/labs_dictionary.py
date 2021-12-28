class Lab:
    def __init__(self, comp_name, proc_name_list, group_name):
        self.lab_comp_name = comp_name
        self.proc_names = proc_name_list
        self.group = group_name

labs_dict = {}

# CRP
lab = Lab('C-REACTIVE PROTEIN', [], 'crp')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('HIGH SENSITIVITY CRP', [], 'crp')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('CRP QUANTITATION', [], 'crp')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('C-REACTIVE PROTEIN', [], 'crp')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('CARDIO CRP', [], 'crp')
labs_dict[lab.lab_comp_name] = lab



# ESR
lab = Lab('SEDRATE - AUTOMATED', [], 'esr')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('ESR', [], 'esr')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('WESTERGREN ESR', [], 'esr')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('WESTERGREN SED RATE', [], 'esr')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('SED RATE', [], 'esr')
labs_dict[lab.lab_comp_name] = lab

# Albumin
proc_albumin = [
'COMP METABOLIC PANEL',
'RENAL FUNCTION PANEL (OO)',
'HEPATIC FUNCTION PANEL',
'ALBUMIN',
'ALBUMIN (OO)',
'LIVER TRANSPLANT POST-OP LONG LAB SET',
'HILLMAN COMPREHENSIVE COMP PANEL',
'COMP METABOLIC PANEL W/EGFR',
'TOTAL PROTEIN ELECTROPHORESIS, SERUM',
'PROTEIN ELECTROPHORESIS, SERUM',
'ADD ON HEPATIC FUNCTION PANEL',
'CMP & GFR(QUEST ONLY)',
'COMMUNITY HEALTH SCREEN W/LDL',
'SMALL BOWEL TRANSPLANT POST-OP LAB SET',
'RENAL TRANSPLANT POST-OP LONG LAB SET',
'HEPATORENAL PANEL',
'TOTAL PROTEIN ELECT. (OO)',
'SERUM PROTEIN ELEC WITH GRAPH',
'CHEM-SCREEN PANEL',
'AUTOLOGOUS LYMPHOMA',
'LIVER TRANSPLANT POST-OP SHORT LAB SET',
'CMP W/EGFR',
'CHEM-SCREEN PNL+HDL,TIBC (QUEST)',
'MULTIPLE MYELOMA PANEL',
'PROTEIN ELECTRO/TPROT W/RFX, SERUM',
'COMP METABOLIC PANEL W/O EGFR',
'TESTOSTERONE,FREE,BIOAVAIL/TOT',
'BASIC METABOLIC PANEL',
'HEPATIC FUNCTION PANEL WITHOUT TOTAL PROTEIN']

lab = Lab('ALBUMIN', proc_albumin, 'albumin')
labs_dict[lab.lab_comp_name] = lab


# Vitamin D
lab = Lab('VITAMIN D, 25-HYDROXY', [], 'vitamin_d')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('25-HYDROXYVITAMIN D', [], 'vitamin_d')
labs_dict[lab.lab_comp_name] = lab

# Hemoglobin
lab = Lab('HGB', [], 'hemoglobin')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('HEMOGLOBIN-PLASMA', [], 'hemoglobin')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('HEMOGLOBIN-ARTERIAL', [], 'hemoglobin')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('BEDSIDE HEMOGLOBIN POCT', [], 'hemoglobin')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('HEMOGLOBIN - MIXED VENOUS', [], 'hemoglobin')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('TOTAL HGB(VENOUS)', [], 'hemoglobin')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('HEMOGLOBIN-VENOUS &&', [], 'hemoglobin')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('TOTAL HEMOGLOBIN', [], 'hemoglobin')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('HEMOGLOBIN (POCT)', [], 'hemoglobin')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('HEMOGLOBIN CAPILLARY (POC)', [], 'hemoglobin')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('HEMOGLOBIN ISTAT', [], 'hemoglobin')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('HEMOGLOBIN', [], 'hemoglobin')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('THB (HEMOGLOBIN)', [], 'hemoglobin')
labs_dict[lab.lab_comp_name] = lab


# Eosinophils (EOS)
proc_eos = [
'CBC & DIFF INC PLATELET',
'CBC AND DIFF W/ PLATELETS',
'MANUAL DIFFERENTIAL',
'LIVER TRANSPLANT POST-OP LONG LAB SET',
'LIVER TRANSPLANT POST-OP SHORT LAB SET',
'OBSTETRIC PANEL',
'DIFFERENTIAL (MANUAL)',
'VISUAL DIFFERENTIAL',
'DIFFERENTIAL',
'CBC/MANUAL DIFF-LAB USE ONLY',
'OBSTETRIC PANEL W/HIV',
'DIFFERENTIAL (AUTO)',
'OBSTETRIC PANEL (MAGEE ONLY)',
'SMALL BOWEL TRANSPLANT POST-OP LAB SET',
'OBSTETRIC PANEL W/REFLEX',
'CBC/AUTO DIFF, ONCOLOGY',
'CBC/DIFF AMBIGIOUS DEFAULT',
'CBC AND DIFF W/ PLATELET',
'CBC WITHOUT DIFF W/ PLATELETS',
'CBC AND DIFF W/O PLATELETS',
'RENAL TRANSPLANT POST-OP LONG LAB SET',
'RENAL TRANSPLANT POST-OP SHORT LAB SET',
'PRENATAL PANEL',
'CBC AND DIFF',
'MULTIPLE MYELOMA PANEL',
'COMMUNITY HEALTH SCREEN W/LDL',
'DIFF ADD ONTO CBCP',
'AUTOLOGOUS LYMPHOMA'
]

lab = Lab('ABS EOSINOPHILS', proc_eos, 'eos')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('EOSINOPHILS', proc_eos, 'eos')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('AUTO. ABSOL. EOSIN', proc_eos, 'eos')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('AUTOMATED EOSIN %', proc_eos, 'eos')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('ABSOLUTE EOS', proc_eos, 'eos')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('EOS %', proc_eos, 'eos')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('AUTO ABSOL EOSIN', proc_eos, 'eos')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('AUTOMATED EOSIN%', proc_eos, 'eos')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('ABS EOS', proc_eos, 'eos')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('EOSINOPHIL', proc_eos, 'eos')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('EOS#', proc_eos, 'eos')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('ABSOLUTE EOSINOPHILS', proc_eos, 'eos')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('EOSINOPHILS, ABS', proc_eos, 'eos')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('EOS', proc_eos, 'eos')
labs_dict[lab.lab_comp_name] = lab


# Monocytes
proc_monocytes = [
'CBC & DIFF INC PLATELET',
'CBC AND DIFF W/ PLATELETS',
'MANUAL DIFFERENTIAL',
'LIVER TRANSPLANT POST-OP LONG LAB SET',
'LIVER TRANSPLANT POST-OP SHORT LAB SET',
'OBSTETRIC PANEL',
'DIFFERENTIAL (MANUAL)',
'VISUAL DIFFERENTIAL',
'DIFFERENTIAL',
'CBC/MANUAL DIFF-LAB USE ONLY',
'OBSTETRIC PANEL W/HIV',
'DIFFERENTIAL (AUTO)',
'OBSTETRIC PANEL (MAGEE ONLY)',
'CBC WITHOUT DIFF W/ PLATELETS',
'SMALL BOWEL TRANSPLANT POST-OP LAB SET',
'VISUAL DIFF ONCOLOGY',
'OBSTETRIC PANEL W/REFLEX',
'CBC/AUTO DIFF, ONCOLOGY',
'CBC/DIFF AMBIGIOUS DEFAULT',
'CBC AND DIFF W/ PLATELET',
'CBC AND DIFF W/O PLATELETS',
'RENAL TRANSPLANT POST-OP LONG LAB SET',
'RENAL TRANSPLANT POST-OP SHORT LAB SET',
'CBC, ONCOLOGY',
'PRENATAL PANEL',
'CBC AND DIFF',
'MULTIPLE MYELOMA PANEL',
'COMMUNITY HEALTH SCREEN W/LDL',
'DIFF ADD ONTO CBCP',
'AUTOLOGOUS LYMPHOMA'
]

lab = Lab('ABS MONOCYTES', proc_monocytes, 'monocytes')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('MONOCYTES', proc_monocytes, 'monocytes')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('AUTO. ABSOL. MONOS', proc_monocytes, 'monocytes')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('AUTOMATED MONOS %', proc_monocytes, 'monocytes')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('MONOS %', proc_monocytes, 'monocytes')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('AUTO ABSOL MONOS', proc_monocytes, 'monocytes')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('AUTOMATED MONO%', proc_monocytes, 'monocytes')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('MONOCYTE', proc_monocytes, 'monocytes')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('ABS MONO', proc_monocytes, 'monocytes')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('MONO#', proc_monocytes, 'monocytes')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('MONOCYTES (%) @', proc_monocytes, 'monocytes')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('ABSOLUTE MONOCYTES', proc_monocytes, 'monocytes')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('ABS MONOCYTE', proc_monocytes, 'monocytes')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('MONO', proc_monocytes, 'monocytes')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('MONOCYTES @', proc_monocytes, 'monocytes')
labs_dict[lab.lab_comp_name] = lab
lab = Lab('MONOCYTES, ABS', proc_monocytes, 'monocytes')
labs_dict[lab.lab_comp_name] = lab