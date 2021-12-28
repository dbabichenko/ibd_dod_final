class FormParserService:
    def get_bool(self, form, key):
        return form.get(key) == 'on'

    def get_float(self, form, key):
        value = form.get(key)

        if value is None or value == '' or not value.isnumeric():
            return 0.0

        return float(value)

    def get_int(self, form, key):
        value = form.get(key)

        if value is None or value == '' or not value.isnumeric():
            return 0

        return int(value)

    def bools_to_ints(self, parsed):
        out = []

        for value in parsed:
            if not isinstance(value, bool):
                out.append(value)
            else:
                out.append(1 if value else 0)

        return out

    def parseCdForm(self, form):
        parsed = [
            not self.get_bool(form, 'isEmployed'),  # 'unemployed': 0,
            not self.get_bool(form, 'isPartnered'),  # 'notMarried': 1,
            self.get_bool(form, 'is50plus'),  # 'age50': 2,
            self.get_bool(form, 'isTobaccoUser'),  # 'tobaccoEver': 3,
            self.get_bool(form, 'isAlcoholUser'),  # 'alcohol2yes': 4,
            self.get_bool(form, 'isPsychologicalPatient'),  # 'psyche': 5,
            form.get('diseaseDuration') == '0-15',  # 'durationGroup0-15': 6,
            form.get('diseaseDuration') == '16-25',  # 'durationGroup15-25': 7,
            form.get('diseaseDuration') == '25+',  # 'durationGroup25+': 8,
            self.get_float(form, 'avgAnnualHospitalizations'),  # 'Discharge.Summary_ave3': 9,
            self.get_bool(form, 'isERAdmitted'),  # 'ER.Report_bin': 10,
            form.get('noPhoneCalls') == '0',  # 'TelephoneGroup0': 11,
            form.get('noPhoneCalls') == '1-10',  # 'TelephoneGroup0-10': 12,
            form.get('noPhoneCalls') == '11+',  # 'TelephoneGroup10+': 13,
            self.get_bool(form, 'isPrescribedImmunomodulators'),  # 'Immunomodulators_bin': 14,
            self.get_bool(form, 'isPrescribedNarcotics'),  # 'narcotics_bin3': 15,
            self.get_bool(form, 'isPrescribedSteroids'),  # 'Systemic_steroids_bin': 16,
            form.get('noSteroidsYears') == '0',  # 'SystemicSteroids3Group0': 17,
            form.get('noSteroidsYears') == '1-2',  # 'SystemicSteroids3Group0-2': 18,
            form.get('noSteroidsYears') == '3',  # 'SystemicSteroids3Group2+': 19,
            form.get('noLabHemoglobinYears') == '0',  # 'LabHemoglobinGroup0': 20,
            form.get('noLabHemoglobinYears') == '1-2',  # 'LabHemoglobinGroup(0-2]': 21,
            form.get('noLabHemoglobinYears') == '3',  # 'LabHemoglobinGroup[2,': 22,
            form.get('noLabEosYears') == '0',  # 'LabEOS0': 23,
            form.get('noLabEosYears') == '0' == '1',  # 'LabEOS1': 24,
            form.get('noLabEosYears') == '0' == '2' or form.get('noLabEosYears') == '0' == '3',  # 'LabEOS2+': 25,
            self.get_int(form, 'noAbnormalMonocytes') / 3,  # 'LabMonocytes_ave3': 26,
            form.get('noLabMonocytesYears') == '3',  # 'LabMonocytes_bin3': 27,
            self.get_int(form, 'noAbnormalMonocytes') > 2,  # 'LabMonocytesNoAbnormalTests>2': 28,
            self.get_int(form, 'noLabEosYears') > 0,  # 'Lab_EOS_bin3': 29,
            self.get_bool(form, 'isCurrentYearNormalAlbumin'),  # 'Step2Albumin': 30,
            self.get_bool(form, 'isCurrentYearNormalEos'),  # 'Step2EOS': 31,
            self.get_bool(form, 'isCurrentYearNormalHemoglobin'),  # 'Step2Hemoglobin': 32,
            self.get_bool(form, 'isCurrentYearNormalMonocytes'),  # 'Step2Monocytes': 33
        ]

        return self.bools_to_ints(parsed)

    def parseUcForm(self, form):
        disease_duration = self.get_int(form, 'diseaseDuration')
        no_phone_calls = self.get_int(form, 'noPhoneCalls')
        no_years_prescribed_steroids = self.get_int(form, 'noYearsSteroids')

        parsed = [
            self.get_int(form, 'age'),  # 'age': 0,
            self.get_bool(form, 'isEmployed'),  # 'unemployed': 1,
            self.get_bool(form, 'isTobaccoUser'),  # 'tobaccoEver': 2,
            self.get_bool(form, 'isFamilyHistoryOfIbd'),  # 'family_hx': 3,
            disease_duration,  # 'duration': 4,
            disease_duration >= 5 and disease_duration < 25,  # 'duration_cat_5-25': 5,
            disease_duration > 25,  # 'duration_cat_25+': 6,
            self.get_bool(form, 'isHospitalized'),  # 'Discharge.Summary_bin': 7,
            self.get_bool(form, 'isHospitalizedInLast3Years'),  # 'Discharge.Summary_bin3': 8,
            self.get_bool(form, 'isERAdmitted'),  # 'ER.Report_bin': 9,
            self.get_bool(form, 'isMultipleOfficeVisits'),  # 'office_visit_bin': 10,
            no_phone_calls >= 1 and no_phone_calls < 10,  # 'Telephone_cat_1-10': 11,
            no_phone_calls >= 10,  # 'Telephone_cat_10+': 12,
            self.get_bool(form, 'isColectomy'),  # 'colectomy_bin3': 13,
            self.get_bool(form, 'isPrescribedImmunomodulators'),  # 'rx_immunomodulators_bin': 14,
            self.get_bool(form, 'isPrescribedNarcotics'),  # 'rx_narcotics_bin3': 15,
            no_years_prescribed_steroids > 0,  # 'rx_systemic_steroids_bin': 16,
            no_years_prescribed_steroids == 0,  # 'rx_systemic_steroids_years_0-1': 17,
            no_years_prescribed_steroids == 1,  # 'rx_systemic_steroids_years_1-2': 18,
            no_years_prescribed_steroids >= 2,  # 'rx_systemic_steroids_years_2+': 19,
            self.get_bool(form, 'isAbnormalAlbumin'),  # 'lab_albumin_bin3': 20,
            self.get_bool(form, 'isAbnormalHemoglobin'),  # 'lab_hemoglobin_bin3': 21,
            self.get_bool(form, 'isAbnormalEos'),  # 'lab_eos_bin3': 22,
            self.get_bool(form, 'isAbnormalMonocytes'),  # 'lab_monocytes_bin3': 23,
            self.get_int(form, 'noAbnormalMonocytes'),  # 'lab_monocytes_years': 24,
            self.get_bool(form, 'isCurrentYearNormalAlbumin'),  # 'step2_albumin_bin': 26,
            self.get_bool(form, 'isCurrentYearNormalEos'),  # 'step2_eos_bin': 27,
            self.get_bool(form, 'isCurrentYearNormalHemoglobin'),  # 'step2_hemoglobin_bin': 28,
            self.get_bool(form, 'isCurrentYearNormalMonocytes'),  # 'step2_monocytes_bin': 29,
        ]

        return self.bools_to_ints(parsed)
