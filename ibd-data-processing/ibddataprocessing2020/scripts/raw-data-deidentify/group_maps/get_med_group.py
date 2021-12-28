from .meds_dict import meds_dict


def get_med_group(generic_name, med_name):
    if len(meds_dict[generic_name].med_generic_name) == 0 or med_name in meds_dict[generic_name].med_names:
        return meds_dict[generic_name].group

    return ''
