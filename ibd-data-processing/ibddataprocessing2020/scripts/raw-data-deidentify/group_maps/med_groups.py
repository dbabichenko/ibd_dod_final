from .meds_dict import meds_dict


def get_is_ibd_med(generic_name):
    return generic_name in meds_dict
