from enum import Enum, auto


class PreprocessingWindowMode(Enum):
    # given a prediction window size and a target window size,
    # predict DVs in the target window from the IVs in the prediction window
    FIXED_LENGTH = auto()

    # predict DVs for the patient's last year based on the
    ONE_WINDOW_PER_PATIENT = auto()

    # predict DVs for the current year from the IVs for the current year
    ONE_WINDOW_PER_PATIENT_YEAR = auto()
