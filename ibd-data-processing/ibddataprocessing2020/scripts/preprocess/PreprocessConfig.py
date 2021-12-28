from enum import Enum, auto


class PreprocessConfig:
    def __init__(self, config):
        self.config = config

    def is_enabled(self, key):
        if self.config is None:
            return True

        return key in self.config


class ConfigKey(Enum):
    ENCOUNTERS = auto()
    LABS = auto()
    MEDS = auto()
    PROBLEMS = auto()
    PROCEDURES = auto()
    TARGET = auto()
    TOBACCO = auto()
    VITALS = auto()
