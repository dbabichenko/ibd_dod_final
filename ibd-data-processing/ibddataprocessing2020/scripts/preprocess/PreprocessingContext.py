import os


class PreprocessingContext:
    def __init__(self, config, files, window_lengths, window_mode, working_directory):
        self.config = config
        self.files = files
        self.window_lengths = window_lengths
        self.window_mode = window_mode
        self.working_directory = working_directory
