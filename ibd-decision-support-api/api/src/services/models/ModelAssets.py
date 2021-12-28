class ModelAssets:
    coefficients = None
    covariance_matrix = None

    def __init__(self, coefficients=None, covariance_matrix=None):
        self.coefficients = coefficients
        self.covariance_matrix = covariance_matrix
