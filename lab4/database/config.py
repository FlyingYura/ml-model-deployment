import os

DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'breathwell_clinic'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),
    'port': os.getenv('DB_PORT', '5432')
}

MODEL_CONFIG = {
    'model_type': 'LogisticRegression',
    'best_params': {
        'solver': 'lbfgs',
        'penalty': 'l2',
        'class_weight': 'balanced',
        'C': 0.001
    },
    'model_dir': 'lab4/models/',
    'random_state': 42
}

FEATURE_COLS = [
    'age', 'gender', 'bmi', 'smoking', 'physicalactivity', 'dietquality',
    'sleepquality', 'pollutionexposure', 'pollenexposure', 'dustexposure',
    'wheezing', 'shortnessofbreath', 'chesttightness', 'coughing',
    'nasalcongestion', 'gastroesophagealreflux', 'lungfunctionfev1',
    'lungfunctionfvc', 'allergichistory', 'familyhistory', 'medicationuse',
    'exerciseinduced', 'ethnicity_1', 'ethnicity_2', 'ethnicity_3',
    'educationlevel_1', 'educationlevel_2', 'educationlevel_3'
]

EXCLUDE_COLS = ['patientid', 'timestamp', 'diagnosis', 'doctorincharge']

