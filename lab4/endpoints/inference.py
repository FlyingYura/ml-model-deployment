"""
Модуль для передбачення через FastAPI
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from ..database.database import get_engine  
from ..database.config import MODEL_CONFIG, FEATURE_COLS, EXCLUDE_COLS
import json


def load_model():
    model_path = f"{MODEL_CONFIG['model_dir']}/model_latest.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не знайдено: {model_path}")
    
    model = joblib.load(model_path)
    return model


def load_metadata():
    metadata_path = f"{MODEL_CONFIG['model_dir']}/model_latest_metadata.pkl"
    if not os.path.exists(metadata_path):
        return None
    
    metadata = joblib.load(metadata_path)
    return metadata


def predict(input_data: dict, patientid: int = None):

    engine = get_engine()
    
    model = load_model()
    metadata = load_metadata()
    
    if metadata is None:
        feature_cols = FEATURE_COLS
    else:
        feature_cols = metadata['feature_names']
    
    # Перетворення вхідних даних у DataFrame
    missing_features = set(feature_cols) - set(input_data.keys())
    if missing_features:
        raise ValueError(f"Відсутні необхідні поля: {missing_features}")
    
    input_df = pd.DataFrame([input_data])[feature_cols]
    

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0, 1]
    
    if patientid is None:
        patientid = input_data.get('patientid', None)
    
    input_data_for_db = {k: v for k, v in input_data.items() if k != 'patientid'}
    
    inference_inputs_df = pd.DataFrame([{
        'patientid': patientid,
        'input_data': json.dumps(input_data_for_db),
        'model_name': MODEL_CONFIG['model_type']
    }])
    
    inference_inputs_df.to_sql('inference_inputs', engine, if_exists='append', index=False)
    
    predictions_df = pd.DataFrame([{
        'patientid': patientid,
        'true_label': None,  
        'predicted_label': int(prediction),
        'predicted_proba': float(probability),
        'model_name': MODEL_CONFIG['model_type'],
        'source': 'inference'
    }])
    
    predictions_df.to_sql('predictions', engine, if_exists='append', index=False)
    
    return {
        'prediction': int(prediction),
        'probability': float(probability),
        'patientid': patientid,
        'model_name': MODEL_CONFIG['model_type']
    }


def predict_batch(input_data_list: list):

    results = []
    for data in input_data_list:
        try:
            result = predict(data, data.get('patientid'))
            results.append(result)
        except Exception as e:
            results.append({
                'error': str(e),
                'patientid': data.get('patientid')
            })
    
    return results

