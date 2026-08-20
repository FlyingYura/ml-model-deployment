import pandas as pd
import numpy as np
from sqlalchemy import text
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
from datetime import datetime
from ..database.database import get_engine 
from ..database.config import MODEL_CONFIG, FEATURE_COLS, EXCLUDE_COLS 
import json


def train_model():

    engine = get_engine()
    
    print("Завантаження даних з БД")
    df = pd.read_sql_table('asthma_patients_data', engine)
    print(f"Завантажено {len(df)} записів")
    
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS]
    X = df[feature_cols]
    y = df['diagnosis'].astype(int)
    
    X_train, X_new_input, y_train, y_new_input = train_test_split(
        X, y,
        test_size=0.1,
        random_state=42,
        stratify=y
    )
    
    print(f"Розмір тренувальної вибірки: {X_train.shape}")
    print(f"Розмір нової вибірки: {X_new_input.shape}")
    
    print("Тренування моделі")
    model = LogisticRegression(
        **MODEL_CONFIG['best_params'],
        random_state=MODEL_CONFIG['random_state'],
        max_iter=1000
    )
    
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    
    metrics = {
        'accuracy': float(accuracy_score(y_train, y_train_pred)),
        'precision': float(precision_score(y_train, y_train_pred, zero_division=0)),
        'recall': float(recall_score(y_train, y_train_pred, zero_division=0)),
        'f1_score': float(f1_score(y_train, y_train_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_train, y_train_proba))
    }
    
    os.makedirs(MODEL_CONFIG['model_dir'], exist_ok=True)
    model_filename = f"{MODEL_CONFIG['model_dir']}/model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    joblib.dump(model, model_filename)
    
    latest_model_filename = f"{MODEL_CONFIG['model_dir']}/model_latest.pkl"
    joblib.dump(model, latest_model_filename)
    
    metadata = {
        'model_type': MODEL_CONFIG['model_type'],
        'best_params': MODEL_CONFIG['best_params'],
        'train_metrics': metrics,
        'feature_names': feature_cols,
        'saved_at': datetime.now().isoformat()
    }
    metadata_filename = f"{MODEL_CONFIG['model_dir']}/model_latest_metadata.pkl"
    joblib.dump(metadata, metadata_filename)
    
    train_ids = df.loc[X_train.index, 'patientid'].values
    predictions_df = pd.DataFrame({
        'patientid': train_ids,
        'true_label': y_train.values,
        'predicted_label': y_train_pred,
        'predicted_proba': y_train_proba,
        'model_name': MODEL_CONFIG['model_type'],
        'source': 'train'
    })
    
    predictions_df.to_sql('predictions', engine, if_exists='append', index=False)
    print(f"Збережено {len(predictions_df)} передбачень у БД")
    
    metrics_df = pd.DataFrame([{
        'model_name': MODEL_CONFIG['model_type'],
        'dataset_type': 'train',
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'roc_auc': metrics['roc_auc'],
        'hyperparameters': json.dumps(MODEL_CONFIG['best_params']),
        'optimized': True
    }])
    
    metrics_df.to_sql('model_metrics', engine, if_exists='append', index=False)
    print("Метрики збережено у БД")
    
    return {
        'status': 'success',
        'model_file': model_filename,
        'metrics': metrics,
        'train_size': len(X_train),
        'new_input_size': len(X_new_input)
    }

