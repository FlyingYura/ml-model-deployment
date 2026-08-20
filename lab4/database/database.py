from sqlalchemy import create_engine, text
from lab4.database.config import DATABASE_CONFIG


def get_connection_string():
    return (
        f"postgresql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}"
        f"@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}"
        f"/{DATABASE_CONFIG['database']}"
    )


def get_engine():
    connection_string = get_connection_string()
    return create_engine(connection_string)


def init_tables():
    try:
        engine = get_engine()
        
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    patientid INTEGER,
                    true_label INTEGER,
                    predicted_label INTEGER,
                    predicted_proba FLOAT,
                    model_name VARCHAR(255),
                    source VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS inference_inputs (
                    id SERIAL PRIMARY KEY,
                    patientid INTEGER,
                    input_data JSONB,
                    model_name VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(255),
                    dataset_type VARCHAR(50),
                    accuracy FLOAT,
                    precision FLOAT,
                    recall FLOAT,
                    f1_score FLOAT,
                    roc_auc FLOAT,
                    hyperparameters TEXT,
                    optimized BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
        
        print("Таблиці успішно ініціалізовано")
        return True
    except Exception as e:
        error_msg = str(e)
        print(f"Помилка ініціалізації таблиць: {error_msg}")
        return False

