from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import lab4.endpoints.training as training
import lab4.endpoints.inference as inference
from lab4.database.database import init_tables
import subprocess
import os
import time

app = FastAPI(
    title="Asthma Prediction API",
    description="API для тренування та передбачення моделі діагностики астми",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    patientid: Optional[int] = None
    age: float
    gender: float
    bmi: float
    smoking: float
    physicalactivity: float
    dietquality: float
    sleepquality: float
    pollutionexposure: float
    pollenexposure: float
    dustexposure: float
    wheezing: float
    shortnessofbreath: float
    chesttightness: float
    coughing: float
    nasalcongestion: float
    gastroesophagealreflux: float
    lungfunctionfev1: float
    lungfunctionfvc: float
    allergichistory: Optional[float] = None
    historyofallergies: Optional[float] = None
    familyhistory: Optional[float] = None
    familyhistoryasthma: Optional[float] = None
    nighttimesymptoms: Optional[float] = None
    eczema: Optional[float] = None
    petallergy: Optional[float] = None
    hayfever: Optional[float] = None
    medicationuse: float
    exerciseinduced: float
    ethnicity_1: float
    ethnicity_2: float
    ethnicity_3: float
    educationlevel_1: float
    educationlevel_2: float
    educationlevel_3: float


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    patientid: Optional[int]
    model_name: str


class TrainingResponse(BaseModel):
    status: str
    model_file: str
    metrics: Dict[str, float]
    train_size: int
    new_input_size: int


def normalize_field_names(input_data: dict) -> dict:

    field_mapping = {
        "allergichistory": "historyofallergies",
        "familyhistory": "familyhistoryasthma",
    }
    
    normalized = {}
    
    for key, value in input_data.items():
        if key in field_mapping:
            new_key = field_mapping[key]
            if new_key not in input_data:
                normalized[new_key] = value
        else:
            normalized[key] = value
    
    required_fields = {
        "historyofallergies": 0.0,
        "familyhistoryasthma": 0.0,
        "nighttimesymptoms": 0.0,
        "eczema": 0.0,
        "petallergy": 0.0,
        "hayfever": 0.0
    }
    
    for field, default_value in required_fields.items():
        if field not in normalized:
            normalized[field] = default_value
    
    return normalized


@app.on_event("startup")
async def startup_event():
    success = init_tables()
    if not success:
        print("не вдалося ініціалізувати таблиці")
    print("FastAPI додаток запущено")


@app.get("/")
async def root():
    return {
        "message": "Asthma Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "train": "/train-model",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "monitor": "/monitor",
            "docs": "/docs"
        }
    }


@app.post("/train-model", response_model=TrainingResponse)
async def train_model():

    try:
        result = training.train_model()
        return TrainingResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):

    try:
        input_data = request.dict()
        
        input_data = {k: v for k, v in input_data.items() if v is not None}

        patientid = input_data.pop('patientid', None)
        
        input_data = normalize_field_names(input_data)
        
        result = inference.predict(input_data, patientid)
        return PredictionResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404, 
            detail=f"Модель не знайдено. Спочатку потрібно навчити модель через /train-model. Помилка: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Внутрішня помилка: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(requests: List[PredictionRequest]):

    try:
        results = []
        for req in requests:
            try:
                input_data = req.dict()
                
                input_data = {k: v for k, v in input_data.items() if v is not None}
                
                patientid = input_data.pop('patientid', None)
                
                input_data = normalize_field_names(input_data)
                
                result = inference.predict(input_data, patientid)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'patientid': req.patientid
                })
        
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/monitor")
async def monitor():
  
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        monitoring_script = os.path.join(current_dir, "lab5_monitoring.py")
        
        print(f"Шукаємо файл: {monitoring_script}")
        
        if not os.path.exists(monitoring_script):
            return HTMLResponse(content=f"""
            <html>
            <head><title>Помилка</title></head>
            <body style="font-family: Arial; padding: 40px; text-align: center;">
                <h1>Файл lab5_monitoring.py не знайдено</h1>
                <p>Шлях: {monitoring_script}</p>
                <a href="/">← Повернутись до API</a>
            </body>
            </html>
            """)
        
        evidently_env_python = r"F:\jupyter\gulp\lab4\evidently_env\Scripts\python.exe"
        
        if not os.path.exists(evidently_env_python):
            possible_python_paths = [
                evidently_env_python,
                r"F:\jupyter\gulp\lab4\evidently_env\Scripts\python",
                r"F:\jupyter\gulp\lab4\evidently_env\bin\python",
                # Фолбеки
                r"F:\jupyter\gulp\venv\Scripts\python.exe",
                r".\venv\Scripts\python.exe",
                "python3.12",
                "python3",
                "python"
            ]
            
            python_path = None
            for path in possible_python_paths:
                try:
                    if not os.path.isabs(path):
                        abs_path = os.path.join(current_dir, path)
                        if os.path.exists(abs_path):
                            python_path = abs_path
                            break
                    elif os.path.exists(path):
                        python_path = path
                        break
                    
                    result = subprocess.run([path, "--version"], capture_output=True, text=True)
                    if result.returncode == 0:
                        python_path = path
                        break
                except:
                    continue
            
            if not python_path:
                python_path = "python"
        else:
            python_path = evidently_env_python
        
        print(f"Використовуємо Python: {python_path}")
        
        try:
            check_result = subprocess.run(
                [python_path, "-c", "import evidently; print(f'Evidently version: {evidently.__version__}')"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            print(f"Перевірка evidently: {check_result.stdout}")
            if check_result.returncode != 0:
                print(f"Помилка перевірки evidently: {check_result.stderr}")
        except Exception as e:
            print(f"Помилка при перевірці evidently: {e}")
        
        print(f"Запуск генерації дашборду моніторингу з {python_path}...")
        result = subprocess.run(
            [python_path, monitoring_script], 
            capture_output=True, 
            text=True, 
            cwd=current_dir,
            timeout=60
        )
        
        if result.returncode == 0:
            print("Дашборд успішно згенеровано!")
            print(f"Вивід: {result.stdout}")
            
            time.sleep(2)
            
            dashboard_file = os.path.join(current_dir, "lab5_monitoring_dashboard.html")
            if os.path.exists(dashboard_file):
                return FileResponse(
                    dashboard_file, 
                    media_type='text/html',
                    filename='monitoring_dashboard.html'
                )
            else:
                return HTMLResponse(content=f"""
                <html>
                <head><title>Помилка</title></head>
                <body style="font-family: Arial; padding: 40px; text-align: center;">
                    <h1>Файл дашборду не створено</h1>
                    <p>Скрипт виконався успішно, але файл lab5_monitoring_dashboard.html не знайдено</p>
                    <h3>Вивід скрипту:</h3>
                    <pre style="background: white; padding: 15px; border-radius: 5px; overflow-x: auto;">
{result.stdout}
                    </pre>
                    <a href="/">← Повернутись до API</a>
                </body>
                </html>
                """)
        else:
            error_html = f"""
            <html>
            <head><title>Помилка генерації дашборду</title></head>
            <body style="font-family: Arial; padding: 40px;">
                <h1>Помилка генерації дашборду</h1>
                <p><strong>Використовувався Python:</strong> {python_path}</p>
                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0;">
                    <h3>Вивід скрипту:</h3>
                    <pre style="background: white; padding: 15px; border-radius: 5px; overflow-x: auto;">
{result.stdout}
                    </pre>
                    <h3>Помилки:</h3>
                    <pre style="background: #ffe6e6; padding: 15px; border-radius: 5px; overflow-x: auto;">
{result.stderr}
                    </pre>
                </div>
                <a href="/">← Повернутись до API</a>
            </body>
            </html>
            """
            return HTMLResponse(content=error_html)
            
    except subprocess.TimeoutExpired:
        error_html = f"""
        <html>
        <head><title>Таймаут</title></head>
        <body style="font-family: Arial; padding: 40px; text-align: center;">
            <h1>Таймаут генерації дашборду</h1>
            <p>Скрипт моніторингу виконувався надто довго</p>
            <a href="/">← Повернутись до API</a>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html)
    except Exception as e:
        error_html = f"""
        <html>
        <head><title>Помилка</title></head>
        <body style="font-family: Arial; padding: 40px; text-align: center;">
            <h1>Помилка запуску моніторингу</h1>
            <p>{str(e)}</p>
            <a href="/">← Повернутись до API</a>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html)
    
@app.get("/monitor/status")
async def monitor_status():
    """
    Endpoint для перевірки статусу моніторингу
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_file = os.path.join(current_dir, "lab5_monitoring_dashboard.html")
    monitoring_script = os.path.join(current_dir, "lab5_monitoring.py")
    
    status = {
        "monitoring_script_exists": os.path.exists(monitoring_script),
        "dashboard_file_exists": os.path.exists(dashboard_file),
        "dashboard_file_size": os.path.getsize(dashboard_file) if os.path.exists(dashboard_file) else 0,
        "dashboard_last_modified": os.path.getmtime(dashboard_file) if os.path.exists(dashboard_file) else None,
        "current_directory": current_dir
    }
    
    return status
@app.get("/check-evidently-env")
async def check_evidently_env():
    import subprocess
    
    evidently_env_python = r"F:\jupyter\gulp\lab4\evidently_env\Scripts\python.exe"
    
    checks = []
    
    if os.path.exists(evidently_env_python):
        checks.append(f"Python знайдено: {evidently_env_python}")
        
        try:
            version_result = subprocess.run(
                [evidently_env_python, "--version"], 
                capture_output=True, 
                text=True
            )
            checks.append(f"Версія Python: {version_result.stdout.strip()}")
        except Exception as e:
            checks.append(f"Помилка перевірки версії: {e}")
        
        try:
            evidently_result = subprocess.run(
                [evidently_env_python, "-c", "import evidently; print(f'Evidently: {evidently.__version__}')"], 
                capture_output=True, 
                text=True
            )
            if evidently_result.returncode == 0:
                checks.append(f"Evidently: {evidently_result.stdout.strip()}")
            else:
                checks.append(f"Evidently не встановлено: {evidently_result.stderr}")
        except Exception as e:
            checks.append(f"Помилка перевірки evidently: {e}")
    else:
        checks.append(f"Python не знайдено за шляхом: {evidently_env_python}")
    
    html_content = f"""
    <html>
    <head><title>Перевірка evidently_env</title></head>
    <body style="font-family: Arial; padding: 40px;">
        <h1>Перевірка віртуального середовища evidently_env</h1>
        <div style="background: #f8f9fa; padding: 20px; border-radius: 8px;">
            <h3>Результати перевірки:</h3>
            <pre style="background: white; padding: 15px; border-radius: 5px;">
{chr(10).join(checks)}
            </pre>
        </div>
        <a href="/monitor">Спробувати моніторинг</a> | 
        <a href="/">← Повернутись до API</a>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)