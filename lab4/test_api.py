import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print("Health check:", response.json())
    return response.status_code == 200


def test_train_model():
    print("\nТестування тренування моделі")
    response = requests.post(f"{BASE_URL}/train-model")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Модель навчена: {result['model_file']}")
        print(f"Метрики: {json.dumps(result['metrics'], indent=2)}")
        return True
    else:
        print(f"Помилка: {response.text}")
        return False


def test_predict():
    print("\nТестування передбачення")
    
    test_data = {
        "patientid": 9999,
        "age": 0.5,
        "gender": -0.98,
        "bmi": -1.58,
        "smoking": -0.41,
        "physicalactivity": -1.43,
        "dietquality": 0.16,
        "sleepquality": 0.97,
        "pollutionexposure": 0.81,
        "pollenexposure": -0.78,
        "dustexposure": 0.5,
        "wheezing": 0.3,
        "shortnessofbreath": 0.2,
        "chesttightness": 0.1,
        "coughing": 0.4,
        "nasalcongestion": 0.2,
        "gastroesophagealreflux": 0.1,
        "lungfunctionfev1": 0.8,
        "lungfunctionfvc": 0.9,
        "allergichistory": 0.3,
        "familyhistory": 0.2,
        "medicationuse": 0.1,
        "exerciseinduced": 0.8,
        "ethnicity_1": 2.01,
        "ethnicity_2": -0.33,
        "ethnicity_3": -0.32,
        "educationlevel_1": -0.80,
        "educationlevel_2": -0.68,
        "educationlevel_3": -0.33
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Передбачення: {result['prediction']}")
        print(f"Ймовірність: {result['probability']:.4f}")
        return True
    else:
        print(f"Помилка: {response.text}")
        return False


if __name__ == "__main__":
    print("Тестування")
    
    if not test_health():
        print("API лежить і не встає (і я теж так хочу)")
        exit(1)

    
    print("\nТестування завершено")

