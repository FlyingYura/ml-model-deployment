'use client';

import { useState } from 'react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Home() {
  const [trainingStatus, setTrainingStatus] = useState<string>('');
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [formData, setFormData] = useState({
    patientid: '',
    age: '',
    gender: '',
    bmi: '',
    smoking: '',
    physicalactivity: '',
    dietquality: '',
    sleepquality: '',
    pollutionexposure: '',
    pollenexposure: '',
    dustexposure: '',
    wheezing: '',
    shortnessofbreath: '',
    chesttightness: '',
    coughing: '',
    nasalcongestion: '',
    gastroesophagealreflux: '',
    lungfunctionfev1: '',
    lungfunctionfvc: '',
    allergichistory: '',
    familyhistory: '',
    medicationuse: '',
    exerciseinduced: '',
    ethnicity_1: '',
    ethnicity_2: '',
    ethnicity_3: '',
    educationlevel_1: '',
    educationlevel_2: '',
    educationlevel_3: '',
  });

  const handleTrainModel = async () => {
    setLoading(true);
    setTrainingStatus('Тренування моделі...');
    try {
      const response = await fetch(`${API_URL}/train-model`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      const data = await response.json();
      if (response.ok) {
        setTrainingStatus(`Модель успішно натренована! Метрики: ${JSON.stringify(data.metrics, null, 2)}`);
      } else {
        setTrainingStatus(`Помилка: ${data.detail || 'Невідома помилка'}`);
      }
    } catch (error: any) {
      setTrainingStatus(`Помилка: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setPredictionResult(null);
    try {
      // Конвертуємо всі значення в числа
      const data: any = {};
      Object.keys(formData).forEach((key) => {
        const value = formData[key as keyof typeof formData];
        if (value !== '') {
          if (key === 'patientid') {
            data[key] = parseInt(value);
          } else {
            data[key] = parseFloat(value);
          }
        }
      });

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });
      const result = await response.json();
      if (response.ok) {
        setPredictionResult(result);
      } else {
        setPredictionResult({ error: result.detail || 'Невідома помилка' });
      }
    } catch (error: any) {
      setPredictionResult({ error: error.message });
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  return (
    <div className="min-h-screen bg-gray-100 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold text-center mb-8">Asthma Prediction API</h1>
        
        {/* Training Section */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-2xl font-semibold mb-4">Тренування моделі</h2>
          <button
            onClick={handleTrainModel}
            disabled={loading}
            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded disabled:bg-gray-400"
          >
            {loading ? 'Тренування...' : 'Навчити модель'}
          </button>
          {trainingStatus && (
            <div className="mt-4 p-4 bg-gray-100 rounded">
              <pre className="whitespace-pre-wrap text-sm">{trainingStatus}</pre>
            </div>
          )}
        </div>

        {/* Prediction Section */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-2xl font-semibold mb-4">Передбачення</h2>
          <form onSubmit={handlePredict} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">Patient ID (опціонально)</label>
                <input
                  type="number"
                  name="patientid"
                  value={formData.patientid}
                  onChange={handleInputChange}
                  className="w-full border border-gray-300 rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Age *</label>
                <input
                  type="number"
                  step="any"
                  name="age"
                  value={formData.age}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Gender *</label>
                <input
                  type="number"
                  step="any"
                  name="gender"
                  value={formData.gender}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">BMI *</label>
                <input
                  type="number"
                  step="any"
                  name="bmi"
                  value={formData.bmi}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Smoking *</label>
                <input
                  type="number"
                  step="any"
                  name="smoking"
                  value={formData.smoking}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Physical Activity *</label>
                <input
                  type="number"
                  step="any"
                  name="physicalactivity"
                  value={formData.physicalactivity}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Diet Quality *</label>
                <input
                  type="number"
                  step="any"
                  name="dietquality"
                  value={formData.dietquality}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Sleep Quality *</label>
                <input
                  type="number"
                  step="any"
                  name="sleepquality"
                  value={formData.sleepquality}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Pollution Exposure *</label>
                <input
                  type="number"
                  step="any"
                  name="pollutionexposure"
                  value={formData.pollutionexposure}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Pollen Exposure *</label>
                <input
                  type="number"
                  step="any"
                  name="pollenexposure"
                  value={formData.pollenexposure}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Dust Exposure *</label>
                <input
                  type="number"
                  step="any"
                  name="dustexposure"
                  value={formData.dustexposure}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Wheezing *</label>
                <input
                  type="number"
                  step="any"
                  name="wheezing"
                  value={formData.wheezing}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Shortness of Breath *</label>
                <input
                  type="number"
                  step="any"
                  name="shortnessofbreath"
                  value={formData.shortnessofbreath}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Chest Tightness *</label>
                <input
                  type="number"
                  step="any"
                  name="chesttightness"
                  value={formData.chesttightness}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Coughing *</label>
                <input
                  type="number"
                  step="any"
                  name="coughing"
                  value={formData.coughing}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Nasal Congestion *</label>
                <input
                  type="number"
                  step="any"
                  name="nasalcongestion"
                  value={formData.nasalcongestion}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Gastroesophageal Reflux *</label>
                <input
                  type="number"
                  step="any"
                  name="gastroesophagealreflux"
                  value={formData.gastroesophagealreflux}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Lung Function FEV1 *</label>
                <input
                  type="number"
                  step="any"
                  name="lungfunctionfev1"
                  value={formData.lungfunctionfev1}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Lung Function FVC *</label>
                <input
                  type="number"
                  step="any"
                  name="lungfunctionfvc"
                  value={formData.lungfunctionfvc}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Allergic History *</label>
                <input
                  type="number"
                  step="any"
                  name="allergichistory"
                  value={formData.allergichistory}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Family History *</label>
                <input
                  type="number"
                  step="any"
                  name="familyhistory"
                  value={formData.familyhistory}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Medication Use *</label>
                <input
                  type="number"
                  step="any"
                  name="medicationuse"
                  value={formData.medicationuse}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Exercise Induced *</label>
                <input
                  type="number"
                  step="any"
                  name="exerciseinduced"
                  value={formData.exerciseinduced}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Ethnicity 1 *</label>
                <input
                  type="number"
                  step="any"
                  name="ethnicity_1"
                  value={formData.ethnicity_1}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Ethnicity 2 *</label>
                <input
                  type="number"
                  step="any"
                  name="ethnicity_2"
                  value={formData.ethnicity_2}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Ethnicity 3 *</label>
                <input
                  type="number"
                  step="any"
                  name="ethnicity_3"
                  value={formData.ethnicity_3}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Education Level 1 *</label>
                <input
                  type="number"
                  step="any"
                  name="educationlevel_1"
                  value={formData.educationlevel_1}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Education Level 2 *</label>
                <input
                  type="number"
                  step="any"
                  name="educationlevel_2"
                  value={formData.educationlevel_2}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Education Level 3 *</label>
                <input
                  type="number"
                  step="any"
                  name="educationlevel_3"
                  value={formData.educationlevel_3}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 rounded px-3 py-2"
                />
              </div>
            </div>
            <button
              type="submit"
              disabled={loading}
              className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded disabled:bg-gray-400"
            >
              {loading ? 'Передбачення...' : 'Зробити передбачення'}
            </button>
          </form>
          {predictionResult && (
            <div className="mt-4 p-4 bg-gray-100 rounded">
              <h3 className="font-semibold mb-2">Результат:</h3>
              <pre className="whitespace-pre-wrap text-sm">
                {JSON.stringify(predictionResult, null, 2)}
              </pre>
            </div>
          )}
        </div>

        {/* API Documentation Link */}
        <div className="mt-8 text-center">
          <a
            href={`${API_URL}/docs`}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-500 hover:underline"
          >
            Переглянути документацію API (Swagger UI)
          </a>
        </div>
      </div>
    </div>
  );
}
