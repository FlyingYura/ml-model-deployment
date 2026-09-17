# ML Model Deployment — Asthma Prediction API

Deploy the asthma diagnosis model as a service: **FastAPI** backend + **Next.js** UI for training and inference.

## What this project covers

- REST API: train model, single/batch predict, health checks
- Model persistence (`lab4/models/model_latest.pkl`)
- PostgreSQL for patients / predictions
- Simple web UI to trigger training and send patient features
- CORS-enabled local full-stack setup

## Stack

| Layer | Tech |
|--------|------|
| API | FastAPI, Uvicorn, Pydantic |
| ML | scikit-learn, joblib |
| DB | PostgreSQL, SQLAlchemy |
| UI | Next.js 16, React 19, Tailwind CSS |
