# Rotation Optimizer

A machine learning and optimization platform that generates optimal starting pitcher rotations for all 30 MLB teams.

The project combines MLB statistical data, machine learning models, and dynamic programming techniques to determine which pitchers should start against specific opponents throughout a season. Rather than using a static rotation, the optimizer evaluates matchup quality, divisional importance, projected pitcher effectiveness, and scheduling constraints to maximize overall rotation value.

---

## Project Goal

> Better pitchers should face stronger opponents and division rivals.

The optimizer is built around the idea that not every game carries equal value. Games against division rivals, playoff contenders, and high-performing offenses have a greater impact on a team's success and should receive stronger pitching assignments whenever possible.

Rivalry weighting can also be adjusted dynamically based on division standings, increasing the importance of critical late-season matchups.

---

## Features

- Generates optimized pitching rotations for all 30 MLB teams
- Collects and processes historical MLB data from multiple sources
- Uses machine learning models to project pitcher performance
- Implements dynamic programming for rotation optimization
- Supports asynchronous data collection for improved performance
- Stores and manages data using PostgreSQL
- Provides REST API endpoints through FastAPI
- Performs automated feature engineering and statistical normalization

---

## Tech Stack

### Languages

- Python

### Backend

- FastAPI
- Pydantic
- SQLAlchemy

### Database

- PostgreSQL

### Data Engineering

- Pandas
- NumPy
- HTTPX
- Requests
- PyBaseball

### Machine Learning

- Scikit-learn
- XGBoost
- LightGBM

### Testing

- PyTest

---

## System Architecture

```text
MLB API / PyBaseball
          │
          ▼
Asynchronous Data Collection
          │
          ▼
Data Cleaning & Normalization
          │
          ▼
Feature Engineering
          │
          ▼
Machine Learning Models
          │
          ▼
Dynamic Programming Optimizer
          │
          ▼
PostgreSQL Database
          │
          ▼
FastAPI Endpoints
```

---

## Data Pipeline

The platform follows a multi-stage pipeline:

### 1. Data Collection

Historical pitching, schedule, and team performance data are retrieved from MLB APIs and PyBaseball.

### 2. Asynchronous Processing

Data retrieval is performed asynchronously using HTTPX.

This reduced collection times by approximately **20x** compared to a sequential implementation.

### 3. Feature Engineering

Statistics are transformed into model-ready features using:

- Pandas
- NumPy
- Custom feature generation logic

#
