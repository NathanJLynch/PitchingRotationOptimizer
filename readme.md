# Rotation Optimizer

A machine learning and optimization platform that generates optimal starting pitcher rotations for all 30 MLB teams.

The project combines MLB statistical data, machine learning models, and dynamic programming techniques to determine which pitchers should start against specific opponents throughout a season. Rather than using a static rotation, the optimizer evaluates matchup quality, divisional importance, projected pitcher effectiveness, and scheduling constraints to maximize overall rotation value.

---

## Project Goal

> Better pitchers should face stronger opponents and division rivals.

Not every game carries equal value. Games against divisional rivals, playoff contenders, and elite offenses can have a larger impact on a team's success than lower-stakes matchups.

Rotation Optimizer uses performance projections and optimization algorithms to determine how a team's starting pitchers should be allocated across future games. Rivalry weighting can also be adjusted based on divisional standings, increasing the importance of critical late-season matchups.

---

## Key Features

- Generates optimized pitching rotations for all 30 MLB teams
- Collects and processes MLB data from multiple sources
- Uses machine learning models to predict pitcher effectiveness
- Implements dynamic programming for rotation optimization
- Supports asynchronous data collection for improved performance
- Stores and manages data using PostgreSQL
- Exposes functionality through a FastAPI backend
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

### 1. Data Collection

Historical pitcher, schedule, and team performance data are collected from MLB APIs and PyBaseball.

### 2. Asynchronous Processing

Data retrieval is performed asynchronously using HTTPX.

**Result:** Data ingestion times were reduced by approximately **20x** compared to a sequential implementation.

### 3. Feature Engineering

Raw statistics are transformed into model-ready features using:

- Pandas
- NumPy
- Custom feature generation logic

Examples include:

- Rolling performance metrics
- Opponent strength indicators
- Team-level performance statistics
- Schedule-based features

### 4. Normalization

Features often exist on different statistical scales and require normalization before modeling.

The project uses sigmoid-based scaling techniques to improve consistency across statistical categories and reduce sensitivity to extreme values.

### 5. Machine Learning Prediction

The processed features are used to train machine learning models that estimate future pitcher effectiveness.

Models evaluated include:

- Scikit-learn models
- XGBoost
- LightGBM

### 6. Rotation Optimization

Model predictions are passed into a dynamic programming optimizer that determines the most effective allocation of starting pitchers across future games.

---

## Machine Learning Approach

The machine learning component focuses on estimating future pitcher performance using historical MLB data.

The prediction system evaluates relationships between:

- Historical pitcher performance
- Opponent offensive strength
- Recent trends
- Team context
- Schedule information

Several models were tested during development, including XGBoost and LightGBM, to compare predictive performance and improve forecast quality.

The machine learning system and optimization system remain intentionally separated:

- Machine Learning predicts performance
- Dynamic Programming determines deployment strategy

This allows both systems to be improved independently.

---

## Optimization Strategy

After generating pitcher performance projections, the optimization engine determines how pitchers should be assigned to future games.

The optimizer considers:

- Projected pitcher quality
- Opponent strength
- Division rival status
- Team standings
- Workload management
- Rotation spacing constraints

Rather than optimizing individual games, the algorithm seeks to maximize projected value across an entire schedule window.

This prevents the common issue of treating every game as equally important.

---

## Example Workflow

```text
1. Collect MLB Data
        ↓
2. Clean and Normalize Statistics
        ↓
3. Generate Features
        ↓
4. Train/Run Machine Learning Models
        ↓
5. Generate Pitcher Projections
        ↓
6. Run Dynamic Programming Optimizer
        ↓
7. Produce Optimal Rotation
```

---

## Performance Improvements

### Asynchronous Data Collection

The initial version of the application relied on sequential API requests.

After transitioning to asynchronous requests using HTTPX:

- Data collection became approximately 20x faster
- League-wide updates completed significantly more quickly
- Scalability improved for all 30 MLB teams

---

## Future Improvements

- Injury-aware rotation generation
- Bullpen integration and reliever optimization
- Automated model retraining
- Explainable AI metrics for decision transparency
- Enhanced feature engineering
- Interactive dashboard and visualizations
- Cloud deployment
- CI/CD integration with GitHub Actions
- Expanded automated test coverage

---

## Known Limitations

### Shohei Ohtani

Players with limited pitching appearances may not satisfy minimum data thresholds required by the prediction models.

Future versions may introduce alternative handling strategies for elite players with smaller pitching samples.

### Incomplete Rotations

Some MLB teams temporarily lack enough healthy qualified starting pitchers to construct a complete projected rotation.

Results generally improve as additional season data becomes available.


---


## Screenshots

### Rotation Output

<img width="1102" height="723" alt="Screenshot 2026-08-03 173206" src="https://github.com/user-attachments/assets/1aa36dc3-e7c9-4e4f-b3d8-335f62c698a4" />


## Author

**Nathan Lynch**

Computer Science Student  
University of Wisconsin-Milwaukee

- LinkedIn: https://linkedin.com/in/nathan-j-lynch27
- GitHub: https://github.com/NathanJLynch
