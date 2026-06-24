# Deep-Learning-Project

# Agricultural Commodity Price Prediction using Deep Learning

## Overview

Agricultural commodity prices are highly volatile and influenced by seasonal production, climatic conditions, market demand, transportation costs, inflation, and various economic factors. These fluctuations significantly impact farmers, traders, consumers, and supply chain stakeholders.

This project presents a deep learning-based forecasting system that predicts short-term and mid-term agricultural commodity prices using historical market data. Multiple sequence learning models including LSTM, GRU, and Transformer architectures are implemented and compared to identify the most effective approach for commodity price forecasting.


## Problem Statement

Agricultural commodity prices fluctuate frequently due to multiple interdependent factors such as seasonal supply variations, market demand, climate conditions, logistics disruptions, government policies, and global economic trends.

The objective of this project is to design an intelligent forecasting system capable of accurately predicting agricultural commodity prices using historical market data.

### Objectives

- Analyze historical agricultural commodity price data.
- Perform extensive exploratory data analysis (EDA).
- Engineer relevant features for forecasting.
- Learn long-term temporal dependencies in market behavior.
- Generate reliable short-term and medium-term price forecasts.
- Compare multiple deep learning architectures for prediction accuracy.
- Evaluate model performance using industry-standard metrics.


## Dataset

### Source

Kaggle Dataset:
https://www.kaggle.com/datasets/anshtanwar/current-daily-price-of-various-commodities-india

### Dataset Statistics

| Attribute | Value |
|------------|---------|
| Records | 23,093 |
| Features | 10 |
| Format | CSV |
| Frequency | Weekly |
| Domain | Agricultural Commodity Prices |
| Target Variable | Modal Price |

### Dataset Features

| Feature | Description |
|----------|------------|
| State | State where commodity was sold |
| District | District of market |
| Market | Market name |
| Commodity | Commodity type |
| Variety | Commodity variety |
| Grade | Commodity grade |
| Arrival_Date | Date of market arrival |
| Min Price | Minimum market price |
| Max Price | Maximum market price |
| Modal Price | Most frequent market price (Target Variable) |


## Project Workflow

```text
Dataset Collection
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Scaling
        │
        ▼
Sequence Generation
(Sliding Window)
        │
        ▼
Train-Test Split
        │
        ▼
Model Training
(LSTM / GRU / Transformer)
        │
        ▼
Model Evaluation
(RMSE, MAE, R²)
        │
        ▼
Prediction Visualization
```


## Exploratory Data Analysis (EDA)

EDA was performed to understand commodity price behavior and market trends.

### Analysis Performed

- Commodity-wise filtering
- Price trend visualization
- Distribution analysis
- Statistical summaries
- Time-series trend analysis
- Outlier identification

### Visualizations

- Commodity Price Trend over Time
- Price Distribution Boxplots
- Actual vs Predicted Price Comparison


## Feature Engineering

Several derived features were created to improve forecasting performance.

### Existing Features

| Feature | Purpose |
|-----------|----------|
| Price Spread | Market volatility indicator |
| MA_4 | 4-week Moving Average |
| MA_8 | 8-week Moving Average |
| EMA_4 | Exponential Moving Average |
| Momentum | Trend strength indicator |

### Lag Features

Historical context was incorporated through:

- lag_1
- lag_2
- lag_3

These features allow the model to directly access recent price history.

### Rolling Statistics

| Feature | Purpose |
|-----------|----------|
| rolling_mean_7 | Long-term trend estimation |
| rolling_std_7 | Market volatility estimation |

### Target Transformation

The target variable was smoothed using:

- 4-week rolling average
- Log transformation

Benefits:

- Reduces noise
- Stabilizes variance
- Improves model learning


## Data Preprocessing

The following preprocessing steps were performed:

1. Date conversion
2. Commodity filtering
3. Missing value handling
4. Feature engineering
5. Data normalization
6. Sequence generation
7. Time-series train-test splitting

### Normalization

Min-Max Scaling was applied to scale all features into the range:

```python
[0,1]
```

This improves neural network convergence and training stability.


## Time-Series Transformation

Deep learning models require sequential data as input.

### Sliding Window Technique

```python
WINDOW = 24
HORIZON = 3
```

### Explanation

- Window Size = 24 weeks (~6 months)
- Forecast Horizon = 3 weeks

The model learns from the previous 24 weeks to predict the next 3-week average trend.


## Deep Learning Models

### 1. Bidirectional LSTM

#### Architecture

```text
Bidirectional LSTM (128)
        ↓
Dropout (0.2)
        ↓
LSTM (64)
        ↓
Dense (1)
```

#### Advantages

- Excellent at learning long-term dependencies
- Captures temporal patterns effectively
- Highest forecasting accuracy

#### Disadvantages

- Slower training
- Higher computational cost


### 2. Bidirectional GRU

#### Architecture

```text
Bidirectional GRU (128)
        ↓
Dropout (0.3)
        ↓
GRU (64)
        ↓
Dense (1)
```

#### Advantages

- Faster training than LSTM
- Fewer parameters
- Lower computational requirements

#### Disadvantages

- Slightly lower forecasting accuracy
- Less effective for very long-term dependencies


### 3. Hybrid Transformer

#### Architecture

```text
Input Layer
      ↓
Multi-Head Attention
      ↓
Layer Normalization
      ↓
LSTM Layer
      ↓
Dense Layer
      ↓
Output Layer
```

#### Advantages

- Captures long-range dependencies
- Attention mechanism improves context learning
- Strong forecasting capability

#### Disadvantages

- Computationally expensive
- Requires careful hyperparameter tuning
- Generally requires larger datasets


## Training Configuration

| Parameter | Value |
|------------|---------|
| Optimizer | Adam |
| Loss Function | Mean Squared Error (MSE) |
| Epochs | 50–60 |
| Batch Size | 16–32 |
| Validation Split | 10–15% |
| Early Stopping | Enabled |


## Evaluation Metrics

The models were evaluated using the following metrics:

### Root Mean Square Error (RMSE)

Measures prediction error magnitude.

### Mean Absolute Error (MAE)

Measures average absolute deviation between actual and predicted values.

### R² Score

Measures how well the model explains variance in the data.

Higher R² values indicate better predictive performance, while lower RMSE and MAE values indicate lower prediction error.


## Results and Comparison

| Model | Performance |
|---------|---------|
| LSTM | Best Overall |
| GRU | Moderate |
| Transformer | Competitive |

### Observations

#### LSTM

- Lowest RMSE
- Lowest MAE
- Highest R² Score
- Best overall forecasting performance

#### GRU

- Faster training
- Lower computational cost
- Good trade-off between speed and accuracy

#### Transformer

- Competitive accuracy
- Strong long-range dependency learning
- Higher computational requirements

Among all evaluated models, LSTM demonstrated the highest prediction accuracy and stability for agricultural commodity price forecasting.


## Visualization

The project generates:

### Actual vs Predicted Price Trends

Comparison between:

- Actual Market Prices
- LSTM Predictions
- GRU Predictions
- Transformer Predictions

These visualizations help evaluate trend-following capability and forecasting accuracy.


## Technologies Used

### Programming Language

- Python

### Development Environment

- Google Colab

### Libraries

```python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-Learn
TensorFlow
Keras
```


## Project Structure

```text
Agricultural-Commodity-Price-Prediction/
│
├── data/
│   └── Price_Agriculture_commodities_Week.csv
│
├── notebooks/
│   └── DL_Project_Code.ipynb
│
├── src/
│   └── model_training.py
│
├── images/
│   ├── price_trend.png
│   ├── distribution.png
│   └── prediction_comparison.png
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```


## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Agricultural-Commodity-Price-Prediction.git
cd Agricultural-Commodity-Price-Prediction
```

Install required dependencies:

```bash
pip install -r requirements.txt
```


## Running the Project

Run the notebook:

```bash
jupyter notebook DL_Project_Code.ipynb
```

Or execute the Python script:

```bash
python model_training.py
```


## Future Scope

Potential improvements include:

- Integrating weather and climate data
- Incorporating inflation and economic indicators
- Including demand and supply metrics
- Supporting multiple commodity forecasting
- Real-time prediction pipelines
- Web and mobile deployment for farmers
- Advanced Transformer architectures such as Informer and Temporal Fusion Transformer (TFT)


## Conclusion

This project demonstrates the effectiveness of deep learning techniques in agricultural commodity price forecasting. Through extensive experimentation with LSTM, GRU, and Transformer models, the study shows that deep learning can successfully capture temporal patterns and market dynamics.

Among all evaluated models, Bidirectional LSTM achieved the highest prediction accuracy and stability, making it the most reliable approach for short-term and medium-term agricultural price prediction.


## Authors

- Soumyajit Kuila

Centre for Artificial Intelligence & Machine Learning  
Department of Computer Science & Engineering  
Siksha 'O' Anusandhan University, Bhubaneswar, Odisha

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
