# ProphetBnB: Airbnb Price Prediction & Host Segmentation

<img width="1785" height="934" alt="image" src="https://github.com/user-attachments/assets/41063a43-fc62-4194-8f21-bd211a19830a" />




Welcome to ProphetBnB, a comprehensive Python project for predicting Airbnb listing prices and segmenting hosts for actionable insights. ProphetBnB combines robust machine learning, interactive visualizations, and a Streamlit web app for seamless exploration of Airbnb data.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

---

## Features

### 🏷️ Price Prediction
- Predicts Airbnb listing prices using regression models
- Flags underpriced and overpriced listings

### 👥 Host Clustering
- Segments hosts using clustering on price, reviews, and availability
- Provides insights for targeted marketing and strategy

### 📊 Interactive Visualizations
- Folium maps for geographic analysis
- Plotly charts for cluster and feature exploration

### 🚀 Streamlit App
- Intuitive interface for data exploration and prediction
- Accessible locally or via web deployment

---

<img width="1316" height="897" alt="image" src="https://github.com/user-attachments/assets/ce05aa77-5afd-46f5-a7f6-e42256b4e26c" />


## Project Structure

```
AirBnB-PriceSense/
│
├── data/
│   ├── raw/               # Original datasets (CSV/JSON)
│   └── processed/         # Cleaned datasets for modeling
│
├── notebooks/
│   ├── 01_EDA.ipynb       # Exploratory Data Analysis
│   └── 02_Modeling.ipynb  # Regression & Clustering Models
│
├── src/
│   ├── data_preprocessing.py   # Data cleaning & feature engineering
│   ├── model_training.py       # ML model training & evaluation
│   └── visualizations.py       # Plotting & mapping functions
│
├── app/
│   └── prophetbnb_app.py       # Streamlit web app
│
├── assets/
│   └── logo.png                # Logo/images
│
├── environment.yml             # Conda environment
├── requirements.txt            # Pip dependencies
└── README.md                   # Project overview
```

---

## Installation

**1. Using Conda (Recommended)**
```bash
# Navigate to the project folder
cd path/to/AirBnB-PriceSense

# Create environment
conda env create -f environment.yml

# Activate environment
conda activate ProphetBnB
```

**2. Using Pip / Virtualenv**
```bash
# Create virtual environment
python -m venv ProphetBnB_env

# Activate environment (Windows)
ProphetBnB_env\Scripts\activate

# Activate environment (macOS/Linux)
source ProphetBnB_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

**1. Run Jupyter Notebooks**
```bash
jupyter notebook
```
- `notebooks/01_EDA.ipynb`: Explore dataset, visualize distributions
- `notebooks/02_Modeling.ipynb`: Train regression & clustering models

**2. Run Streamlit App**
```bash
streamlit run app/prophetbnb_app.py
```
- Explore predictions and visualizations interactively
- Supports local and web deployment

---

## Dataset

- Place your Airbnb dataset in `data/raw/listings.csv`
- The script `src/data_preprocessing.py` creates a cleaned dataset in `data/processed/listings_clean.csv`

**Recommended Dataset Sources:**
- [Inside Airbnb](http://insideairbnb.com/get-the-data.html)
- Any Airbnb dataset you have access to

---

## Dependencies

- Python 3.10+
- pandas, numpy, scikit-learn
- matplotlib, seaborn, plotly
- geopandas, folium, streamlit, streamlit-folium
- joblib, pyyaml
- Jupyter Notebook

See `environment.yml` & `requirements.txt` for full details.

---

## Contributing

1. Fork the repository & create a new branch
2. Make your changes & test thoroughly
3. Submit a pull request with a clear description

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Example: Streamlit App

<img width="1348" height="756" alt="image" src="https://github.com/user-attachments/assets/a53ab3d8-7c9c-4b7e-85e4-d7579f0dbd9a" />

<img width="1267" height="905" alt="image" src="https://github.com/user-attachments/assets/ce83ee2c-e9cc-4c2c-a4ca-902ce8afe0f0" />


> **How to Begin:**  
> 1. Choose your data source from Airbnb, CSV, or web  
> 2. Set your filters for price, guests, reviews, etc.  
> 3. Click *Analyze Listings* for insights

**Supported Data:**  
- InsideAirbnb, CSV (with at least: id, name, price)
- For best experience: use clean, recent datasets

---

*Empower your Airbnb decisions with smart predictions and interactive insights!*
