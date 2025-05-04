# Zoonotic Outbreak Predictor 🦠🔬🤖

This project models and predicts the risk of zoonotic disease outbreaks using machine learning. It simulates the transmission chain from wildlife to livestock to humans through a series of predictive models trained on synthetic but ecologically-informed data. The system leverages ensemble learning to forecast the likelihood of an outbreak and provide insight into one of the most pressing global public health threats.

---

## 🚀 Project Objectives

- Model zoonotic pathogen spillover at three key interfaces:
  1. Wildlife infection risk
  2. Wildlife-to-livestock transmission
  3. Livestock-to-human spillover
- Use real-world ecological, environmental, and behavioural insights to inform model features
- Build an ensemble model to combine the predictions and improve overall accuracy
- Demonstrate machine learning's potential in public health forecasting

---

## 🧠 Technologies Used

- **Python 3**
- `scikit-learn` for model development
- `XGBoost` for gradient boosting models
- `pandas` & `numpy` for data processing
- `matplotlib` & `seaborn` for visualizations

---

## 🧪 Models & Structure

The system is divided into three primary models:

1. **Wildlife Infection Risk Model**
   - Predicts probability of wildlife species being infected based on habitat loss, climate, and human proximity.

2. **Spillover to Livestock Model**
   - Estimates risk of pathogen jumping to livestock given species interactions, biosecurity, and land use.

3. **Livestock-to-Human Model**
   - Predicts final transmission risk based on farming practices, consumption habits, and human susceptibility.

A final **ensemble model** aggregates all three predictions for a comprehensive outbreak risk score.

---

## 📊 Results

- Final model accuracy: **~90.5%**
- Ensemble method significantly outperformed individual models.
- Visualizations and confusion matrices confirm balanced performance.

---

## 🔭 Future Work

- Incorporate real-time data sources (e.g. weather APIs, deforestation trackers)
- Build a web-based interface or dashboard
- Extend to time-series forecasting using LSTM or similar deep learning models

---

## 📚 Acknowledgements

This project was developed as part of a final year Computer Science project at [Your University Name], integrating real-world scientific research with machine learning techniques.


