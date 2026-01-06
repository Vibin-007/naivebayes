# 📧 Email Spam Detection (Naive Bayes)

This project implements a **Naive Bayes Classifier** to detect spam emails/SMS using the SMS Spam Collection dataset.

## 🚀 Features

- **Spam Classification**: Accurately classifies messages as 'Spam' or 'Ham'.
- **Interactive UI**: Type in any message to instantly check if it's spam.
- **Visualizations**: Message length analysis and class distribution.

## 🛠️ Usage

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App**:
   ```bash
   python -m streamlit run app.py
   ```

## 📁 Project Structure

- `app.py`: Streamlit application file.
- `naive_bayes_analysis.ipynb`: Jupyter notebook for text classification analysis.
- `spam_nb.csv`: Dataset containing labeled messages.
- `requirements.txt`: Python package dependencies.

## 📂 Dataset

The project uses `spam_nb.csv`. It contains ~5,572 messages labeled as Ham or Spam.

## 📦 Requirements

- streamlit
- pandas
- scikit-learn
- matplotlib
- seaborn
