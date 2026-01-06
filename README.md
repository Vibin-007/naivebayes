# 📧 Email Spam Detector (Naive Bayes)

A Streamlit application that detects whether an email or SMS message is "Spam" or "Ham" (Not Spam) using a Naive Bayes Classifier.

## 📊 Features

- **Spam Classification**: Instantly predicts the category of any text message entered by the user.
- **Probability Analysis**: Uses probabilistic methods to determine the likelihood of spam.
- **Text Visualization**: Word clouds and message length distributions for Spam vs. Ham.
- **Real-time Training**: Trains the model on `spam_nb.csv` instantly.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Vibin-007/naivebayes.git
   cd naivebayes
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

- `app.py`: Main application file containing the Streamlit interface and logic.
- `spam_nb.csv`: Dataset containing labeled messages (Spam/Ham).
- `naive_bayes_analysis.ipynb`: Jupyter notebook for natural language processing and model building.
- `requirements.txt`: List of Python dependencies.

## 📈 Model Information

The model uses **Multinomial Naive Bayes** to classify text based on:
- **Word Frequencies** (Count Vectorization)
- **Message Keywords**
