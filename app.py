import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Set page configuration
st.set_page_config(page_title="Email Spam Detection", layout="wide")

st.title("📧 Email Spam Detection (Naive Bayes)")
st.markdown("""
This application classifies emails as **Spam** or **Ham** (Not Spam) using the **Naive Bayes** algorithm.
""")

# Load dataset
@st.cache_data
def load_data():
    try:
        # Try latin-1 encoding which is common for this dataset
        df = pd.read_csv('spam.csv', encoding='latin-1')
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

df = load_data()

if df is not None:
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", ["Data Overview", "Spam Check (Classification)"])

    def preprocess_data(df):
        df_p = df.copy()
        
        # Cleanup: Drop unnecessary columns and rename
        # Dataset usually has v1 (label), v2 (text), and Unnamed cols
        drop_cols = [col for col in df_p.columns if 'Unnamed' in col]
        df_p.drop(columns=drop_cols, inplace=True)
        
        if 'v1' in df_p.columns and 'v2' in df_p.columns:
            df_p.rename(columns={'v1': 'Label', 'v2': 'Message'}, inplace=True)
        
        # Binary encoding for Label: spam=1, ham=0
        df_p['Target'] = df_p['Label'].map({'spam': 1, 'ham': 0})
        
        return df_p

    df_clean = preprocess_data(df)

    if app_mode == "Data Overview":
        st.header("📊 Dataset Overview")
        st.write("First 10 rows:")
        st.dataframe(df_clean.head(10))
        
        st.write("Statistics:")
        st.write(df_clean.describe())
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Class Distribution")
            st.bar_chart(df_clean['Label'].value_counts())
            
        with col2:
            st.subheader("Message Length Analysis")
            df_clean['Length'] = df_clean['Message'].apply(len)
            fig, ax = plt.subplots()
            sns.histplot(data=df_clean, x='Length', hue='Label', kde=True, ax=ax)
            ax.set_title("Message Length Distribution by Class")
            st.pyplot(fig)

    elif app_mode == "Spam Check (Classification)":
        st.header("🕵️ Spam Detector")
        
        # Text Vectorization
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df_clean['Message'])
        y = df_clean['Target']
        
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Model Training
        model = MultinomialNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Metrics Display
        col_metrics1, col_metrics2 = st.columns(2)
        with col_metrics1:
            st.metric("Model Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        
        st.subheader("Test Your Own Message")
        user_input = st.text_area("Enter an email or SMS message to check:")
        
        if st.button("Check Message"):
            if user_input:
                user_vec = vectorizer.transform([user_input])
                prediction = model.predict(user_vec)[0]
                result = "SPAM 🚨" if prediction == 1 else "HAM (Safe) ✅"
                
                if prediction == 1:
                    st.error(f"Prediction: {result}")
                else:
                    st.success(f"Prediction: {result}")
            else:
                st.warning("Please enter some text.")
                
        # Advanced Metrics Expander
        with st.expander("View Detailed Model Performance"):
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                        xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
            ax_cm.set_ylabel('Actual')
            ax_cm.set_xlabel('Predicted')
            st.pyplot(fig_cm)
