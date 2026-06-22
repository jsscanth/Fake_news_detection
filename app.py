import streamlit as st
import joblib
from utils import preprocessing  

st.set_page_config(
    page_title="Fake News Detection System",
    layout="centered"
)

@st.cache_resource
def load_pipeline(model_choice):
    if model_choice == "Logistic Regression":
        return joblib.load("models/logistic_regression.joblib")
    elif model_choice == "Random Forest":
        return joblib.load("models/random_forest.joblib")
    elif model_choice == "Support Vector Machine (SVM)":
        return joblib.load("models/svm.joblib")
    else:
        raise ValueError("Unknown model selected.")

st.sidebar.header("App Configuration")
st.sidebar.write("Choose which machine learning model you want to use for the classification task.")

selected_model = st.sidebar.selectbox(
    "Active Model",
    ["Logistic Regression", "Random Forest", "Support Vector Machine (SVM)"]
)

st.sidebar.success(f"Loaded: {selected_model}")

st.title("NLP Fake News Detection Dashboard")
st.markdown("""
This application utilizes advanced NLP pipelines to analyze incoming news text 
and determine its statistical authenticity. 
""")
st.write("---")

user_input = st.text_area(
    "Paste News Article Content Below:", 
    height=250, 
    placeholder="Type or paste the full text of the article here..."
)

if st.button("Run Text Analysis", type="primary"):
    word_count = len(user_input.split())
    
    if not user_input.strip():
        st.warning("Action required: Please input article text before analyzing.")
    elif word_count < 30:
        st.warning("Action required: Please input a longer article text before analyzing.")
    else:
        with st.spinner("Processing text and running prediction algorithms..."):
            
            current_pipeline = load_pipeline(selected_model)
            cleaned_text = preprocessing(user_input)
            prediction = current_pipeline.predict([cleaned_text])[0]
            
            confidence_text = ""
            try:
                probabilities = current_pipeline.predict_proba([cleaned_text])[0]
                confidence_score = max(probabilities) * 100
                confidence_text = f" (Confidence: {confidence_score:.2f}%)"
            except AttributeError:
                pass
        
        st.subheader("Analysis Results:")
         
        if prediction == 0:
            st.error(f"**Potential Misinformation / Fake News Detected**{confidence_text}")
            st.markdown("""
            **Recommendation:** This text exhibits linguistic patterns frequently associated 
            with unreliable news reporting. Cross-verify these claims with trusted sources.
            """)
        else:
            st.success(f"**Likely Authentic / Credible News**{confidence_text}")
            st.markdown("""
            **Recommendation:** The structural features of this article match patterns 
            typically found in factual, verified news publications.
            """)