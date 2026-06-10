import streamlit as st
import joblib
from utils import preprocessing  # Make sure your text cleaning function is in utils.py

# Page Configuration & UI Theme

st.set_page_config(
    page_title="Fake News Detection System",
    layout="centered"
)


# Optimized Model Loading (Cached)

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

#Sidebar UI - Select model and show which one is active

st.sidebar.header("App Configuration")
st.sidebar.write("Choose which machine learning model you want to use for the classification task.")

selected_model = st.sidebar.selectbox(
    "Active Model",
    ["Logistic Regression", "Random Forest", "Support Vector Machine (SVM)"]
)

# Visual indicator in the sidebar showing which model is loaded
st.sidebar.success(f"Loaded: {selected_model}")

# Main Application UI Layout

st.title("NLP Fake News Detection Dashboard")
st.markdown("""
This application utilizes advanced NLP pipelines to analyze incoming news text 
and determine its statistical authenticity. 
""")
st.write("---")

# Text input field for the news article
user_input = st.text_area(
    "Paste News Article Content Below:", 
    height=250, 
    placeholder="Type or paste the full text of the article here..."
)

# Inference Logic Execution

if st.button("Run Text Analysis", type="primary"):
    word_count = len(user_input.split())
    # check if the input is empty or too short to analyze effectively
    if not user_input.strip():
        st.warning("Action required: Please input article text before analyzing.")
    elif word_count < 30:
        st.warning("Action required: Please input a longer article text before analyzing.")
    else:
        with st.spinner("Processing text and running prediction algorithms..."):
            
            # Step 1: Load the user's chosen pipeline
            current_pipeline = load_pipeline(selected_model)
            
            # Step 2: Preprocess the input via your utils.py function
            cleaned_text = preprocessing(user_input)
            
            # Step 3: Run prediction through the pipeline 
            # (Pipeline automatically vectorizes the clean text and passes it to the model)
            prediction = current_pipeline.predict([cleaned_text])[0]
            
            # Step 4: Extract probability score if available
            confidence_text = ""
            try:
                probabilities = current_pipeline.predict_proba([cleaned_text])[0]
                confidence_score = max(probabilities) * 100
                confidence_text = f" (Confidence: {confidence_score:.2f}%)"
            except AttributeError:
                # If your SVM model wasn't trained with probability=True, it skips this
                pass

        # Display Results Nicely
        
        st.subheader("Analysis Results:")
        
        # NOTE: Adjust mapping depending on your dataset labels. 
        # This code assumes 0 = Fake, 1 = Real. 
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