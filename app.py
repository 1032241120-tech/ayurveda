# =========================================
# 🔹 AyurGenixAI - Streamlit App
# =========================================
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os

port = int(os.environ.get("PORT", 8501))

# =========================================
# 🔹 Caching Models & Data for Performance
# =========================================

# Cache the Sentence Transformer model to avoid reloading it on every run
@st.cache_resource
def load_bert_model():
    """Loads the SentenceTransformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

# Cache the data and embeddings to avoid re-processing on every interaction
@st.cache_data
def load_data_and_embeddings(_model):
    """
    Loads the dataset, cleans it, and generates symptom embeddings.
    The model is passed as an argument to ensure this function reruns if the model changes.
    """
    try:
        # Load the dataset from the Excel file
        df = pd.read_excel("AyurGenixAI_Dataset.xlsx", sheet_name="in")
    except FileNotFoundError:
        st.error("Error: The dataset file 'AyurGenixAI_Dataset.xlsx' was not found.")
        st.info("Please make sure the dataset file is in the same folder as this app.py file.")
        return None, None

    # Select the columns we need
    df = df[['Disease', 'Symptoms', 'Formulation',
             'Diet and Lifestyle Recommendations',
             'Yoga & Physical Therapy',
             'Medical Intervention']]

    # Drop rows with missing symptoms, as they are crucial for the recommender
    df.dropna(subset=['Symptoms'], inplace=True)
    df = df.reset_index(drop=True)

    # Encode the symptoms using the pre-loaded SentenceTransformer model
    symptom_embeddings = _model.encode(df['Symptoms'].tolist(), convert_to_tensor=True)
    
    return df, symptom_embeddings

# =========================================
# 🔹 Recommendation Function
# =========================================

def recommend_bert(symptom_query, top_k=3):
    """
    Finds the top_k most similar diseases based on the symptom query.
    """
    # Encode the user's query
    query_embedding = model.encode(symptom_query, convert_to_tensor=True)

    # Calculate cosine similarity between the query and all symptom embeddings
    cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]

    # Get the top k results
    top_results = torch.topk(cos_scores, k=top_k)

    # Format and return the results
    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        row = df.iloc[idx.item()]
        results.append({
            "Disease": row['Disease'],
            "Formulation": row['Formulation'],
            "Diet & Lifestyle": row['Diet and Lifestyle Recommendations'],
            "Yoga": row['Yoga & Physical Therapy'],
            "Medical Intervention": row['Medical Intervention'],
            "Similarity Score": f"{score.item():.2f}" # Include the confidence score
        })
    return results

# =========================================
# 🔹 Streamlit User Interface
# =========================================

# --- Page Configuration ---
st.set_page_config(
    page_title="AyurGenixAI Recommender",
    page_icon="🌿",
    layout="centered"
)

# --- Load Models and Data ---
model = load_bert_model()
df, embeddings = load_data_and_embeddings(model)

# --- App Title and Description ---
st.title("🌿 AyurGenixAI Recommender")
st.markdown("Enter your symptoms below to get personalized Ayurvedic and medical recommendations. This tool uses AI to find the closest matches from our dataset.")

# --- User Input ---
symptom_input = st.text_area("Describe your symptoms (e.g., 'frequent urination and fatigue')", height=100)

# --- Recommendation Button and Logic ---
if st.button("Get Recommendations", type="primary"):
    if df is not None and embeddings is not None: # Check if data loaded correctly
        if symptom_input:
            with st.spinner('Analyzing your symptoms...'):
                recommendations = recommend_bert(symptom_input)
            
            st.subheader("Here are your top recommendations:")
            
            for rec in recommendations:
                with st.expander(f"**{rec['Disease']}** (Similarity: {rec['Similarity Score']})"):
                    st.markdown(f"**🌿 Ayurvedic Formulation:** {rec['Formulation']}")
                    st.markdown(f"**🥗 Diet & Lifestyle:** {rec['Diet & Lifestyle']}")
                    st.markdown(f"**🧘 Yoga & Therapy:** {rec['Yoga']}")
                    st.markdown(f"**⚕️ Medical Intervention:** {rec['Medical Intervention']}")
        else:
            st.warning("Please enter your symptoms to get a recommendation.")
            
# --- Disclaimer ---
st.markdown("---")

st.warning("**Disclaimer:** This is an AI-powered informational tool and not a substitute for professional medical advice. Please consult a qualified healthcare provider for any health concerns.", icon="⚠️")
