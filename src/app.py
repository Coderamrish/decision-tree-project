import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Decision Tree Classifier",
    page_icon="🌳",
    layout="wide"
)

st.markdown("""
    <style>
    .main-title {
        font-size: 40px !important;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #00b4d8, #0077b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 20px;
    }
    .sub-title {
        font-size: 22px !important;
        font-weight: 600;
        margin-top: 20px;
    }
    .card {
        padding: 25px;
        border-radius: 12px;
        background: #ffffff;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "📤 Upload Data", "📊 Model Info", "📈 Feature Importance"])

# Load models
gini_model = joblib.load("models/tree_gini.pkl")
entropy_model = joblib.load("models/tree_entropy.pkl")

models = {"Gini": gini_model, "Entropy": entropy_model}

if page == "🏠 Home":
    st.markdown("<h1 class='main-title'>Decision Tree Credit Risk Classifier 🌳</h1>", unsafe_allow_html=True)

    st.markdown("""
    ### 📌 Overview  
    This project uses **Decision Tree Classifiers** (Gini & Entropy) to predict credit risk.

    ### 🚀 Features
    - Upload custom CSV  
    - Select model  
    - Get prediction output instantly  
    - Visualize feature importance  
    - Download prediction results  
    """)

    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Decision_Tree.jpg/640px-Decision_Tree.jpg")

elif page == "📤 Upload Data":
    st.markdown("<h2 class='sub-title'>📤 Upload Dataset for Prediction</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    model_choice = st.selectbox("Choose Model", ["Gini", "Entropy"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(df.head())
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🔮 Predict"):
            model = models[model_choice]
            predictions = model.predict(df)

            df["Prediction"] = predictions

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("✅ Prediction Results")
            st.dataframe(df)
            st.markdown("</div>", unsafe_allow_html=True)

            # Download button
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download Predictions", csv, "predictions.csv")

elif page == "📊 Model Info":
    st.markdown("<h2 class='sub-title'>📊 Model Details</h2>", unsafe_allow_html=True)

    for name, model in models.items():
        st.markdown(f"<div class='card'>", unsafe_allow_html=True)
        st.subheader(f"🌳 Decision Tree ({name})")

        st.write(f"**Depth:** {model.get_depth()}")
        st.write(f"**Number of Leaves:** {model.get_n_leaves()}")
        st.write(f"**Number of Features:** {model.n_features_in_}")

        st.markdown("</div>", unsafe_allow_html=True)

elif page == "📈 Feature Importance":
    st.markdown("<h2 class='sub-title'>📈 Feature Importance</h2>", unsafe_allow_html=True)

    selected_model = st.selectbox("Select Model", ["Gini", "Entropy"])
    model = models[selected_model]

    importance = model.feature_importances_
    features = range(len(importance))

    fig, ax = plt.subplots()
    ax.bar(features, importance)
    ax.set_title("Feature Importance")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Importance")

    st.pyplot(fig)
