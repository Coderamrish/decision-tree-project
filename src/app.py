import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from preprocess import load_and_preprocess

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
gini_model = joblib.load("../models/tree_gini.pkl")
entropy_model = joblib.load("../models/tree_entropy.pkl")

models = {"Gini": gini_model, "Entropy": entropy_model}

def preprocess_uploaded_data(df):
    """
    Preprocess the uploaded dataframe to match the training data format
    """
    from sklearn.preprocessing import LabelEncoder
    
    df_processed = df.copy()
    le = LabelEncoder()
    
    # Encode categorical columns
    for col in df_processed.columns:
        if df_processed[col].dtype == object:
            df_processed[col] = le.fit_transform(df_processed[col])
    
    # Remove target column if present
    if 'status' in df_processed.columns:
        df_processed = df_processed.drop('status', axis=1)
    
    return df_processed

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
    
    ### 📝 Input Data Format
    Your CSV should match the original training data format with columns like:
    - Age, Checking account, Credit amount, Duration, Housing, etc.
    
    The app will automatically preprocess your data to match the model's expected format.
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
            try:
                model = models[model_choice]
                
                # Preprocess the uploaded data
                df_processed = preprocess_uploaded_data(df)
                
                # Make predictions
                predictions = model.predict(df_processed)
                
                # Add predictions to original dataframe (for better readability)
                df["Prediction"] = predictions
                df["Prediction_Label"] = df["Prediction"].map({0: "Good Credit", 1: "Bad Credit"})

                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("✅ Prediction Results")
                st.dataframe(df)
                
                # Show summary
                st.write("### 📊 Summary")
                pred_counts = df["Prediction_Label"].value_counts()
                st.write(pred_counts)
                
                st.markdown("</div>", unsafe_allow_html=True)

                # Download button
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Download Predictions", csv, "predictions.csv")
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.info("💡 Make sure your CSV has the same columns as the training data.")

elif page == "📊 Model Info":
    st.markdown("<h2 class='sub-title'>📊 Model Details</h2>", unsafe_allow_html=True)

    for name, model in models.items():
        st.markdown(f"<div class='card'>", unsafe_allow_html=True)
        st.subheader(f"🌳 Decision Tree ({name})")

        st.write(f"**Depth:** {model.get_depth()}")
        st.write(f"**Number of Leaves:** {model.get_n_leaves()}")
        st.write(f"**Number of Features:** {model.n_features_in_}")
        
        # Show expected feature names
        if hasattr(model, 'feature_names_in_'):
            st.write(f"**Expected Features:** {', '.join(model.feature_names_in_)}")

        st.markdown("</div>", unsafe_allow_html=True)

elif page == "📈 Feature Importance":
    st.markdown("<h2 class='sub-title'>📈 Feature Importance</h2>", unsafe_allow_html=True)

    selected_model = st.selectbox("Select Model", ["Gini", "Entropy"])
    model = models[selected_model]

    importance = model.feature_importances_
    
    # Get feature names if available
    if hasattr(model, 'feature_names_in_'):
        features = model.feature_names_in_
    else:
        features = [f"Feature {i}" for i in range(len(importance))]

    # Create a dataframe for better visualization
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df['Feature'], importance_df['Importance'])
    ax.set_title(f"Feature Importance - {selected_model} Model")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.invert_yaxis()

    st.pyplot(fig)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📋 Feature Importance Table")
    st.dataframe(importance_df)
    st.markdown("</div>", unsafe_allow_html=True)