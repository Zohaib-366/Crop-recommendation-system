import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Streamlit Page Configuration
st.set_page_config(page_title="Crop Recommendation App", layout="centered")

# Custom CSS Styling
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #2e8b57;
    }
    .sub-title {
        text-align: center;
        font-size: 18px;
        color: #555;
    }
    .stButton>button {
        background-color: #2e8b57;
        color: white;
        border-radius: 10px;
        font-weight: bold;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #1e683d;
    }
    .stNumberInput>div>input {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Introduction", "EDA", "Predict"])

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# Page logic
# Overview Page
if app_mode == "Introduction":
    st.title("🌾 Crop Recommendation System")

    st.markdown("""
    ### 📌 Project Overview  
    This intelligent system analyzes environmental and soil conditions to recommend the most suitable crop for cultivation using machine learning.
    
    ---

    ### 🎯 Objective  
    To assist farmers and agriculture planners by identifying the most appropriate crop based on real-time agro-climatic conditions, improving yield and sustainability.
    
    ---

    ### 📂 Dataset Information  
    | 📄 Total Records | 📊 Total Features | 🌿 Target Crops |
    |------------------|-------------------|-----------------|
    | 2200             | 7                 | 22              |

    """)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())



    st.markdown("""
    ### 🔍 Sample Data Insights  
    **🎯 Target Variable:**  
    - `Crop`: Represents the most suitable crop for given soil and climate parameters.

    **🧬 Input Features:**  
    - **🌱 Soil Nutrients:**  
      - Nitrogen (N), Phosphorus (P), Potassium (K)

    - **🌡️ Climate Metrics:**  
      - Temperature (°C), Humidity (%), Rainfall (mm)

    - **🧪 Soil Acidity:**  
      - pH value

    ---

    ### 🎯 Project Goals  
    - 📈 Understand how environmental factors affect crop selection.  
    - 🌾 Provide intelligent crop recommendations to boost farming efficiency.  
    - 🧠 Enable data-driven decision-making in agriculture through predictive modeling.
    
    ---

   ### ⚙️ Model Used  
    - ✅ **Algorithm:** Random Forest Classifier  
    - 🔢 **Preprocessing:** StandardScaler for feature scaling  
    - 🏷️ **Encoding:** LabelEncoder for converting crop labels to numerical format  
    - 📊 **Output:** Most suitable crop + top 5 crop probabilities  
    - 🧪 **Performance:** High accuracy with generalizability across varied conditions

    This model learns from environmental and soil parameters and predicts the crop that has historically performed best under similar conditions.
    
    """)
    # Optional: Explanation of Random Forest
    with st.expander("📘 What is a Random Forest?"):
        st.markdown("""
        A **Random Forest** is an ensemble learning technique that builds multiple decision trees and merges them to get a more accurate and stable prediction.  
        It handles both classification and regression problems and reduces overfitting compared to a single decision tree.
        """)

    # Model Performance
    st.subheader("📊 Model Performance")
    st.markdown("""
    - **Accuracy:** 98.5%  
    - **Evaluation:** Cross-Validation (5-fold)  
    - **Metrics Used:** Accuracy, Precision, Recall, F1-score
    """)



elif app_mode == "EDA":
    import seaborn as sns
    import matplotlib.pyplot as plt

    st.title("🔍 Exploratory Data Analysis")
    st.markdown("Explore the dataset using interactive dropdowns and visual insights.")

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    # ------------------- 1. Summary Statistics -------------------
    with st.expander("📊 Summary Statistics"):
        st.markdown("""
        Understanding central tendencies, spread, and other statistical properties of each numerical feature.
        """)
        st.dataframe(df.describe())

    # ------------------- 2. Unique Value Count -------------------
    with st.expander("📌 Unique Value Count per Column"):
        st.markdown("Helpful for identifying categorical diversity and duplicate values.")
        st.dataframe(df.nunique())

    # ------------------- 3. Feature Distributions -------------------
    with st.expander("📈 Feature Distribution (Histogram + KDE)"):
        st.markdown("""
        ### 📊 Why This Matters:
        - Helps visualize the spread and skew of numeric data.
        - Detects potential outliers and unusual distributions.
        - Useful for understanding normalization needs before ML modeling.
        """)
        mode = st.radio("Choose mode", ["All Features", "Single Feature"], horizontal=True)

        if mode == "Single Feature":
            selected_col = st.selectbox("Select feature", numeric_cols)
            bins = st.slider("Bins", 5, 50, 30)

            fig, ax = plt.subplots()
            sns.histplot(df[selected_col], kde=True, bins=bins, color='seagreen', edgecolor='black', ax=ax)
            ax.set_title(f"Distribution of {selected_col}")
            ax.grid(True)
            st.pyplot(fig)

        elif mode == "All Features":
            cols = st.slider("Columns in grid", 2, 4, 3)
            rows = -(-len(numeric_cols) // cols)

            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
            axes = axes.flatten()

            for i, col in enumerate(numeric_cols):
                sns.histplot(df[col], kde=True, bins=30, color='seagreen', edgecolor='black', ax=axes[i])
                axes[i].set_title(col)
                axes[i].grid(True)

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            st.pyplot(fig)

    # ------------------- 4. Outlier Detection (Boxplots) -------------------
    with st.expander("📦 Outlier Detection using Boxplots"):
        st.markdown("""
        ### 📌 Why This Matters:
        - Boxplots help identify outliers using the IQR method.
        - Useful for data cleaning and feature scaling decisions.
        """)
        mode = st.radio("Choose mode", ["All Features", "Single Feature"], horizontal=True, key="box_mode")

        if mode == "Single Feature":
            selected_col = st.selectbox("Select feature", numeric_cols, key="box_feature")

            fig, ax = plt.subplots()
            sns.boxplot(y=df[selected_col], color='lightblue', ax=ax)
            ax.set_title(f"Boxplot of {selected_col}")
            st.pyplot(fig)

        elif mode == "All Features":
            cols = st.slider("Columns in grid", 2, 4, 3, key="box_col_slider")
            rows = -(-len(numeric_cols) // cols)

            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
            axes = axes.flatten()

            for i, col in enumerate(numeric_cols):
                sns.boxplot(y=df[col], color='lightblue', ax=axes[i])
                axes[i].set_title(col)

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            st.pyplot(fig)

    # ------------------- 5. Correlation Heatmap -------------------
    with st.expander("🧩 Correlation Heatmap"):
        st.markdown("""
        ### 📊 What We'll Use:
        - **Heatmap of correlation matrix** — to visualize the strength and direction of linear relationships.

        ### 📊 Why This Matters:
        - Shows **multicollinearity** — features that are highly correlated with each other.
        - Helps identify **important predictors** or **redundant features**.
        """)
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title("📌 Correlation Matrix")
        st.pyplot(fig)

    # ------------------- 6. Grouped Feature Means by Crop -------------------
    with st.expander("🌾 Grouped Feature Averages by Crop"):
        st.markdown("""
        Shows the average value of each numerical feature per crop type.
        Useful to understand how each crop prefers different ranges of features like temperature or pH.
        """)
        crop_means = df.groupby("label")[numeric_cols].mean().sort_index()
        st.dataframe(crop_means.style.background_gradient(cmap="YlGnBu"))

    # ------------------- 7. Pairplot -------------------
    with st.expander("🔗 Pairwise Feature Relationships (Pairplot)"):
        st.markdown("""
        ### 📊 What We'll Use:
        - **Pairplot** — a grid of scatterplots showing relationships between selected features.

        ### 📊 Why This is Useful:
        - Helps detect **natural groupings** or **visual separability** between crops.
        - Shows **linear and non-linear** relationships.
        - Aids in **feature selection** for classification tasks.
        """)
        selected = st.multiselect("Choose 2–4 features", numeric_cols, default=["temperature", "humidity", "ph", "rainfall"])
        if 2 <= len(selected) <= 4:
            sample_df = df.sample(n=min(500, len(df)), random_state=42)
            fig = sns.pairplot(sample_df[selected + ['label']], hue='label', diag_kind='kde', palette='tab20')
            st.pyplot(fig)
        else:
            st.warning("Select at least 2 and at most 4 features.")

    # ------------------- 8. Crop Count Distribution -------------------
    with st.expander("🌱 Crop Distribution Count"):
        st.markdown("""
        Shows the number of records per crop label. 
        Useful to detect class imbalance in classification problems.
        """)
        crop_counts = df['label'].value_counts()
        st.bar_chart(crop_counts)  

# Prediction Page
elif app_mode == "Predict":
    st.title("🌾 Intelligent Crop Predictor")

    # Load model, scaler, and label encoder
    model = joblib.load("crop_recommendation_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    # Emoji map
    crop_emojis = {
        "rice": "🌾", "maize": "🌽", "chickpea": "🥣", "kidneybeans": "🫘",
        "pigeonpeas": "🟤", "mothbeans": "🥬", "mungbean": "🌿", "blackgram": "🖤",
        "lentil": "🍲", "pomegranate": "🍎", "banana": "🍌", "mango": "🥭",
        "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈", "apple": "🍏",
        "orange": "🍊", "papaya": "🍐", "coconut": "🥥", "cotton": "🧵",
        "jute": "🪢", "coffee": "☕"
    }

    st.markdown("## 📥 Soil Nutrients")
    col1, col2, col3 = st.columns(3)
    N = col1.number_input("Nitrogen (N)", min_value=0, max_value=140, value=60, step=1, help="Nitrogen level in the soil (0–140 ppm)")
    P = col2.number_input("Phosphorous (P)", min_value=0, max_value=145, value=45, step=1, help="Phosphorous level in the soil (0–145 ppm)")
    K = col3.number_input("Potassium (K)", min_value=0, max_value=205, value=50, step=1, help="Potassium level in the soil (0–205 ppm)")

    st.markdown("## 🌡️ Climate Conditions")
    col1, col2, col3 = st.columns(3)
    temperature = col1.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=1.0, help="Average temperature of the region (0–50°C)")
    humidity = col2.number_input("Humidity (%)", min_value=10.0, max_value=100.0, value=60.0, step=1.0, help="Relative humidity percentage (10–100%)")
    rainfall = col3.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0, step=1.0, help="Expected rainfall in millimeters (0–300 mm)")

    st.markdown("## 🧪 Soil Acidity")
    ph = st.number_input("Soil pH", min_value=3.0, max_value=10.0, value=6.5, step=0.1, help="Soil pH value (3.0–10.0), where 7 is neutral")

    st.markdown("---")
    if st.button("🌿 Recommend Best Crop"):
        input_data = [[N, P, K, temperature, humidity, ph, rainfall]]
        input_scaled = scaler.transform(input_data)
        prediction_encoded = model.predict(input_scaled)[0]
        crop_name = label_encoder.inverse_transform([prediction_encoded])[0]

        emoji = crop_emojis.get(crop_name.lower(), "🌱")
        st.success(f"### ✅ Recommended Crop: {emoji} **{crop_name.upper()}**")

        # Top 5 predictions
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_scaled)[0]
            labels_decoded = label_encoder.inverse_transform(np.arange(len(probs)))
            prob_df = pd.DataFrame({'Crop': labels_decoded, 'Probability': probs})
            prob_df_sorted = prob_df.sort_values(by='Probability', ascending=False).head(5)
            prob_df_sorted.index = np.arange(1, len(prob_df_sorted) + 1)  # 1-based index

            st.subheader("📊 Top 5 Most Suitable Crops")
            st.dataframe(prob_df_sorted.style.bar(subset=["Probability"], color='lightgreen'))
