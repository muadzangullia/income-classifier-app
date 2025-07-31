import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Income Classifier", 
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 3rem;
    }
    .prediction-high {
        background: linear-gradient(135deg, #48c774, #55d882);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3em;
        margin: 2rem 0;
    }
    .prediction-low {
        background: linear-gradient(135deg, #ff6b6b, #ff8e8e);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3em;
        margin: 2rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 1rem 3rem;
        font-weight: bold;
        font-size: 1.4em;
        width: 100%;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6);
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%) !important;
    }
    .stButton > button:active {
        transform: translateY(-1px);
    }
    .section-spacing {
        margin: 2rem 0;
    }
    .center-button {
        display: flex;
        justify-content: center;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model = joblib.load("income_model.pkl")
    feature_names = joblib.load("features.pkl")
    return model, feature_names

model, feature_names = load_model()

# Sidebar
with st.sidebar:
    st.markdown("## 📊 Model Information")
    st.markdown("""
    **Algorithm:** Gradient Boosting Classifier  
    **Accuracy:** 85.2%  
    **Precision (>50K):** 67.4%  
    **Recall (>50K):** 73.9%  
    **Features:** 19 selected predictors
    """)

    st.markdown("## 🎯 Key Predictors")
    st.markdown("""
    - Age
    - Years of Education
    - Gender
    - Capital Gain / Loss
    - Weekly Work Hours
    - Work Sector
    - Marital Status
    - Occupation
    - Relationship
    """)

# Header
st.markdown("""
<div class="main-header">
    <h1>💰 Income Prediction App</h1>
    <p>Predict whether someone earns more than $50K annually</p>
</div>
""", unsafe_allow_html=True)

# Input Form
with st.form("prediction_form"):
    st.markdown('<div class="section-spacing">', unsafe_allow_html=True)
    st.markdown("### 👤 Personal Information")
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        age = st.slider("Age", 17, 90, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        education_num = st.slider("Years of Education", 1, 16, 10)
        hours = st.slider("Hours Worked per Week", 1, 100, 40)
    with col2:
        capital_gain = st.number_input("Capital Gain ($)", min_value=0, step=100, value=0)
        capital_loss = st.number_input("Capital Loss ($)", min_value=0, step=100, value=0)
        marital_status = st.selectbox("Marital Status", [
            'Married (Civil)', 'Never Married'
        ])
        relationship = st.selectbox("Relationship", [
            'Not in Family', 'Unmarried', 'Wife'
        ])

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-spacing'>", unsafe_allow_html=True)
    st.markdown("### 💼 Employment Details")
    st.markdown("---")
    col3, col4 = st.columns([1, 1])
    with col3:
        workclass = st.selectbox("Work Sector", ['Private', 'Self-Employed'])
    with col4:
        occupation = st.selectbox("Occupation", [
            'Craft or Repair', 'Farming or Fishing', 'Machine Operator',
            'Other Services', 'Transport or Moving', 'Unemployed'
        ])

    st.markdown('<div class="center-button">', unsafe_allow_html=True)
    submitted = st.form_submit_button("🚀 Predict Income Level")
    st.markdown('</div>', unsafe_allow_html=True)

if submitted:
    with st.spinner("🧐 Making prediction..."):
        input_data = {
            "age": age,
            "educational-num": education_num,
            "gender": 1 if gender == "Female" else 0,
            "capital-gain": capital_gain,
            "capital-loss": capital_loss,
            "hours-per-week": hours
        }

        # One-hot encode selected features
        mapped = {
            "Married (Civil)": "marital-status_Married-civ-spouse",
            "Never Married": "marital-status_Never-married",
            "Private": "workclass_Private",
            "Self-Employed": "workclass_Self-emp-not-inc",
            "Craft or Repair": "occupation_Craft-repair",
            "Farming or Fishing": "occupation_Farming-fishing",
            "Machine Operator": "occupation_Machine-op-inspct",
            "Other Services": "occupation_Other-service",
            "Transport or Moving": "occupation_Transport-moving",
            "Unemployed": "occupation_Unknown",
            "Not in Family": "relationship_Not-in-family",
            "Unmarried": "relationship_Unmarried",
            "Wife": "relationship_Wife"
        }

        for label in [marital_status, workclass, occupation, relationship]:
            key = mapped.get(label)
            if key:
                input_data[key] = 1

        # Fill missing dummy columns with 0s
        X_input = pd.DataFrame([input_data])
        for col in feature_names:
            if col not in X_input.columns:
                X_input[col] = 0
        X_input = X_input[feature_names]

        # Predict
        pred = model.predict(X_input)[0]
        pred_proba = model.predict_proba(X_input)[0]
        label = ">50K" if pred == 1 else "≤$50K"
        confidence = max(pred_proba) * 100

        # Show result
        st.markdown('<div class="section-spacing">', unsafe_allow_html=True)
        st.markdown("## 🌟 Prediction Results")
        if pred == 1:
            st.markdown(f"""
            <div class="prediction-high">
                <h2>💰 High Income Predicted!</h2>
                <p><strong>Predicted Income: {label}</strong></p>
                <p><strong>Confidence: {confidence:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-low">
                <h2>📊 Lower Income Predicted</h2>
                <p><strong>Predicted Income: {label}</strong></p>
                <p><strong>Confidence: {confidence:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)

        # Probability Chart
        st.markdown("<div class='section-spacing'>", unsafe_allow_html=True)
        st.markdown("### 📊 Probability Distribution")
        fig = go.Figure(data=[
            go.Bar(x=['≤$50K', '>$50K'],
                   y=[pred_proba[0]*100, pred_proba[1]*100],
                   marker_color=['#ff6b6b', '#48c774'],
                   text=[f'{pred_proba[0]*100:.1f}%', f'{pred_proba[1]*100:.1f}%'],
                   textposition='auto')
        ])
        fig.update_layout(
            title="Income Prediction Confidence",
            yaxis_title="Probability (%)",
            height=400,
            showlegend=False,
            font=dict(size=14)
        )
        st.plotly_chart(fig, use_container_width=True)



