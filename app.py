import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import re
import string
from streamlit_option_menu import option_menu


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)


# =====================================================
# LOAD MODEL
# =====================================================

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


# =====================================================
# CSS DESIGN
# =====================================================

st.markdown("""
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

[data-testid="stAppViewContainer"] {
    background: #0f172a;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

.block-container {
    padding-top: 2rem;
}

/* Main Title */

.main-title {
    font-size: 50px;
    font-weight: bold;
    color: white;
}

.sub-text {
    color: #94a3b8;
    font-size: 18px;
}

/* Cards */

.card {
    background-color: #111827;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #1e293b;
    text-align: center;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
}

.card-title {
    color: #94a3b8;
    font-size: 15px;
}

.card-value {
    color: white;
    font-size: 32px;
    font-weight: bold;
}

/* Text Area */

.stTextArea textarea {
    background-color: #1e293b;
    color: white;
    border-radius: 15px;
    border: 1px solid #334155;
    padding: 15px;
}

/* Buttons */

.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    color: white;
    font-size: 17px;
    font-weight: 600;
}

/* Result Boxes */

.real-box {
    background-color: rgba(34,197,94,0.15);
    border: 1px solid #22c55e;
    padding: 20px;
    border-radius: 15px;
    color: #bbf7d0;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
}

.fake-box {
    background-color: rgba(239,68,68,0.15);
    border: 1px solid #ef4444;
    padding: 20px;
    border-radius: 15px;
    color: #fecaca;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# TEXT CLEANING
# =====================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r'http\S+', '', text)

    text = text.translate(str.maketrans('', '', string.punctuation))

    return text


# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_news(news):

    news = clean_text(news)

    vector = vectorizer.transform([news])

    prediction = model.predict(vector)

    probability = model.predict_proba(vector)

    confidence = round(max(probability[0]) * 100, 2)

    return prediction[0], confidence


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("## News AI")

    selected = option_menu(
        menu_title=None,
        options=["Home", "Predict", "Analytics", "About"],
        icons=["house", "search", "bar-chart", "info-circle"],
        default_index=0
    )


# =====================================================
# HOME PAGE
# =====================================================

# =====================================================
# HOME PAGE
# =====================================================

if selected == "Home":

    st.markdown("""
    <h1 style='color:white; font-size:55px;'>
    Fake News Detection System
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style='color:#94a3b8; font-size:18px;'>
    AI-powered fake news verification platform using Machine Learning.
    </p>
    """, unsafe_allow_html=True)

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="
            background:#111827;
            padding:25px;
            border-radius:20px;
            text-align:center;
            border:1px solid #1e293b;
        ">
            <h1 style='color:white;'>98%</h1>
            <p style='color:#94a3b8;'>Accuracy</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="
            background:#111827;
            padding:25px;
            border-radius:20px;
            text-align:center;
            border:1px solid #1e293b;
        ">
            <h1 style='color:white;'>TF-IDF</h1>
            <p style='color:#94a3b8;'>Vectorizer</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="
            background:#111827;
            padding:25px;
            border-radius:20px;
            text-align:center;
            border:1px solid #1e293b;
        ">
            <h1 style='color:white;'>Logistic Regression</h1>
            <p style='color:#94a3b8;'>Algorithm</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    st.image(
        "https://images.pexels.com/photos/518543/pexels-photo-518543.jpeg",
        use_container_width=True
    )


# =====================================================
# PREDICTION PAGE
# =====================================================

elif selected == "Predict":

    st.title("News Verification")

    news_input = st.text_area(
        "Paste News Article",
        height=250,
        placeholder="Paste your article here..."
    )

    if st.button("Analyze News"):

        if news_input.strip() == "":

            st.warning("Please enter article text.")

        else:

            prediction, confidence = predict_news(news_input)

            st.subheader("Prediction")

            if prediction == 0:

                st.markdown("""
                <div class="fake-box">
                    Fake News Detected
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown("""
                <div class="real-box">
                    Real News Detected
                </div>
                """, unsafe_allow_html=True)

            st.write("")

            st.subheader("Confidence Score")

            st.progress(int(confidence))

            st.write(f"Confidence: {confidence}%")


# =====================================================
# ANALYTICS PAGE
# =====================================================

elif selected == "Analytics":

    st.title("Analytics Dashboard")

    data = pd.DataFrame({
        "Category": ["Fake News", "Real News"],
        "Count": [23481, 21417]
    })

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            data,
            x="Category",
            y="Count",
            color="Category",
            text="Count",
            title="News Distribution"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(color="white")
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        pie = px.pie(
            data,
            names="Category",
            values="Count",
            hole=0.5,
            title="Dataset Ratio"
        )

        pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            font=dict(color="white")
        )

        st.plotly_chart(pie, use_container_width=True)


# =====================================================
# ABOUT PAGE
# =====================================================

elif selected == "About":

    st.title("About Project")

    st.markdown("""
    ### Project Description

    This project detects whether a news article is real or fake
    using Machine Learning and NLP techniques.

    ### Features

    - AI News Detection
    - Confidence Score
    - Interactive Dashboard
    - Analytics Visualization
    - Professional UI

    ### Technologies

    - Python
    - Streamlit
    - Scikit-learn
    - TF-IDF
    - Logistic Regression
    - Plotly
    """)