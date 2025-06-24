import streamlit as st
st.set_page_config(
    page_title="TruDiagnosis",
    page_icon="healthcare.png",  
)

st.markdown("""
<style>
/* 1. Make header transparent */
[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* 2. Hide ALL right-side elements (including 3-dot menu) */
[data-testid="stToolbarActions"], 
[data-testid="baseButton-headerNoPadding"],
[data-testid="stDecoration"] {
    display: none !important;
}

/* 3. Protect and position the sidebar arrow */
[data-testid="collapsedControl"] {
    display: block !important;
    position: absolute !important;
    left: 0px !important;
    z-index: 9999 !important;
}

/* 4. Hide other Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

import pandas as pd
import joblib
from PIL import Image
from streamlit_option_menu import option_menu
from model_codes.DiseaseModel import DiseaseModel
from model_codes.helper import prepare_symptoms_array
# Load models
diabetes_model = joblib.load("models/diabetes_model.sav")
heart_model = joblib.load("models/heart_disease_model.sav")
liver_model = joblib.load("models/liver_model.sav")
breast_cancer_model = joblib.load("models/breast_cancer.sav")
covid_model = joblib.load("models/covid19_trained.pkl")
lung_model = joblib.load("models/lung_trained.pkl")

st.markdown("""
    <style>
    /* === HEADER === */
    .app-header {
        background-color: #8B0000;
        padding: 1rem 2rem;
        color: white;
        font-size: 28px;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 100;
    }

    /* === BACKGROUND === */
    html, body, .stApp {
        background-color: #dbabab !important;
        font-family: 'Open Sans', sans-serif;
        margin: 0;
    }

    /* === INPUT FIELDS (fixed!) === */
    input[type="text"], input[type="number"] {
        background-color: #ffffff !important;
        color: #4a0000 !important;
        border: 1.5px solid #ff9999 !important;
        border-radius: 10px !important;
        padding: 8px 10px !important;
        caret-color: #8B0000 !important;
        outline: none !important;  /* 🛠️ Removes second red outline */
    }

    /* === PLACEHOLDER FIX === */
    input::placeholder {
        color: #a94442 !important;
        font-weight: 500 !important;
        opacity: 1 !important;
    }

    /* === SELECTED TEXT IN DROPDOWNS === */
    div[data-baseweb="select"] *,
    div[data-baseweb="input"] * {
        color: #4a0000 !important;
        background-color: white !important;
    }

    .main > div {
        padding-top: 100px;
    }

    /* === SIDEBAR === */
    section[data-testid="stSidebar"] {
        background-color: #b30000 !important;
    }

    .css-1v3fvcr, .css-10trblm, .css-1cpxqw2, .css-1d391kg, .css-1v0mbdj {
        color: #ffffff !important;
        font-weight: 600;
    }

    .block-container {
        padding: 2rem 3rem;
    }

    /* === LABEL TEXT === */
    h1, h2, h3, .stRadio label, .stNumberInput label, label, .css-1cpxqw2 label {
        color: #8B0000 !important;
        font-weight: 600 !important;
    }

    /* === RADIO BUTTONS === */
    .stRadio div[role="radiogroup"] > label {
        display: inline-block;
        margin-right: 20px;
    }

    /* === INPUTS INSIDE ST COMPONENTS === */
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"],
    div[data-baseweb="radio"] > div {
        background-color: white !important;
        color: #8B0000 !important;
        border: 1.5px solid #ff9999 !important;
        border-radius: 8px;
        padding: 5px;
    }

    /* === BUTTON STYLING === */
    .stButton > button {
        background-color: #cc0000 !important;
        color: white !important;
        font-weight: bold;
        padding: 8px 20px;
        border: none;
        border-radius: 6px;
        transition: 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #8B0000 !important;
    }

    /* === IMAGES === */
    .stImage img {
        border-radius: 10px;
        margin: 1rem auto;
        display: block;
        max-width: 100%;
        height: auto;
    }

    /* === ALERTS === */
    .stAlert.success {
        background-color: #691611 !important;
        color: #800000 !important;
        font-weight: 600;
    }

    /* === SLIDER FIXES === */
    div.stSlider > div[data-baseweb="slider"] > div[data-testid="stTickBar"] > div {
        background: transparent;
    }

    div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"] {
        background-color: white;
        box-shadow: rgba(139, 0, 0, 0.2) 0px 0px 0px 0.2rem;
    }

    div.stSlider > div[data-baseweb="slider"] > div > div > div > div {
        color: #8B0000;
    }

    /* === SLIDER LABEL TEXT === */
    div[data-testid="stTickBar"] svg text {
        fill: #8B0000 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }
    </style>

""", unsafe_allow_html=True)


# === Header ===
st.markdown("""
    <style>
    .app-header {
        background-color: #8B0000;
        padding: 1rem 2rem;
        color: white;
        font-size: 28px;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 100;
    }
    .header-desktop {
        display: block;
    }
    .header-mobile {
        display: none;
    }

    @media screen and (max-width: 768px) {
        .header-desktop {
            display: none;
        }
        .header-mobile {
            display: block;
        }
    }
    </style>

    <div class="app-header">
        <div class="header-desktop">TruDiagnosis — an all-in-one health diagnostic app</div>
        <div class="header-mobile">TruDiagnosis</div>
    </div>
""", unsafe_allow_html=True)


# # === Sidebar ===
# selected = st.sidebar.radio("Choose Assessment:", [
#     'Diabetes Prediction',
#     'Heart Disease Prediction',
#     'Liver Prediction',
#     'Breast Cancer Prediction',
#     'Covid-19 Prediction',
#     'Lung Cancer Prediction'
# ])


with st.sidebar:
    selected = option_menu(
        menu_title="Choose Assessment",
        options=[
            'General Disease Prediction',
            'Diabetes Prediction',
            'Heart Disease Prediction',
            'Liver Prediction',
            'Breast Cancer Prediction',
            'Covid-19 Prediction',
            'Lung Cancer Prediction',
            'Pneumonia Prediction'
            
        ],
        icons=['activity','activity', 'heart-pulse', 'droplet', 'heart', 'virus', 'lungs','lungs'],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "5px", "background-color": "#581818"},
            "icon": {"color": "#ffffff", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "color": "#ffffff",
                "border-radius": "8px"
            },
            "nav-link-selected": {
                "background-color": "#943131",
                "font-weight": "bold",
                "color": "#ffffff"
            }
        }
    )

if selected == 'General Disease Prediction': 
    # Load model
    disease_model = DiseaseModel()
    disease_model.load_xgboost('models/xgboost_model.json')

    # Page title
    st.header("AI-Based General Disease Identifier")

    # Symptom input
    st.subheader("Select your symptoms below:")
    selected_symptoms = st.multiselect(
        "Symptoms", 
        options=disease_model.all_symptoms,
        help="Choose one or more symptoms you're currently experiencing."
    )

    # Prepare input for model
    input_vector = prepare_symptoms_array(selected_symptoms)

    # Predict button
    if st.button("Analyze and Predict"):
        # Run prediction
        predicted_disease, confidence = disease_model.predict(input_vector)

        st.markdown(f"""
            ### 🧠 Prediction Result  
            **Condition Detected:** `{predicted_disease}`  
            **Confidence:** `{confidence*100:.2f}%`
        """)

        # Tabs for details
        tab1, tab2 = st.tabs(["🧾 What it Means", "🛡️ Suggested Precautions"])

        with tab1:
            st.info(disease_model.describe_predicted_disease())

        with tab2:
            precautions = disease_model.predicted_disease_precautions()
            st.success("Here are some general precautions you can follow:")
            for idx, tip in enumerate(precautions, 1):
                st.write(f"**{idx}.** {tip}")


# Sidebar menu with tiles
# with st.sidebar:
#     selected = option_menu(
#         menu_title="Choose Assessment",
#         options=[
#             'Diabetes Prediction',
#             'Heart Disease Prediction',
#             'Liver Prediction',
#             'Breast Cancer Prediction',
#             'Covid-19 Prediction',
#             'Lung Cancer Prediction'
#         ],
#         icons=['activity', 'heart-pulse', 'droplet', 'female', 'virus', 'lungs'],
#         menu_icon="cast",
#         default_index=0,
#         orientation="vertical",
#         styles={
#             "container": {"padding": "5px", "background-color": "#003366"},
#             "icon": {"color": "#ffffff", "font-size": "20px"},
#             "nav-link": {
#                 "font-size": "16px",
#                 "text-align": "left",
#                 "margin": "5px",
#                 "color": "#ffffff",
#                 "border-radius": "8px"
#             },
#             "nav-link-selected": {
#                 "background-color": "#005eb8",
#                 "font-weight": "bold",
#                 "color": "#ffffff"
#             }
#         }
#     )

# ========== DIABETES ==========
if selected == 'Diabetes Prediction':
    st.header("Diabetes Risk Assessment")
    image = Image.open('dd.jpg')
    st.image(image)
    df = pd.read_csv("data/diabetes.csv")
    features = df.columns[:-1]
    inputs = [st.text_input(f, value="", placeholder="Enter value") for f in features]
    if st.button("Predict Diabetes"):
        try:
            input_data = [float(i) for i in inputs]
            prediction = diabetes_model.predict([input_data])
            if prediction[0] == 0:
                image = Image.open("negative.png")
                st.image(image, caption="")
                st.success("✅ Not Diabetic")
            else:
                image = Image.open("positive.png")
                st.image(image, caption="")
                st.warning("⚠️ You may be Diabetic")

        except:
            st.error("❌ Please enter valid numbers.")

# ========== HEART ==========
elif selected == 'Heart Disease Prediction':
    st.header("Heart Disease Risk Assessment")
    image = Image.open('heart2.jpg')
    st.image(image)
    # Read dataset and get column names
    df = pd.read_csv("data/heart-disease.csv")
    features = df.columns[:-1]

    # Label mappings for display
    heart_feature_labels = {
        'age': "Age (in years)",
        'sex': "Sex",
        'cp': "Chest Pain Type",
        'trestbps': "Resting Blood Pressure (mm Hg)",
        'chol': "Cholesterol (mg/dl)",
        'fbs': "Fasting Blood Sugar > 120 mg/dl",
        'restecg': "Resting ECG Result",
        'thalach': "Max Heart Rate Achieved",
        'exang': "Exercise Induced Angina",
        'oldpeak': "ST Depression Induced by Exercise",
        'slope': "Slope of the Peak Exercise ST Segment",
        'ca': "Number of Major Vessels Colored by Fluoroscopy",
        'thal': "Thalassemia Type"
    }

    # Dropdown choices for categorical fields
    dropdown_choices = {
        'sex': {"Male": 1, "Female": 0},
        'cp': {"Typical Angina (0)": 0, "Atypical Angina (1)": 1, "Non-anginal Pain (2)": 2, "Asymptomatic (3)": 3},
        'fbs': {"Yes (1)": 1, "No (0)": 0},
        'restecg': {"Normal (0)": 0, "ST-T Abnormality (1)": 1, "LV Hypertrophy (2)": 2},
        'exang': {"Yes (1)": 1, "No (0)": 0},
        'slope': {"Upsloping (0)": 0, "Flat (1)": 1, "Downsloping (2)": 2},
        'ca': {"0": 0, "1": 1, "2": 2, "3": 3},
        'thal': {"Normal (1)": 1, "Fixed Defect (2)": 2, "Reversible Defect (3)": 3}
    }

    inputs = []

    for col in features:
        label = heart_feature_labels.get(col, col)
        if col in dropdown_choices:
            choice = st.selectbox(label, list(dropdown_choices[col].keys()))
            value = dropdown_choices[col][choice]
        else:
            value = st.text_input(label, placeholder="Enter value", key=col)
        inputs.append(value)

    if st.button("Predict Heart Disease"):
        try:
            values = [float(i) for i in inputs]
            prediction = heart_model.predict([values])
            if prediction[0] == 0:
                image = Image.open("negative.png")
                st.image(image, caption="")
                st.success("✅ No Heart Disease")
            else:
                image = Image.open("positive.png")
                st.image(image, caption="")
                st.warning("⚠️ Heart Disease Detected")

        except:
            st.error("❌ Please enter valid numeric values for all fields.")


# ========== LIVER ==========
elif selected == 'Liver Prediction':
    st.header("Liver Disease Prediction")
    image = Image.open('li.jpg')
    st.image(image)
    # Replace number_input with text_input to remove +/–
    Sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    age = st.text_input("Age", placeholder="e.g. 45")
    Total_Bilirubin = st.text_input("Total Bilirubin", placeholder="e.g. 1.2")
    Direct_Bilirubin = st.text_input("Direct Bilirubin", placeholder="e.g. 0.4")
    Alkaline_Phosphotase = st.text_input("Alkaline Phosphotase", placeholder="e.g. 250")
    Alamine_Aminotransferase = st.text_input("Alamine Aminotransferase", placeholder="e.g. 30")
    Aspartate_Aminotransferase = st.text_input("Aspartate Aminotransferase", placeholder="e.g. 45")
    Total_Protiens = st.text_input("Total Proteins", placeholder="e.g. 6.5")
    Albumin = st.text_input("Albumin", placeholder="e.g. 3.4")
    Albumin_and_Globulin_Ratio = st.text_input("Albumin and Globulin Ratio", placeholder="e.g. 1.1")


    if st.button("Predict Liver Disease"):
        try:
            input_df = pd.DataFrame([[
                Sex, float(age), float(Total_Bilirubin), float(Direct_Bilirubin),
                float(Alkaline_Phosphotase), float(Alamine_Aminotransferase),
                float(Aspartate_Aminotransferase), float(Total_Protiens),
                float(Albumin), float(Albumin_and_Globulin_Ratio)
            ]])
            prediction = liver_model.predict(input_df)
            if prediction[0] == 1:
                image = Image.open("positive.png")
                st.image(image, caption="")
                st.warning("Sorry but you have Liver Disease 😥")
            else:
                image = Image.open("negative.png")
                st.image(image, caption="")
                st.success("No Liver Disease 🎉")

        except:
            st.error("❌ Please enter valid numeric values.")



# ========== BREAST CANCER ==========
elif selected == 'Breast Cancer Prediction':
    st.title("Breast Cancer Prediction")
    image = Image.open('d.jpg')
    st.image(image)
    # Columns
    # No inputs from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        radius_mean = st.slider("Enter your Radius Mean", 6.0, 30.0, 15.0)
        texture_mean = st.slider("Enter your Texture Mean", 9.0, 40.0, 20.0)
        perimeter_mean = st.slider("Enter your Perimeter Mean", 43.0, 190.0, 90.0)

    with col2:
        area_mean = st.slider("Enter your Area Mean", 143.0, 2501.0, 750.0)
        smoothness_mean = st.slider("Enter your Smoothness Mean", 0.05, 0.25, 0.1)
        compactness_mean = st.slider("Enter your Compactness Mean", 0.02, 0.3, 0.15)

    with col3:
        concavity_mean = st.slider("Enter your Concavity Mean", 0.0, 0.5, 0.2)
        concave_points_mean = st.slider("Enter your Concave Points Mean", 0.0, 0.2, 0.1)
        symmetry_mean = st.slider("Enter your Symmetry Mean", 0.1, 1.0, 0.5)

    with col1:
        fractal_dimension_mean = st.slider("Enter your Fractal Dimension Mean", 0.01, 0.1, 0.05)
        radius_se = st.slider("Enter your Radius SE", 0.1, 3.0, 1.0)
        texture_se = st.slider("Enter your Texture SE", 0.2, 2.0, 1.0)

    with col2:
        perimeter_se = st.slider("Enter your Perimeter SE", 1.0, 30.0, 10.0)
        area_se = st.slider("Enter your Area SE", 6.0, 500.0, 150.0)
        smoothness_se = st.slider("Enter your Smoothness SE", 0.001, 0.03, 0.01)

    with col3:
        compactness_se = st.slider("Enter your Compactness SE", 0.002, 0.2, 0.1)
        concavity_se = st.slider("Enter your Concavity SE", 0.0, 0.05, 0.02)
        concave_points_se = st.slider("Enter your Concave Points SE", 0.0, 0.03, 0.01)

    with col1:
        symmetry_se = st.slider("Enter your Symmetry SE", 0.1, 1.0, 0.5)
        fractal_dimension_se = st.slider("Enter your Fractal Dimension SE", 0.01, 0.1, 0.05)

    with col2:
        radius_worst = st.slider("Enter your Radius Worst", 7.0, 40.0, 20.0)
        texture_worst = st.slider("Enter your Texture Worst", 12.0, 50.0, 25.0)
        perimeter_worst = st.slider("Enter your Perimeter Worst", 50.0, 250.0, 120.0)

    with col3:
        area_worst = st.slider("Enter your Area Worst", 185.0, 4250.0, 1500.0)
        smoothness_worst = st.slider("Enter your Smoothness Worst", 0.07, 0.3, 0.15)
        compactness_worst = st.slider("Enter your Compactness Worst", 0.03, 0.6, 0.3)

    with col1:
        concavity_worst = st.slider("Enter your Concavity Worst", 0.0, 0.8, 0.4)
        concave_points_worst = st.slider("Enter your Concave Points Worst", 0.0, 0.2, 0.1)
        symmetry_worst = st.slider("Enter your Symmetry Worst", 0.1, 1.0, 0.5)

    with col2:
        fractal_dimension_worst = st.slider("Enter your Fractal Dimension Worst", 0.01, 0.2, 0.1)

        # Code for prediction
    breast_cancer_result = ''

    # Button
    if st.button("Predict Breast Cancer"):
        # Create a DataFrame with user inputs
        user_input = pd.DataFrame({
            'radius_mean': [radius_mean],
            'texture_mean': [texture_mean],
            'perimeter_mean': [perimeter_mean],
            'area_mean': [area_mean],
            'smoothness_mean': [smoothness_mean],
            'compactness_mean': [compactness_mean],
            'concavity_mean': [concavity_mean],
            'concave points_mean': [concave_points_mean],  # Update this line
            'symmetry_mean': [symmetry_mean],
            'fractal_dimension_mean': [fractal_dimension_mean],
            'radius_se': [radius_se],
            'texture_se': [texture_se],
            'perimeter_se': [perimeter_se],
            'area_se': [area_se],
            'smoothness_se': [smoothness_se],
            'compactness_se': [compactness_se],
            'concavity_se': [concavity_se],
            'concave points_se': [concave_points_se],  # Update this line
            'symmetry_se': [symmetry_se],
            'fractal_dimension_se': [fractal_dimension_se],
            'radius_worst': [radius_worst],
            'texture_worst': [texture_worst],
            'perimeter_worst': [perimeter_worst],
            'area_worst': [area_worst],
            'smoothness_worst': [smoothness_worst],
            'compactness_worst': [compactness_worst],
            'concavity_worst': [concavity_worst],
            'concave points_worst': [concave_points_worst],  # Update this line
            'symmetry_worst': [symmetry_worst],
            'fractal_dimension_worst': [fractal_dimension_worst],
        })

        # Perform prediction
        breast_cancer_prediction = breast_cancer_model.predict(user_input)
        # Display result
        if breast_cancer_prediction[0] == 1:
            image = Image.open('positive.png')
            st.image(image, caption='')
            breast_cancer_result = "The model predicts that you have Breast Cancer."
        else:
            image = Image.open('negative.png')
            st.image(image, caption='')
            breast_cancer_result = "The model predicts that you don't have Breast Cancer."

        st.success(breast_cancer_result)


# ========== COVID-19 ==========
elif selected == 'Covid-19 Prediction':
    st.header("Covid-19 Risk Assessment")
    image = Image.open('cov.jpg')
    st.image(image)
    covid_features = pd.read_csv("data/Covid.csv").columns.tolist()[:-1]
    values = [1 if st.selectbox(f, ['NO', 'YES']) == 'YES' else 0 for f in covid_features]
    if st.button("Predict Covid-19"):
        prediction = covid_model.predict([values])
        if prediction[0] == 0:
            image = Image.open("negative.png")
            st.image(image, caption="")
            st.success("✅ Covid-19 Negative")
        else:
            image = Image.open("positive.png")
            st.image(image, caption="")
            st.warning("⚠️ Covid-19 Positive")


# ========== LUNG CANCER ==========
elif selected == 'Lung Cancer Prediction':
    st.header("Lung Cancer Risk Assessment")
    image = Image.open('lu.jpeg')
    st.image(image)
    lung_data = pd.read_csv('data/lung_cancer_survey.csv')
    lung_features = lung_data.columns[:-1]
    lung_inputs = {}
    cols = st.columns(3)
    for i, col in enumerate(lung_features):
        with cols[i % 3]:
            if col == 'AGE':
                lung_inputs[col] = st.text_input(col, value="")
            elif col == 'GENDER':
                lung_inputs[col] = st.selectbox(col, ['Male', 'Female'])
            else:
                lung_inputs[col] = st.selectbox(col, ['NO', 'YES'])
    if st.button("Predict Lung Cancer"):
        try:
            df = pd.DataFrame(lung_inputs, index=[0])
            df.replace({'NO': 1, 'YES': 2, 'Male': 1, 'Female': 2}, inplace=True)
            prediction = lung_model.predict(df)
            if prediction[0] == 0:
                image = Image.open("negative.png")
                st.image(image, caption="")
                st.success("✅ No Lung Cancer Risk")
            else:
                image = Image.open("positive.png")
                st.image(image, caption="")
                st.warning("⚠️ Lung Cancer Risk Detected")

        except:
            st.error("❌ Please check inputs.")

if selected == 'Pneumonia Prediction':
    st.header("Pneumonia Prediction from Xray Image")
     st.subheader("COMING SOON :)")
    

st.markdown("""
    <div style="
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #581818;
        padding: 5px 0;
        text-align: center;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    ">
        <div style='
            font-weight: 500; 
            font-size: 15px;
            color: white;
        '>
            Made with ❤️ by 
            <a href='https://mehul-raul.github.io/mehul.dev.portfolio/' 
               target='_blank' 
               style='
                   color: #ffd700;
                   text-decoration: none;
                   font-weight: bold;
                   font-style: italic;
               '>
                Mehul Raul
            </a>
        </div>
    </div>
""", unsafe_allow_html=True)
