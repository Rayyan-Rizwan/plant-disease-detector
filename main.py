import os
import numpy as np
import streamlit as st
import tensorflow as tf

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Detector 🌿",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom CSS Styling (Sidebar + Upload UI + Cards + Metrics)
# ---------------------------------------------------------
st.markdown("""
    <style>
    /* ---------------------------------------------------------
       1. SIDEBAR NAVIGATION STYLING
    --------------------------------------------------------- */
    [data-testid="stSidebar"] {
        background-color: #161e17 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label {
        color: #e0e0e0 !important;
        font-family: 'Segoe UI', Roboto, sans-serif;
        font-weight: 600;
    }

    /* Transform Radio Options into Modern Navigation Cards */
    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        background-color: rgba(255, 255, 255, 0.04) !important;
        padding: 12px 16px !important;
        border-radius: 10px !important;
        margin-bottom: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        transition: all 0.25s ease-in-out !important;
        width: 100%;
        cursor: pointer;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background-color: rgba(76, 175, 80, 0.15) !important;
        border-color: #4caf50 !important;
        transform: translateX(4px);
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] input:checked + div {
        background-color: #4caf50 !important;
        border-color: #4caf50 !important;
    }

    /* ---------------------------------------------------------
       2. METRIC & STAT CARDS STYLING (ABOUT PAGE)
    --------------------------------------------------------- */
    .stat-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: transform 0.3s ease;
    }
    .stat-card:hover {
        transform: translateY(-4px);
        border-color: #4caf50;
    }
    .stat-title {
        font-size: 14px;
        color: #a0a0a0;
        font-weight: 500;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .stat-value {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
    }

    /* ---------------------------------------------------------
       3. FILE UPLOADER & PREDICTION UI STYLING
    --------------------------------------------------------- */
    [data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 2px dashed rgba(76, 175, 80, 0.4) !important;
        border-radius: 14px !important;
        padding: 15px !important;
        transition: all 0.3s ease-in-out !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #4caf50 !important;
        background-color: rgba(76, 175, 80, 0.08) !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: #2e7d32 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
    }

    /* Info / Warning Box Upgrade */
    .info-card {
        background-color: rgba(255, 255, 255, 0.03);
        border-left: 5px solid #4caf50;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-top: 5px;
    }

    /* Custom Card Style */
    .feature-card {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    
    /* Result Display Box */
    .result-box {
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
        color: white;
        padding: 22px;
        border-radius: 12px;
        text-align: center;
        font-size: 20px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.3);
    }
    
    /* Primary Action Buttons Styling */
    .stButton>button {
        width: 100%;
        background-color: #2e7d32;
        color: white;
        border-radius: 8px;
        height: 48px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        margin-top: 10px;
    }

    .stButton>button:hover {
        background-color: #1b5e20;
        color: white;
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Model Loading with Caching
# ---------------------------------------------------------
MODEL_PATH = "trained_model.keras"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        st.error(f"⚠️ Model file `{MODEL_PATH}` not found! Please check your directory.")
        return None

model = load_model()

def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/628/628324.png", width=75)
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to Page", ["Home", "About Project", "Disease Recognition"])

st.sidebar.markdown("---")
st.sidebar.caption("🌱 Powered by Machine Learning & TensorFlow")

# ---------------------------------------------------------
# Page 1: Home
# ---------------------------------------------------------
if app_mode == "Home":
    st.title("🌿 Plant Disease Recognition System")
    st.markdown("##### Detect plant diseases instantly using Artificial Intelligence.")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        image_path = "home_page.jpeg"
        if os.path.exists(image_path):
            st.image(image_path, use_container_width=True)
        else:
            st.image("https://images.unsplash.com/photo-1530836369250-ef72a3f5cda8?q=80&w=1000", use_container_width=True)
            
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>⚡ Quick Guide</h3>
            <ol>
                <li><b>Upload Image:</b> Navigate to <b>Disease Recognition</b> from the sidebar.</li>
                <li><b>Run Analysis:</b> Upload a leaf photo and click <i>Predict Disease</i>.</li>
                <li><b>Instant Results:</b> Receive precise disease classification instantly!</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🌟 Key Features</h3>
            <ul>
                <li><b>High Accuracy:</b> Trained on an extensive 87K+ image dataset.</li>
                <li><b>38 Classes:</b> Identifies health and disease states across major crops.</li>
                <li><b>Fast & Efficient:</b> Optimized deep learning model for quick results.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Page 2: About Project
# ---------------------------------------------------------
elif app_mode == "About Project":
    st.title("ℹ️ About the Project")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-title">Total Images</div>
            <div class="stat-value">87,000+</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-title">Total Classes</div>
            <div class="stat-value">38 Categories</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-title">Train Split</div>
            <div class="stat-value">80 / 20 Ratio</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h4>📊 Dataset Overview</h4>
        <p>This dataset was created using offline augmentation from original plant disease repositories. It contains high-resolution RGB images of healthy and diseased leaves.</p>
        <hr style="border-color: rgba(255,255,255,0.1); margin: 15px 0;">
        <ul>
            <li><b>Training Set:</b> 70,295 images</li>
            <li><b>Validation Set:</b> 17,572 images</li>
            <li><b>Test Set:</b> 33 images</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Page 3: Disease Recognition
# ---------------------------------------------------------
elif app_mode == "Disease Recognition":
    st.title("🔍 Plant Disease Recognition")
    st.write("Upload a leaf image below to run the classification model.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📤 Step 1: Upload Image")
        test_image = st.file_uploader("Choose an image (JPG, PNG)", type=["jpg", "jpeg", "png"])
        
        if test_image:
            st.image(test_image, caption="Uploaded Image Preview", use_container_width=True)

    with col2:
        st.markdown("### 🤖 Step 2: Prediction")
        
        if test_image is not None:
            if st.button("🔍 Predict Disease"):
                if model is not None:
                    st.snow()
                    with st.spinner("Analyzing Leaf Image..."):
                        result_index = model_prediction(test_image)

                    class_name = [
                        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy'
                    ]

                    prediction_text = class_name[result_index].replace("___", " - ").replace("_", " ")
                    
                    st.markdown(f"""
                    <div class="result-box">
                        Prediction Result:<br>
                        <span style="font-size: 24px;">🌿 {prediction_text}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("Model Error: Failed to load trained weights.")
        else:
            st.markdown("""
            <div class="info-card">
                <h4 style="margin: 0; color: #4caf50;">📌 Ready for Prediction</h4>
                <p style="margin-top: 8px; color: #b0bec5;">Please upload a crop leaf image in <b>Step 1</b> to activate the model prediction.</p>
            </div>
            """, unsafe_allow_html=True)