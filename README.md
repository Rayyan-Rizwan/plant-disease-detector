
# 🌿 Plant Disease Recognition System

An AI-powered web application built using **Streamlit** and **TensorFlow / Keras** that detects plant leaf diseases from uploaded images with high accuracy.

---

## 🚀 Live Demo
🔗 **[Access the Live Web Application](https://plant-disease-detector-fhqvnffpnmpjdexzmu8fx2.streamlit.app/)**

---

## ✨ Features
- **Instant Disease Detection:** Upload a leaf image (`.jpg`, `.jpeg`, `.png`) to detect diseases across 38 categories.
- **Interactive UI:** Clean Streamlit dashboard for real-time inference.
- **Deep Learning Model:** Trained on over 87,000 leaf images using CNN architecture.

---

## 🛠️ Tech Stack
- **Frontend / Framework:** Streamlit
- **Machine Learning:** TensorFlow, Keras
- **Notebooks:** Jupyter Notebooks (`.ipynb`)
- **Image Processing & Math:** PIL (Pillow), NumPy
- **Runtime Environment:** Python 3.11

---

## 📁 Repository Structure

```text
plant_disease_detection/
│
├── test/                         # Test image datasets
├── train/                        # Training image datasets
├── valid/                        # Validation image datasets
├── .gitignore                    # Git ignore file
├── main.py                       # Main Streamlit web application script
├── Plant_Disease_Training.ipynb  # Jupyter Notebook for model training & building
├── test_plant_desease.ipynb     # Jupyter Notebook for testing and validation
├── requirements.txt             # Python project dependencies
├── runtime.txt                  # Python runtime version for deployment
├── trained_model.keras          # Trained TensorFlow/Keras deep learning model file
└── training_hist.json           # JSON file containing model training metrics & history

```

---

## 📊 Dataset & Model Details

* **Classes:** 38 distinct plant and disease categories.
* **Dataset split:** Training, Validation, and Testing sets.
* **Model Training:** Built and compiled via `Plant_Disease_Training.ipynb` and exported as `trained_model.keras`.

---

## 💻 Local Setup & Installation

1. **Clone the repository:**
```bash
git clone [https://github.com/Rayyan-Rizwan/plant-disease-detector.git](https://github.com/Rayyan-Rizwan/plant-disease-detector.git)
cd plant-disease-detector

```


2. **Create and activate a virtual environment:**
```bash
python -m venv env
# On Windows:
env\Scripts\activate
# On macOS/Linux:
source env/bin/activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```


4. **Run the application:**
```bash
streamlit run main.py

```



---

## 👨‍💻 Author

Developed by **[Rayyan Rizwan](https://www.google.com/search?q=https://github.com/Rayyan-Rizwan)**.

```

---

### Step 2: Run the Terminal Commands

Once `README.md` is saved, paste this entire command block into your VS Code terminal and press **Enter**:

```bash
git add main.py requirements.txt runtime.txt README.md .gitignore Plant_Disease_Training.ipynb test_plant_desease.ipynb training_hist.json
git commit -m "Add core project files, notebooks, and README"
git push origin main --force

```
