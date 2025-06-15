# Crop-recommendation-system
"ML-based crop recommendation system using environmental data."

# 🌾 Crop Recommendation System

This project provides a Machine Learning-based solution to recommend the most suitable crop to grow based on environmental conditions such as **soil nutrients**, **temperature**, **humidity**, **pH**, and **rainfall**.

---

## 📁 Project Structure
.
├── Crop\_recommendation.csv             # Dataset used for training
├── Crop\_Recommendation\Model.ipynb  # EDA + Preprocessing + Model Training
├── streamlit\_app.py                    # Web app code using Streamlit
├── crop\_recommendation\_model.pkl       # Trained ML model
├── scaler.pkl                          # Feature scaling object
├── label\_encoder.pkl                   # Label encoder for crop names
├── README.md                           # Project description--

## 📊 Dataset

- **Source**: [Kaggle - Crop Recommendation Dataset](https://www.kaggle.com/datasets/madhuraatmarambhagat/crop-recommendation-dataset)
- **Features**:
  - `N`, `P`, `K` – soil nutrients
  - `temperature`, `humidity`, `ph`, `rainfall`
  - `label` – crop name

---

## 🚀 How It Works

1. **EDA + Preprocessing** in the Jupyter Notebook.
2. **Model Training** with a classification algorithm.
3. **Trained model** saved as `.pkl` files.
4. **Interactive Web App** created using `Streamlit`.

---

## 📦 How to Run the App

bash
pip install -r requirements.txt
streamlit run streamlit_app.py

> Make sure all `.pkl` files and the dataset are in the same directory as `streamlit_app.py`.

---

## 🎯 Output

The app takes user inputs for temperature, humidity, soil nutrients, pH, and rainfall — and predicts the **best crop to grow**.

---

## 📜 License

This project is licensed under the MIT License.

```
