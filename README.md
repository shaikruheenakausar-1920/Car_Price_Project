# 🚗 Car Price Predictor

A machine learning web app that predicts the resale price of a used car based on its details, built with Python, Scikit-learn, and Streamlit.

## 🔗 Demo
> Add a screenshot or GIF of the app here after running it locally.
> Example: `![App Screenshot](screenshot.png)`

## 📌 Problem Statement
The price of a used car depends on several factors — present price, mileage, fuel type, seller type, transmission, number of previous owners, and age of the car. This project builds a regression model to predict the selling price based on these factors.

## 🛠 Tech Stack
- Python
- Pandas (data handling)
- Scikit-learn (machine learning)
- Streamlit (web app)

## 📊 Dataset
`car_data.csv` — ~300 used car listings with the following columns:
`Car_Name, Year, Selling_Price, Present_Price, Driven_kms, Fuel_Type, Selling_type, Transmission, Owner`

## 🧠 Approach
1. **Feature Engineering**: Converted `Year` into `Car_Age` (2024 - Year), since age is more directly related to depreciation than the year itself.
2. **Encoding**: Categorical columns (`Fuel_Type`, `Selling_type`, `Transmission`) were converted to numbers using `LabelEncoder`.
3. **Model**: Trained a `RandomForestRegressor` on features: Present Price, Kilometers Driven, Fuel Type, Seller Type, Transmission, Owner Count, and Car Age.
4. **Evaluation**: Split data 80/20 (train/test) and evaluated using R² score.

## ✅ Result
**R² Score: 0.96** on the test set — meaning the model explains 96% of the variation in car selling prices.

## 📥 Sample Prediction
| Present Price | Year | Fuel | Seller | Transmission | Owner | → Predicted Price |
|---|---|---|---|---|---|---|
| ₹9.83L | 2018 | Diesel | Dealer | Manual | 0 | ~₹9.1L |

## 🚀 How to Run
```bash
pip install streamlit pandas scikit-learn
streamlit run app.py
```
The app opens in your browser. The model (`car_model.pkl`) is already trained, so it works immediately.

To retrain the model from scratch:
```bash
python train_model.py
```

## 📂 Project Structure
```
Car_Price_Project/
├── app.py              # Streamlit app
├── train_model.py       # Model training script
├── car_data.csv          # Dataset
├── car_model.pkl         # Trained model
├── le_fuel.pkl           # Encoder for Fuel Type
├── le_seller.pkl         # Encoder for Seller Type
└── le_trans.pkl          # Encoder for Transmission
```

## 💡 What I Learned
- Feature engineering (converting Year → Age) to improve model relevance
- Encoding categorical variables for ML models
- Training and evaluating a regression model using R² score
- Deploying a trained model into an interactive Streamlit app
