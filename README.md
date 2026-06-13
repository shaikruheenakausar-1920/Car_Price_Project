🚗 Car Price Predictor

What it does

A web app where you enter details about a used car (present price, kilometers
driven, fuel type, seller type, transmission, owner count, and purchase year)
and it predicts the expected selling price (in Lakhs ₹).

How it works


Dataset: car_data.csv (~300 used car listings)
Model: Random Forest Regressor (Scikit-learn)
Accuracy: R² score = 0.96 on test data
Frontend: Streamlit


How to run

cd Car_Price_Project
streamlit run app.py

This opens automatically in your browser. The model is already trained
(car_model.pkl) so it works immediately — no need to retrain.

To retrain from scratch (optional, prints accuracy score):

python train_model.py
