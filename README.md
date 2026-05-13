📦 Delivery Time Prediction System
A machine learning–powered web application that predicts order delivery time (in days) using e-commerce order data.
Built with Python, Flask, and Scikit-learn, and deployed locally with a clean UI and data visualizations.

🚀 Project Overview
This project uses the Brazilian E-Commerce Public Dataset (Olist) to:
▹ Analyze delivery patterns
▹ Train multiple ML models
▹ Predict delivery time for new orders
▹ Visualize insights using charts
▹ Serve predictions through a Flask web app

🧠 Machine Learning Models Used
The following models were trained and evaluated:
▹ Linear Regression
▹ Decision Tree Regressor
▹ Random Forest Regressor (Best Model ✅)

Evaluation Metrics
▹ MAE (Mean Absolute Error)
▹ MSE (Mean Squared Error)
▹ RMSE
▹ R² Score
The Random Forest model performed best and was saved as model.pkl.

🧾 Features Used for Prediction
▹ price – Product price
▹ freight_value – Shipping cost
▹ review_score – Customer review (1–5)
▹ purchase_weekday – Day of purchase (0 = Monday … 6 = Sunday)

🖥️ Web Application (Flask)
Pages Included
▹ Home Page – Input order details
▹ Prediction Page – Shows predicted delivery time
▹ Visualizations Page – Displays model & data insights

📊 Visualizations Generated
All plots are saved inside static/plots/:
▹ Delivery Time Distribution
▹ Top Product Categories
▹ Review Score vs Delivery Time
▹ Freight Value vs Delivery Time
▹ Late vs On-Time Deliveries
▹ Feature Correlation Heatmap
▹ Model Comparison
▹ Actual vs Predicted (Random Forest)

📌 Dataset Source
▹ Brazilian E-Commerce Public Dataset by Olist
▹ Available on Kaggle (https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data)

✨ Author
KENI THAPA
Student | Machine Learning & Data Science Enthusiast
