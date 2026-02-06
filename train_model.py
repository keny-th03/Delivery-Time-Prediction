import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#load datasets
orders = pd.read_csv("olist_orders_dataset.csv")
order_items = pd.read_csv("olist_order_items_dataset.csv")
customers = pd.read_csv("olist_customers_dataset.csv")
products = pd.read_csv("olist_products_dataset.csv")
order_reviews = pd.read_csv("olist_order_reviews_dataset.csv")
sellers = pd.read_csv("olist_sellers_dataset.csv")

#Data Cleaning
orders = orders.dropna(subset=[
    'order_purchase_timestamp',
    'order_delivered_customer_date'
])
orders = orders.drop_duplicates(subset='order_id')

#Datetime Conversion
time_cols = [
    'order_purchase_timestamp',
    'order_delivered_customer_date',
    'order_estimated_delivery_date'
]
for col in time_cols:
    orders[col] = pd.to_datetime(orders[col])
    
#Target Variable
orders['delivery_time_days'] = (
    orders['order_delivered_customer_date']-orders['order_purchase_timestamp']
).dt.days

#Remove invalid delivery times
orders = orders[orders['delivery_time_days'] >= 0]

#Merge Datasets
df = orders.merge(order_items, on='order_id', how='left') \
           .merge(customers, on='customer_id', how='left') \
           .merge(products, on='product_id', how='left') \
           .merge(sellers, on='seller_id', how='left') \
           .merge(order_reviews[['order_id', 'review_score']], on='order_id', how='left')
           
#Feature Engineering
df['purchase_weekday'] = df['order_purchase_timestamp'].dt.weekday
df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour

df['delivery_delay'] = (
    df['order_delivered_customer_date'] - df['order_estimated_delivery_date']
).dt.days

df['is_late'] = (df['delivery_delay'] > 0).astype(int)

features = ['price', 'freight_value', 'review_score', 'purchase_weekday']
df_model = df[features + ['delivery_time_days']].dropna()

X = df_model[features]
y = df_model['delivery_time_days']

#Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}
results = {
    name: {
        "MAE": mean_absolute_error(y_test, model.fit(X_train, y_train).predict(X_test)),
        "MSE": mean_squared_error(y_test, model.fit(X_train, y_train).predict(X_test)),
        "RMSE": np.sqrt(mean_squared_error(y_test, model.fit(X_train, y_train).predict(X_test))),
        "R2_SCORE": r2_score(y_test, model.fit(X_train, y_train).predict(X_test))
    }
    for name, model in models.items()
}
# Convert results dictionary to a DataFrame
results_df = pd.DataFrame(results).T  # transpose so models are rows
results_df = results_df.round(2)       # round metrics to 2 decimals
results_df.index.name = "Model"        # name the index

# Display the DataFrame
print(results_df)

#Save nest model (Random Forest)
best_model = models["Random Forest"]
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)
    
#Save dataframe
with open("df.pkl", "wb") as f:
    pickle.dump(df, f)
    
# Feature Analysis with visualization
# Delivery time distribution
plt.figure(figsize=(8,4))
sns.histplot(df['delivery_time_days'], bins=50, kde=True, color='blue')
plt.title("Delivery Time Distribution (Days)")
plt.savefig("static/plots/delivery_time_dist.png", bbox_inches="tight")
plt.close()

# Orders per Product Category
product_cat= df['product_category_name'].value_counts().head(10)
plt.figure(figsize=(8,4))
product_cat.plot(kind='bar', color='pink', edgecolor='black')
plt.title("Top 10 Product Categories", fontsize=15)
plt.savefig("static/plots/product_category.png", bbox_inches="tight")
plt.close()

# Review Score vs Delivery Time
plt.figure(figsize=(8,4))
sns.boxplot(x='review_score', y='delivery_time_days', data=df, color='orange')
plt.title("Review Score vs Delivery Time")
plt.savefig("static/plots/review_vs_delivery.png", bbox_inches="tight")
plt.close()

# Freight vs Delivery Time
plt.figure(figsize=(8,4))
sns.scatterplot(
    x='freight_value', 
    y='delivery_time_days',
    data=df, alpha=0.3, color='purple'
)
plt.title("Freight vs Delivery Time")
plt.savefig("static/plots/freight_vs_delivery.png", bbox_inches="tight")
plt.close()

# Late vs On-Time
plt.figure(figsize=(8,4))
df['is_late'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title("Late vs On-Time Deliveries")
plt.xlabel("Delivery Status (0=On-Time, 1=Late)")
plt.ylabel("Number of Orders")
plt.xticks(rotation=0)
plt.savefig("static/plots/late_vs_ontime.png", bbox_inches="tight")
plt.close()

# Correlation Analysis
corr_f = df[['price', 'freight_value', 'review_score', 'delivery_time_days']]
sns.heatmap(corr_f.corr(), annot=True, cmap='coolwarm')
plt.savefig("static/plots/correlation.png", bbox_inches="tight")
plt.close()

# Overall Model Comparison
results_df.plot(kind='bar', figsize=(8,4))
plt.title("Overall Model Comparison")
plt.savefig("static/plots/model_comparison.png", bbox_inches="tight")
plt.close()

# Actual vs Predicted of best model
rf_model = models["Random Forest"]
y_pred = rf_model.fit(X_train, y_train).predict(X_test)

actual_vs_pred = pd.DataFrame({
    "Actual_Delivery_Time": y_test.values,
    "Predicted_Delivery_Time": y_pred
})

print(actual_vs_pred.head())

# Visualization
plt.figure(figsize=(8,4))
plt.scatter(y_test, y_pred, alpha=0.4, color='skyblue', edgecolor='black')
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red', linestyle='--', linewidth=2
)
plt.title("Actual vs Predicted Delivery Time (Random Forest)")
plt.savefig("static/plots/actual_vs_predicted_RF.png", bbox_inches="tight")
plt.close()

print("Model, data & plots saved successfully")