import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ----------------------------
# LOAD MODEL & DATA
# ----------------------------

@st.cache_resource
def load_model():
    import joblib
    return joblib.load("models/units_sold_model.pkl")

@st.cache_data
def load_data():
    import pandas as pd
    return pd.read_csv("retail_dynamic_pricing_dataset.csv")


model = load_model()
df = load_data()

# Feature columns must match training
feature_cols = [
    "effective_price",
    "competitor_price",
    "cost_price",
    "stock_level",
    "promotion",
    "category",
    "season",
    "day_of_week",
    "price_diff_vs_competitor",
    "margin",
]


# ----------------------------
# PRICE OPTIMIZATION FUNCTION
# ----------------------------

def suggest_optimal_price(row, model, price_min_factor=0.8, price_max_factor=1.2, steps=40):
    current_price = row["effective_price"]
    competitor_price = row["competitor_price"]
    cost_price = row["cost_price"]

    prices = np.linspace(current_price * price_min_factor,
                         current_price * price_max_factor, steps)

    best_price = None
    best_units = None
    best_revenue = -np.inf

    predicted_units_list = []
    revenue_list = []

    for p in prices:
        temp = row.copy()

        temp["effective_price"] = p
        temp["price_diff_vs_competitor"] = p - competitor_price
        temp["margin"] = p - cost_price

        temp_df = pd.DataFrame([temp[feature_cols]])

        units = model.predict(temp_df)[0]
        units = max(units, 0)

        revenue = units * p

        predicted_units_list.append(units)
        revenue_list.append(revenue)

        if revenue > best_revenue:
            best_revenue = revenue
            best_price = p
            best_units = units

    return (
        best_price,
        best_units,
        best_revenue,
        prices,
        predicted_units_list,
        revenue_list
    )


# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("🛒 AI Dynamic Pricing Engine Dashboard - Created By AjayGvijay")
st.write("Powered by Machine Learning | Retail Optimization")

# Select product
product_ids = sorted(df["product_id"].unique())
product_choice = st.selectbox("Select Product ID", product_ids)

# Get latest data for that product
row = df[df["product_id"] == product_choice].iloc[-1]

st.subheader("📦 Product Details")
col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"{row['effective_price']:.2f}")
col2.metric("Competitor Price", f"{row['competitor_price']:.2f}")
col3.metric("Units Sold Last Day", f"{row['units_sold']}")

st.write("---")

# Run optimization
(
    best_price,
    best_units,
    best_revenue,
    prices,
    predicted_units,
    revenues
) = suggest_optimal_price(row, model)

current_revenue = row["effective_price"] * row["units_sold"]
revenue_gain = best_revenue - current_revenue

# Output Cards
st.subheader("💡 AI Pricing Recommendation")

colA, colB, colC = st.columns(3)
colA.metric("Suggested Price", f"{best_price:.2f}")
colB.metric("Predicted Units Sold", f"{best_units:.1f}")
colC.metric("Predicted Revenue", f"{best_revenue:.2f}")

st.success(f"💰 Expected Revenue Gain: **{revenue_gain:.2f}**")

st.write("---")

# ----------------------------
# PLOTS
# ----------------------------

st.subheader("📉 Demand Curve (Price vs Units Sold)")

fig1, ax1 = plt.subplots()
ax1.plot(prices, predicted_units, marker="o")
ax1.set_xlabel("Price")
ax1.set_ylabel("Predicted Units Sold")
ax1.set_title("Demand Curve")
st.pyplot(fig1)

st.subheader("💸 Revenue Curve (Price vs Revenue)")

fig2, ax2 = plt.subplots()
ax2.plot(prices, revenues, marker="o")
ax2.axvline(best_price, linestyle="--", label=f"Optimal Price {best_price:.2f}")
ax2.set_xlabel("Price")
ax2.set_ylabel("Revenue")
ax2.set_title("Revenue Curve")
ax2.legend()
st.pyplot(fig2)

st.write("📊 The peak of this curve is the **optimal price** predicted by the ML model.")
