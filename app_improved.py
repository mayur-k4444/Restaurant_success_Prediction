import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# PAGE SETTINGS
# -------------------------------
st.set_page_config(
    page_title="Restaurant Success Dashboard",
    page_icon="🍽",
    layout="wide"
)

# -------------------------------
# LOAD MODEL + DATA
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("restaurant_success_model4.pkl")
    model_columns = joblib.load("model_columns4.pkl")
    return model, model_columns

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_restaurant_data4.csv")

model, model_columns = load_model()
df = load_data()

# -------------------------------
# EXTRACT CUISINES & LOCATIONS
# ✅ FIX: Now correctly extracts all locations including previously dropped one
# -------------------------------
cuisine_columns = [
    col for col in model_columns
    if col not in [
        'Average Price', 'Average Delivery Time',
        'Price_per_min', 'Cuisine_count',
        'Is_fast_delivery', 'Is_budget_friendly', 'Is_premium', 'Is_diverse_menu'
    ]
    and not col.startswith('Location_')
]

location_columns = [
    col.replace("Location_", "")
    for col in model_columns
    if col.startswith("Location_")
]

# -------------------------------
# HEADER
# -------------------------------
st.title("🍽 Zomato Restaurant Success AI Dashboard")
st.markdown("Predict restaurant success probability and analyze city-level trends.")

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Model Type", "GradientBoostingClassifier")
col_m2.metric("Features Used", str(len(model_columns)))
col_m3.metric("Locations Covered", str(len(location_columns)))

st.divider()

# -------------------------------
# TABS
# -------------------------------
tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Analysis"])

# ============================================================
# 🔮 TAB 1: PREDICTION
# ============================================================
with tab1:
    st.header("🔮 Restaurant Success Prediction")
    st.info("ℹ️ Fill in your restaurant details below and click **Predict Success**.")

    col1, col2 = st.columns(2)

    with col1:
        price = st.number_input("Average Price (₹)", min_value=10.0, max_value=2000.0, value=200.0, step=10.0)
        delivery_time = st.number_input("Delivery Time (minutes)", min_value=5.0, max_value=90.0, value=30.0, step=1.0)

    with col2:
        # ✅ FIX: All locations now available (previously one was silently dropped)
        location = st.selectbox("Select Location", sorted(location_columns), key="location_select")
        cuisines = st.multiselect("Select Cuisines", sorted(cuisine_columns))

    if st.button("🚀 Predict Success", type="primary"):

        # Build input with all zeros
        input_data = pd.DataFrame(0, index=[0], columns=model_columns)

        # Fill numeric features
        input_data['Average Price'] = price
        input_data['Average Delivery Time'] = delivery_time

        # Fill cuisine flags
        for cuisine in cuisines:
            if cuisine in input_data.columns:
                input_data[cuisine] = 1

        # Fill location flag
        location_col = f"Location_{location}"
        if location_col in input_data.columns:
            input_data[location_col] = 1

        # ✅ FIX: Recalculate engineered features (must match training pipeline)
        input_data['Price_per_min'] = price / (delivery_time + 1)
        input_data['Cuisine_count'] = len([c for c in cuisines if c in cuisine_columns])
        input_data['Is_fast_delivery'] = int(delivery_time < 25)
        input_data['Is_budget_friendly'] = int(price < 150)
        input_data['Is_premium'] = int(price > 400)
        input_data['Is_diverse_menu'] = int(len(cuisines) >= 3)

        # Predict
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        success_prob = probabilities[1] * 100
        failure_prob = probabilities[0] * 100

        # -------------------------------------------------------
        st.subheader("📊 Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                label="✅ Success Probability",
                value=f"{success_prob:.1f}%",
                delta=f"{success_prob - 55:.1f}% vs average" if success_prob >= 55 else f"{success_prob - 55:.1f}% vs average"
            )
            st.progress(int(success_prob))

        with result_col2:
            st.metric(
                label="❌ Failure Probability",
                value=f"{failure_prob:.1f}%"
            )
            st.progress(int(failure_prob))

        # Risk Assessment
        st.subheader("🎯 Risk Assessment")

        if success_prob >= 75:
            st.success("🟢 **LOW RISK** — Strong Business Potential. Your restaurant profile looks excellent!")
        elif success_prob >= 55:
            st.warning("🟡 **MODERATE RISK** — Reasonable potential but some areas need improvement.")
        else:
            st.error("🔴 **HIGH RISK** — Significant strategy improvements required before launch.")

        # Recommendations
        st.subheader("📌 Actionable Recommendations")

        recs = []

        if delivery_time > 35:
            recs.append(("⚡", "Reduce delivery time below 30 minutes. Faster delivery strongly correlates with higher ratings."))

        if delivery_time <= 25:
            recs.append(("🚀", "Excellent delivery speed! Keep it consistent to maintain customer satisfaction."))

        if price > 400:
            recs.append(("💰", "Premium pricing detected. Ensure quality justifies cost or consider competitive pricing."))

        if price < 150:
            recs.append(("👍", "Budget-friendly pricing — a strong competitive advantage for volume sales."))

        if len(cuisines) == 0:
            recs.append(("🍽", "No cuisines selected. Add cuisines to help the model give a more accurate prediction."))
        elif len(cuisines) < 3:
            recs.append(("📋", "Offering 3+ cuisine types tends to attract a broader customer base."))

        if success_prob < 55:
            recs.append(("📈", "Consider improving marketing strategy, menu diversity, or operational efficiency."))

        if not recs:
            recs.append(("✅", "Your restaurant profile is well-optimized! Maintain these standards."))

        for icon, msg in recs:
            st.info(f"{icon} {msg}")

# ============================================================
# 📊 TAB 2: ANALYSIS DASHBOARD
# ============================================================
with tab2:
    st.header("📊 Location-Level Business Analysis")

    location_columns_df = [col for col in df.columns if col.startswith("Location_")]
    cities = sorted([col.replace("Location_", "") for col in location_columns_df])

    selected_city = st.selectbox("Select City to Analyze", cities)

    city_col = f"Location_{selected_city}"

    # ✅ FIX: Handle case where city column might not exist in df
    if city_col in df.columns:
        city_data = df[df[city_col] == 1]
    else:
        st.warning(f"No data found for {selected_city}")
        st.stop()

    if len(city_data) == 0:
        st.warning(f"No restaurants found for {selected_city}")
        st.stop()

    # -----------------------
    # Key Metrics
    # -----------------------
    st.subheader("📈 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Restaurants", f"{len(city_data):,}")
    col2.metric("Avg Price (₹)", f"{city_data['Average Price'].mean():.0f}")
    col3.metric("Avg Delivery Time (min)", f"{city_data['Average Delivery Time'].mean():.1f}")

    # ✅ FIX: Use Binary_Success column (was using string comparison which can fail)
    if 'Binary_Success' in city_data.columns:
        success_rate = city_data['Binary_Success'].mean() * 100
    else:
        success_rate = (city_data['Success'] == 'Successful').mean() * 100
    col4.metric("Success Rate (%)", f"{success_rate:.1f}")

    # -----------------------
    # Cuisine Analysis
    # -----------------------
    st.subheader("🍽 Top 10 Popular Cuisines in " + selected_city)

    # ✅ FIX: Only use cuisine columns that actually exist in the dataframe
    valid_cuisine_cols = [c for c in cuisine_columns if c in city_data.columns]
    cuisine_counts = city_data[valid_cuisine_cols].sum().sort_values(ascending=False).head(10)

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    cuisine_counts.plot(kind='bar', ax=ax1, color='#3498db')
    ax1.set_title(f'Top 10 Cuisines in {selected_city}')
    ax1.set_ylabel('Number of Restaurants')
    ax1.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig1)

    # -----------------------
    # Success Rate by Cuisine
    # -----------------------
    st.subheader("🏆 Top 10 Most Successful Cuisines in " + selected_city)

    cuisine_success = {}
    target_col = 'Binary_Success' if 'Binary_Success' in city_data.columns else 'Success'

    for cuisine in valid_cuisine_cols:
        cuisine_rows = city_data[city_data[cuisine] == 1]
        if len(cuisine_rows) >= 5:  # ✅ FIX: minimum 5 restaurants for reliable stats
            if target_col == 'Binary_Success':
                rate = cuisine_rows['Binary_Success'].mean()
            else:
                rate = (cuisine_rows['Success'] == 'Successful').mean()
            cuisine_success[cuisine] = rate

    if cuisine_success:
        cuisine_success_df = pd.Series(cuisine_success).sort_values(ascending=False).head(10)

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        cuisine_success_df.plot(kind='bar', ax=ax2, color='#2ecc71')
        ax2.set_title(f'Cuisine Success Rate in {selected_city}')
        ax2.set_ylabel('Success Rate (0-1)')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)
    else:
        st.info("Not enough data to compute cuisine success rates for this city.")

    # -----------------------
    # Price Distribution
    # -----------------------
    st.subheader("💰 Price Distribution")

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    city_data['Average Price'].hist(bins=30, ax=ax3, color='#9b59b6', alpha=0.7)
    ax3.axvline(city_data['Average Price'].mean(), color='red', linestyle='--',
                label=f'Mean: ₹{city_data["Average Price"].mean():.0f}')
    ax3.set_xlabel('Average Price (₹)')
    ax3.set_ylabel('Count')
    ax3.set_title(f'Price Distribution in {selected_city}')
    ax3.legend()
    plt.tight_layout()
    st.pyplot(fig3)
