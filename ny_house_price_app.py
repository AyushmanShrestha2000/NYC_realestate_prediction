import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# PAGE CONFIG & CUSTOM CSS
# ==============================================
st.set_page_config(
    page_title="NYC Home Price Predictor", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏠"
)

st.markdown("""
<style>
    body {
        color: #222222;
    }
    .main {
        background-color: #585ab8;
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: #ffffff;
    }
    .stButton > button {
        background: linear-gradient(90deg, #4a90e2, #007aff);
        color: #58b6b8;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s ease-in-out;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metric-card {
        background-color: #007aff;
        padding: 1rem;
        border-radius: 10px;
        color: #58b6b8;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMetric > div {
        background: transparent !important;
    }
    .stMetric > div > div {
        color: #222222 !important;
    }
    .main-header {
        text-align: center;
        color: #222222;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .stSelectbox, .stNumberInput, .stTextInput, .stSlider {
        background-color: #ffffff !important;
        border-radius: 8px;
        color: #58b6b8 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stAlert {
        border-radius: 8px;
        background-color: #f8d7da;
        color: #58b6b8;
    }
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #4a90e2, #007aff);
    }
    h1, h2, h3 {
        color: #222222;
    }
    h2 {
        border-bottom: 2px solid #007aff;
        padding-bottom: 0.3rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #e9ecef;
        color: #58b6b8;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        transition: all 0.3s ease-in-out;
    }
    .stTabs [aria-selected="true"] {
        background: #007aff;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================
# SIDEBAR & DATA LOADING
# ==============================================
with st.sidebar:
    st.markdown('<div style="text-align: center; margin-bottom: 2rem;">'
                '<h2 style="color: white; margin-bottom: 0;">🏠 NYC Home Price Predictor</h2>'
                '<p style="color: #ecf0f1;">AI-powered real estate valuation</p>'
                '</div>', unsafe_allow_html=True)
    
    page = st.radio(
        "Menu",
        ["📋 Dataset Overview", "📈 Market Analysis", "🤖 Price Prediction Models", "💲 Get Price Estimate"],
        index=0,
        label_visibility="collapsed"
    )

@st.cache_data
def load_data():
    try:
        data = pd.read_csv('NY-House-Dataset.csv')
        st.session_state['data_loaded'] = True
        return data
    except Exception as e:
        st.session_state['data_loaded'] = False
        st.error(f"Error loading data: {str(e)}")
        return None

data = load_data()

if not st.session_state.get('data_loaded', False):
    st.error("❗ Dataset not loaded. Please check the data file.")
    st.stop()

# ==============================================
# PAGES IMPLEMENTATION
# ==============================================
st.markdown('<h1 class="main-header">🏠 NYC Home Price Predictor</h1>', unsafe_allow_html=True)

if page == "📋 Dataset Overview":
    st.markdown("### 📊 Explore the NYC Housing Dataset")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Properties", f"{data.shape[0]:,}")
        st.metric("Number of Features", data.shape[1])
    with col2:
        st.metric("Average Price", f"${data['PRICE'].mean():,.0f}")
        st.metric("Price Range", f"${data['PRICE'].min():,.0f} - ${data['PRICE'].max():,.0f}")
    
    with st.expander("🔍 Data Preview", expanded=True):
        st.dataframe(data.head(10))
    
    tab1, tab2, tab3 = st.tabs(["📝 Structure", "📈 Statistics", "❓ Missing Data"])
    with tab1:
        st.dataframe(pd.DataFrame(data.dtypes, columns=['Data Type']))
    with tab2:
        st.dataframe(data.describe())
    with tab3:
        missing = data.isnull().sum()
        st.dataframe(pd.DataFrame({'Missing Values': missing, 'Percentage': (missing/len(data))*100}))

elif page == "📈 Market Analysis":
    st.markdown("### 📈 NYC Housing Market Trends")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        price_log = st.checkbox("Use logarithmic scale", value=True)
    with col2:
        bin_size = st.slider("Bin size", 10, 100, 50)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    if price_log:
        sns.histplot(np.log1p(data['PRICE']), bins=bin_size, kde=True, ax=ax, color='#4a6fa5')
        ax.set_xlabel('Log(Price)')
    else:
        sns.histplot(data['PRICE'], bins=bin_size, kde=True, ax=ax, color='#4a6fa5')
        ax.set_xlabel('Price ($)')
    ax.set_ylabel('Count')
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)
    
    analysis_type = st.radio("Analysis Type", ["Correlation", "Feature vs Price"], horizontal=True)
    
    if analysis_type == "Correlation":
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data.select_dtypes(include=np.number).corr(), annot=True, ax=ax, cmap='coolwarm')
        st.pyplot(fig)
    else:
        feature = st.selectbox("Select feature", data.select_dtypes(include=np.number).columns.drop('PRICE'))
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=data[feature], y=data['PRICE'], ax=ax, color='#4a6fa5')
        ax.set_xlabel(feature)
        ax.set_ylabel('Price ($)')
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

elif page == "🤖 Price Prediction Models":
    st.markdown("### 🤖 Train Prediction Models")
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Size (%)", 10, 40, 20)
    with col2:
        random_state = st.number_input("Random State", 0, 100, 42)
    
    features = ['BEDS', 'BATH', 'PROPERTYSQFT', 'LATITUDE', 'LONGITUDE']
    X = data[features]
    y = data['PRICE']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=random_state)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=random_state),
        'XGBoost': XGBRegressor(random_state=random_state)
    }
    
    selected_models = st.multiselect("Select Models", list(models.keys()), default=list(models.keys()))
    
    if st.button("Train Models"):
        results = []
        for name in selected_models:
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', models[name])
            ]).fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            results.append({
                'Model': name,
                'R2': r2_score(y_test, y_pred),
                'MAE': mean_absolute_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
            })
        
        st.dataframe(pd.DataFrame(results).set_index('Model').style.background_gradient(cmap='Blues'))

elif page == "💲 Get Price Estimate":
    st.markdown("### 💲 Predict Your Home's Value")
    
    col1, col2 = st.columns(2)
    with col1:
        beds = st.slider("Bedrooms", 1, 10, 2)
        bath = st.slider("Bathrooms", 1, 10, 2)
        sqft = st.number_input("Square Feet", 300, 10000, 1000)
    with col2:
        lat = st.number_input("Latitude", 40.5, 41.0, 40.7128)
        lon = st.number_input("Longitude", -74.5, -73.5, -74.0060)
    
    if st.button("Estimate Price"):
        # Simple prediction example
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor())
        ]).fit(data[['BEDS', 'BATH', 'PROPERTYSQFT', 'LATITUDE', 'LONGITUDE']], data['PRICE'])
        
        prediction = model.predict([[beds, bath, sqft, lat, lon]])[0]
        st.success(f"Estimated Value: ${prediction:,.0f}")

# ==============================================
# FOOTER
# ==============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #4a6fa5; font-size: 14px;'>
    🏙️ NYC Housing Market Dashboard | Built with Streamlit
</div>
""", unsafe_allow_html=True)
