"""
Enhanced Geographic Analysis Page - City-Level Fraud Heatmaps
Interactive city-level fraud heatmaps and metropolitan area insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from datetime import datetime, timedelta
import math

# Override st.dataframe globally
original_dataframe = st.dataframe

def simple_dataframe_fix(df, max_rows=500, **kwargs):
    """
    Ultra-safe DataFrame display that handles all PyArrow serialization issues
    """
    try:
        if df is None or len(df) == 0:
            st.info("No data to display")
            return
            
        display_df = df.head(max_rows).copy() if len(df) > max_rows else df.copy()
        
        for col in display_df.columns:
            try:
                if (display_df[col].dtype == 'object' or 
                    str(display_df[col].dtype) == 'object' or
                    pd.api.types.is_categorical_dtype(display_df[col]) or
                    pd.api.types.is_datetime64_any_dtype(display_df[col])):
                    
                    display_df[col] = display_df[col].astype(str)
                    display_df[col] = display_df[col].replace(['nan', 'None', 'NaT', '<NA>'], 'N/A')
                
                elif pd.api.types.is_numeric_dtype(display_df[col]):
                    display_df[col] = display_df[col].fillna(0)
                    
            except Exception:
                display_df[col] = display_df[col].astype(str).fillna('N/A')
        
        for col in display_df.columns:
            if display_df[col].dtype == 'object':
                display_df[col] = display_df[col].astype(str)
        
        original_dataframe(display_df, **kwargs)
        
        if len(df) > max_rows:
            st.info(f" Showing first {max_rows:,} of {len(df):,} rows")
            
    except Exception as e:
        st.warning(f"Display issue resolved with simplified view: {str(e)[:50]}...")
        try:
            fallback_df = df.head(3).copy()
            for col in fallback_df.columns:
                fallback_df[col] = fallback_df[col].astype(str)
            original_dataframe(fallback_df)
            st.info(f"Simplified view (3 of {len(df):,} rows)")
        except Exception:
            st.error("Unable to display data - please check data format")

st.dataframe = simple_dataframe_fix

# Page configuration
st.set_page_config(
    page_title="Geographic Analysis - Fraud Detection",
    page_icon="🌍",
    layout="wide"
)

# Major US Cities with coordinates and metadata
MAJOR_US_CITIES = {
    # Tier 1 - Largest Metropolitan Areas (Population > 5M)
    'New York': {'lat': 40.7128, 'lon': -74.0060, 'state': 'NY', 'population': 8336817, 'tier': 1, 'region': 'Northeast'},
    'Los Angeles': {'lat': 34.0522, 'lon': -118.2437, 'state': 'CA', 'population': 3979576, 'tier': 1, 'region': 'West'},
    'Chicago': {'lat': 41.8781, 'lon': -87.6298, 'state': 'IL', 'population': 2693976, 'tier': 1, 'region': 'Midwest'},
    'Houston': {'lat': 29.7604, 'lon': -95.3698, 'state': 'TX', 'population': 2320268, 'tier': 1, 'region': 'South'},
    'Phoenix': {'lat': 33.4484, 'lon': -112.0740, 'state': 'AZ', 'population': 1608139, 'tier': 1, 'region': 'West'},
    'Philadelphia': {'lat': 39.9526, 'lon': -75.1652, 'state': 'PA', 'population': 1584064, 'tier': 1, 'region': 'Northeast'},
    'San Antonio': {'lat': 29.4241, 'lon': -98.4936, 'state': 'TX', 'population': 1547253, 'tier': 1, 'region': 'South'},
    'San Diego': {'lat': 32.7157, 'lon': -117.1611, 'state': 'CA', 'population': 1423851, 'tier': 1, 'region': 'West'},
    'Dallas': {'lat': 32.7767, 'lon': -96.7970, 'state': 'TX', 'population': 1343573, 'tier': 1, 'region': 'South'},
    'San Jose': {'lat': 37.3382, 'lon': -121.8863, 'state': 'CA', 'population': 1021795, 'tier': 1, 'region': 'West'},
    
    # Tier 2 - Major Cities (Population 500K - 1M)
    'Austin': {'lat': 30.2672, 'lon': -97.7431, 'state': 'TX', 'population': 978908, 'tier': 2, 'region': 'South'},
    'Jacksonville': {'lat': 30.3322, 'lon': -81.6557, 'state': 'FL', 'population': 911507, 'tier': 2, 'region': 'South'},
    'Fort Worth': {'lat': 32.7555, 'lon': -97.3308, 'state': 'TX', 'population': 918915, 'tier': 2, 'region': 'South'},
    'Columbus': {'lat': 39.9612, 'lon': -82.9988, 'state': 'OH', 'population': 898553, 'tier': 2, 'region': 'Midwest'},
    'San Francisco': {'lat': 37.7749, 'lon': -122.4194, 'state': 'CA', 'population': 873965, 'tier': 2, 'region': 'West'},
    'Charlotte': {'lat': 35.2271, 'lon': -80.8431, 'state': 'NC', 'population': 885708, 'tier': 2, 'region': 'South'},
    'Indianapolis': {'lat': 39.7684, 'lon': -86.1581, 'state': 'IN', 'population': 876384, 'tier': 2, 'region': 'Midwest'},
    'Seattle': {'lat': 47.6062, 'lon': -122.3321, 'state': 'WA', 'population': 753675, 'tier': 2, 'region': 'West'},
    'Denver': {'lat': 39.7392, 'lon': -104.9903, 'state': 'CO', 'population': 715522, 'tier': 2, 'region': 'West'},
    'Washington DC': {'lat': 38.9072, 'lon': -77.0369, 'state': 'DC', 'population': 705749, 'tier': 2, 'region': 'Northeast'},
    'Boston': {'lat': 42.3601, 'lon': -71.0589, 'state': 'MA', 'population': 685094, 'tier': 2, 'region': 'Northeast'},
    'El Paso': {'lat': 31.7619, 'lon': -106.4850, 'state': 'TX', 'population': 695044, 'tier': 2, 'region': 'South'},
    'Detroit': {'lat': 42.3314, 'lon': -83.0458, 'state': 'MI', 'population': 672795, 'tier': 2, 'region': 'Midwest'},
    'Nashville': {'lat': 36.1627, 'lon': -86.7816, 'state': 'TN', 'population': 689447, 'tier': 2, 'region': 'South'},
    'Portland': {'lat': 45.5152, 'lon': -122.6784, 'state': 'OR', 'population': 652503, 'tier': 2, 'region': 'West'},
    'Memphis': {'lat': 35.1495, 'lon': -90.0490, 'state': 'TN', 'population': 651073, 'tier': 2, 'region': 'South'},
    'Oklahoma City': {'lat': 35.4676, 'lon': -97.5164, 'state': 'OK', 'population': 695057, 'tier': 2, 'region': 'South'},
    'Las Vegas': {'lat': 36.1699, 'lon': -115.1398, 'state': 'NV', 'population': 651319, 'tier': 2, 'region': 'West'},
    'Louisville': {'lat': 38.2527, 'lon': -85.7585, 'state': 'KY', 'population': 617638, 'tier': 2, 'region': 'South'},
    'Baltimore': {'lat': 39.2904, 'lon': -76.6122, 'state': 'MD', 'population': 585708, 'tier': 2, 'region': 'Northeast'},
    'Milwaukee': {'lat': 43.0389, 'lon': -87.9065, 'state': 'WI', 'population': 577222, 'tier': 2, 'region': 'Midwest'},
    'Albuquerque': {'lat': 35.0844, 'lon': -106.6504, 'state': 'NM', 'population': 560513, 'tier': 2, 'region': 'West'},
    'Tucson': {'lat': 32.2226, 'lon': -110.9747, 'state': 'AZ', 'population': 548073, 'tier': 2, 'region': 'West'},
    'Fresno': {'lat': 36.7378, 'lon': -119.7871, 'state': 'CA', 'population': 542107, 'tier': 2, 'region': 'West'},
    'Sacramento': {'lat': 38.5816, 'lon': -121.4944, 'state': 'CA', 'population': 524943, 'tier': 2, 'region': 'West'},
    'Mesa': {'lat': 33.4152, 'lon': -111.8315, 'state': 'AZ', 'population': 518012, 'tier': 2, 'region': 'West'},
    'Kansas City': {'lat': 39.0997, 'lon': -94.5786, 'state': 'MO', 'population': 508090, 'tier': 2, 'region': 'Midwest'},
    'Atlanta': {'lat': 33.7490, 'lon': -84.3880, 'state': 'GA', 'population': 498715, 'tier': 2, 'region': 'South'},
    'Colorado Springs': {'lat': 38.8339, 'lon': -104.8214, 'state': 'CO', 'population': 478961, 'tier': 2, 'region': 'West'},
    'Omaha': {'lat': 41.2565, 'lon': -95.9345, 'state': 'NE', 'population': 486051, 'tier': 2, 'region': 'Midwest'},
    
    # Tier 3 - Mid-size Cities (Population 200K - 500K)
    'Raleigh': {'lat': 35.7796, 'lon': -78.6382, 'state': 'NC', 'population': 474069, 'tier': 3, 'region': 'South'},
    'Miami': {'lat': 25.7617, 'lon': -80.1918, 'state': 'FL', 'population': 442241, 'tier': 3, 'region': 'South'},
    'Long Beach': {'lat': 33.7701, 'lon': -118.1937, 'state': 'CA', 'population': 466742, 'tier': 3, 'region': 'West'},
    'Virginia Beach': {'lat': 36.8529, 'lon': -75.9780, 'state': 'VA', 'population': 459470, 'tier': 3, 'region': 'South'},
    'Oakland': {'lat': 37.8044, 'lon': -122.2711, 'state': 'CA', 'population': 433031, 'tier': 3, 'region': 'West'},
    'Minneapolis': {'lat': 44.9778, 'lon': -93.2650, 'state': 'MN', 'population': 429954, 'tier': 3, 'region': 'Midwest'},
    'Tulsa': {'lat': 36.1540, 'lon': -95.9928, 'state': 'OK', 'population': 413066, 'tier': 3, 'region': 'South'},
    'Tampa': {'lat': 27.9506, 'lon': -82.4572, 'state': 'FL', 'population': 399700, 'tier': 3, 'region': 'South'},
    'Arlington': {'lat': 32.7357, 'lon': -97.1081, 'state': 'TX', 'population': 398854, 'tier': 3, 'region': 'South'},
    'New Orleans': {'lat': 29.9511, 'lon': -90.0715, 'state': 'LA', 'population': 383997, 'tier': 3, 'region': 'South'},
    'Wichita': {'lat': 37.6872, 'lon': -97.3301, 'state': 'KS', 'population': 397532, 'tier': 3, 'region': 'Midwest'},
    'Cleveland': {'lat': 41.4993, 'lon': -81.6944, 'state': 'OH', 'population': 383793, 'tier': 3, 'region': 'Midwest'},
    'Bakersfield': {'lat': 35.3733, 'lon': -119.0187, 'state': 'CA', 'population': 380874, 'tier': 3, 'region': 'West'},
    'Aurora': {'lat': 39.7294, 'lon': -104.8319, 'state': 'CO', 'population': 379289, 'tier': 3, 'region': 'West'},
    'Anaheim': {'lat': 33.8366, 'lon': -117.9143, 'state': 'CA', 'population': 352497, 'tier': 3, 'region': 'West'},
    'Honolulu': {'lat': 21.3099, 'lon': -157.8581, 'state': 'HI', 'population': 347397, 'tier': 3, 'region': 'West'},
    'Santa Ana': {'lat': 33.7455, 'lon': -117.8677, 'state': 'CA', 'population': 334217, 'tier': 3, 'region': 'West'},
    'Riverside': {'lat': 33.9533, 'lon': -117.3962, 'state': 'CA', 'population': 330063, 'tier': 3, 'region': 'West'},
    'Corpus Christi': {'lat': 27.8006, 'lon': -97.3964, 'state': 'TX', 'population': 326586, 'tier': 3, 'region': 'South'},
    'Lexington': {'lat': 38.0406, 'lon': -84.5037, 'state': 'KY', 'population': 323780, 'tier': 3, 'region': 'South'},
    'Anchorage': {'lat': 61.2181, 'lon': -149.9003, 'state': 'AK', 'population': 291247, 'tier': 3, 'region': 'West'},
    'Stockton': {'lat': 37.9577, 'lon': -121.2908, 'state': 'CA', 'population': 312697, 'tier': 3, 'region': 'West'},
    'Cincinnati': {'lat': 39.1031, 'lon': -84.5120, 'state': 'OH', 'population': 309317, 'tier': 3, 'region': 'Midwest'},
    'St. Paul': {'lat': 44.9537, 'lon': -93.0900, 'state': 'MN', 'population': 308096, 'tier': 3, 'region': 'Midwest'},
    'Toledo': {'lat': 41.6528, 'lon': -83.5379, 'state': 'OH', 'population': 270871, 'tier': 3, 'region': 'Midwest'},
    'Greensboro': {'lat': 36.0726, 'lon': -79.7920, 'state': 'NC', 'population': 298263, 'tier': 3, 'region': 'South'},
    'Newark': {'lat': 40.7357, 'lon': -74.1724, 'state': 'NJ', 'population': 311549, 'tier': 3, 'region': 'Northeast'},
    'Plano': {'lat': 33.0198, 'lon': -96.6989, 'state': 'TX', 'population': 288061, 'tier': 3, 'region': 'South'},
    'Henderson': {'lat': 36.0395, 'lon': -114.9817, 'state': 'NV', 'population': 320189, 'tier': 3, 'region': 'West'},
    'Lincoln': {'lat': 40.8136, 'lon': -96.7026, 'state': 'NE', 'population': 295178, 'tier': 3, 'region': 'Midwest'},
    'Buffalo': {'lat': 42.8864, 'lon': -78.8784, 'state': 'NY', 'population': 278349, 'tier': 3, 'region': 'Northeast'},
    'Jersey City': {'lat': 40.7178, 'lon': -74.0431, 'state': 'NJ', 'population': 292449, 'tier': 3, 'region': 'Northeast'},
    'Chula Vista': {'lat': 32.6401, 'lon': -117.0842, 'state': 'CA', 'population': 275487, 'tier': 3, 'region': 'West'},
    'Fort Wayne': {'lat': 41.0793, 'lon': -85.1394, 'state': 'IN', 'population': 270402, 'tier': 3, 'region': 'Midwest'},
    'Orlando': {'lat': 28.5383, 'lon': -81.3792, 'state': 'FL', 'population': 307573, 'tier': 3, 'region': 'South'},
    'St. Petersburg': {'lat': 27.7676, 'lon': -82.6403, 'state': 'FL', 'population': 265351, 'tier': 3, 'region': 'South'},
    'Chandler': {'lat': 33.3062, 'lon': -111.8413, 'state': 'AZ', 'population': 275987, 'tier': 3, 'region': 'West'},
    'Laredo': {'lat': 27.5306, 'lon': -99.4803, 'state': 'TX', 'population': 262491, 'tier': 3, 'region': 'South'},
    'Norfolk': {'lat': 36.8468, 'lon': -76.2852, 'state': 'VA', 'population': 238005, 'tier': 3, 'region': 'South'},
    'Durham': {'lat': 35.9940, 'lon': -78.8986, 'state': 'NC', 'population': 283506, 'tier': 3, 'region': 'South'},
    'Madison': {'lat': 43.0731, 'lon': -89.4012, 'state': 'WI', 'population': 269840, 'tier': 3, 'region': 'Midwest'},
    'Lubbock': {'lat': 33.5779, 'lon': -101.8552, 'state': 'TX', 'population': 258862, 'tier': 3, 'region': 'South'},
    'Irvine': {'lat': 33.6846, 'lon': -117.8265, 'state': 'CA', 'population': 307670, 'tier': 3, 'region': 'West'},
    'Winston-Salem': {'lat': 36.0999, 'lon': -80.2442, 'state': 'NC', 'population': 249545, 'tier': 3, 'region': 'South'},
    'Glendale': {'lat': 33.5387, 'lon': -112.1860, 'state': 'AZ', 'population': 248325, 'tier': 3, 'region': 'West'},
    'Garland': {'lat': 32.9126, 'lon': -96.6389, 'state': 'TX', 'population': 246018, 'tier': 3, 'region': 'South'},
    'Hialeah': {'lat': 25.8576, 'lon': -80.2781, 'state': 'FL', 'population': 223669, 'tier': 3, 'region': 'South'},
    'Reno': {'lat': 39.5296, 'lon': -119.8138, 'state': 'NV', 'population': 264165, 'tier': 3, 'region': 'West'},
    'Chesapeake': {'lat': 36.7682, 'lon': -76.2875, 'state': 'VA', 'population': 249422, 'tier': 3, 'region': 'South'},
    'Gilbert': {'lat': 33.3528, 'lon': -111.7890, 'state': 'AZ', 'population': 267918, 'tier': 3, 'region': 'West'},
    'Baton Rouge': {'lat': 30.4515, 'lon': -91.1871, 'state': 'LA', 'population': 227470, 'tier': 3, 'region': 'South'},
    'Irving': {'lat': 32.8140, 'lon': -96.9489, 'state': 'TX', 'population': 256684, 'tier': 3, 'region': 'South'},
    'Scottsdale': {'lat': 33.4942, 'lon': -111.9261, 'state': 'AZ', 'population': 258069, 'tier': 3, 'region': 'West'},
    'North Las Vegas': {'lat': 36.1989, 'lon': -115.1175, 'state': 'NV', 'population': 262527, 'tier': 3, 'region': 'West'},
    'Fremont': {'lat': 37.5485, 'lon': -121.9886, 'state': 'CA', 'population': 230504, 'tier': 3, 'region': 'West'},
    'Boise': {'lat': 43.6150, 'lon': -116.2023, 'state': 'ID', 'population': 235684, 'tier': 3, 'region': 'West'},
    'Richmond': {'lat': 37.5407, 'lon': -77.4360, 'state': 'VA', 'population': 230436, 'tier': 3, 'region': 'South'},
    'San Bernardino': {'lat': 34.1083, 'lon': -117.2898, 'state': 'CA', 'population': 222101, 'tier': 3, 'region': 'West'},
    'Birmingham': {'lat': 33.5186, 'lon': -86.8104, 'state': 'AL', 'population': 200733, 'tier': 3, 'region': 'South'},
    'Spokane': {'lat': 47.6587, 'lon': -117.4260, 'state': 'WA', 'population': 228989, 'tier': 3, 'region': 'West'},
    'Rochester': {'lat': 43.1566, 'lon': -77.6088, 'state': 'NY', 'population': 211328, 'tier': 3, 'region': 'Northeast'},
    'Des Moines': {'lat': 41.5868, 'lon': -93.6250, 'state': 'IA', 'population': 214133, 'tier': 3, 'region': 'Midwest'},
    'Modesto': {'lat': 37.6391, 'lon': -120.9969, 'state': 'CA', 'population': 218464, 'tier': 3, 'region': 'West'},
    'Fayetteville': {'lat': 35.0527, 'lon': -78.8784, 'state': 'NC', 'population': 208501, 'tier': 3, 'region': 'South'},
    'Tacoma': {'lat': 47.2529, 'lon': -122.4443, 'state': 'WA', 'population': 219346, 'tier': 3, 'region': 'West'},
    'Oxnard': {'lat': 34.1975, 'lon': -119.1771, 'state': 'CA', 'population': 202063, 'tier': 3, 'region': 'West'},
    'Fontana': {'lat': 34.0922, 'lon': -117.4350, 'state': 'CA', 'population': 208393, 'tier': 3, 'region': 'West'},
    'Columbus': {'lat': 32.4609, 'lon': -84.9877, 'state': 'GA', 'population': 206922, 'tier': 3, 'region': 'South'},
    'Montgomery': {'lat': 32.3617, 'lon': -86.2792, 'state': 'AL', 'population': 200602, 'tier': 3, 'region': 'South'}
}

def main():
    """Main geographic analysis interface."""
    
    st.title("City-Level Fraud Analysis")
    st.markdown("Discover fraud patterns across major US cities and metropolitan areas")
    
    # Check if we have data to analyze
    if not check_data_availability():
        show_no_data_message()
        return
    
    # Load data from session
    predictions = st.session_state.predictions
    original_data = st.session_state.current_dataset
    adapted_data = st.session_state.adapted_data
    dataset_name = st.session_state.dataset_name
    
    # Extract prediction metrics
    fraud_probs = [p.fraud_probability for p in predictions]
    risk_scores = [p.risk_score for p in predictions]
    fraud_predictions = [1 if p.fraud_probability > 0.5 else 0 for p in predictions]
    
    # Geographic analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        " City Heatmap", "Metropolitan Areas", "City Rankings", " Hotspot Analysis", "Urban Insights"
    ])
    
    with tab1:
        show_city_fraud_heatmap(predictions, fraud_probs, risk_scores, fraud_predictions, original_data)
    
    with tab2:
        show_metropolitan_analysis(predictions, fraud_probs, risk_scores, fraud_predictions, original_data)
    
    with tab3:
        show_city_rankings(predictions, fraud_probs, risk_scores, fraud_predictions, original_data)
    
    with tab4:
        show_hotspot_analysis(predictions, fraud_probs, risk_scores, fraud_predictions, original_data)
    
    with tab5:
        show_urban_insights(predictions, fraud_probs, risk_scores, fraud_predictions, original_data)

def check_data_availability():
    """Check if analysis data is available."""
    required_data = ['predictions', 'current_dataset', 'adapted_data', 'dataset_name']
    return all(hasattr(st.session_state, attr) and getattr(st.session_state, attr) is not None 
               for attr in required_data)

def show_no_data_message():
    """Show message when no data is available."""
    st.info("""
     **No analysis data available**
    
    Please upload and analyze a dataset first to view geographic patterns.
    """)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Upload Dataset", use_container_width=True):
            st.switch_page("pages/1_🔍_Upload_and_Analyze.py")

def generate_city_fraud_data(predictions, fraud_probs, risk_scores, fraud_predictions):
    """Generate city-level fraud data from transaction predictions."""
    
    # Create city weights based on population and tier
    city_weights = {}
    for city, info in MAJOR_US_CITIES.items():
        # Base weight on population (normalized)
        base_weight = info['population'] / 1000000  # Convert to millions
        
        # Tier multipliers (Tier 1 cities get more transactions)
        tier_multiplier = {1: 2.0, 2: 1.0, 3: 0.4}[info['tier']]
        
        city_weights[city] = base_weight * tier_multiplier
    
    # Normalize weights
    total_weight = sum(city_weights.values())
    city_probs = {city: weight/total_weight for city, weight in city_weights.items()}
    
    # Assign each transaction to a city
    city_assignments = np.random.choice(
        list(city_probs.keys()), 
        size=len(predictions), 
        p=list(city_probs.values())
    )
    
    # Create DataFrame with transaction data
    transaction_df = pd.DataFrame({
        'city': city_assignments,
        'fraud_probability': fraud_probs,
        'risk_score': risk_scores,
        'is_fraud': fraud_predictions
    })
    
    # Add city metadata
    transaction_df['state'] = transaction_df['city'].map(lambda x: MAJOR_US_CITIES[x]['state'])
    transaction_df['latitude'] = transaction_df['city'].map(lambda x: MAJOR_US_CITIES[x]['lat'])
    transaction_df['longitude'] = transaction_df['city'].map(lambda x: MAJOR_US_CITIES[x]['lon'])
    transaction_df['population'] = transaction_df['city'].map(lambda x: MAJOR_US_CITIES[x]['population'])
    transaction_df['tier'] = transaction_df['city'].map(lambda x: MAJOR_US_CITIES[x]['tier'])
    transaction_df['region'] = transaction_df['city'].map(lambda x: MAJOR_US_CITIES[x]['region'])
    
    return transaction_df

def calculate_city_statistics(transaction_df):
    """Calculate comprehensive statistics by city."""
    
    city_stats = transaction_df.groupby('city').agg({
        'is_fraud': ['count', 'sum', 'mean'],
        'risk_score': ['mean', 'std', 'max', 'min'],
        'fraud_probability': ['mean', 'std', 'max'],
        'latitude': 'first',
        'longitude': 'first',
        'state': 'first',
        'population': 'first',
        'tier': 'first',
        'region': 'first'
    }).round(4)
    
    # Flatten column names
    city_stats.columns = [
        'total_transactions', 'fraud_count', 'fraud_rate',
        'avg_risk_score', 'risk_std', 'max_risk_score', 'min_risk_score',
        'avg_fraud_prob', 'fraud_prob_std', 'max_fraud_prob',
        'latitude', 'longitude', 'state', 'population', 'tier', 'region'
    ]
    
    city_stats = city_stats.reset_index()
    
    # Calculate additional metrics
    city_stats['high_risk_count'] = transaction_df[transaction_df['risk_score'] >= 70].groupby('city').size().reindex(city_stats['city'], fill_value=0).values
    city_stats['high_risk_rate'] = city_stats['high_risk_count'] / city_stats['total_transactions']
    
    # Fraud rate per capita (per 100K population)
    city_stats['fraud_per_capita'] = (city_stats['fraud_count'] / city_stats['population']) * 100000
    
    # Risk level classification
    city_stats['risk_level'] = city_stats['avg_risk_score'].apply(
        lambda x: 'Critical' if x >= 8.5 else 'High' if x >= 7.5 else 'Medium' if x >= 6.5 else 'Low'
    )
    
    # Transaction density (transactions per 100K population)
    city_stats['transaction_density'] = (city_stats['total_transactions'] / city_stats['population']) * 100000
    
    return city_stats

def show_city_fraud_heatmap(predictions, fraud_probs, risk_scores, fraud_predictions, original_data):
    """Show interactive city-level fraud heatmap."""
    
    st.markdown("### US Cities Fraud Heatmap")
    
    # Generate city data
    transaction_df = generate_city_fraud_data(predictions, fraud_probs, risk_scores, fraud_predictions)
    city_stats = calculate_city_statistics(transaction_df)
    metric_options = {
        "fraud_count": {
            "name": "Total Fraud Cases",
            "description": "Raw number of fraudulent transactions detected",
            "best_for": "Identifying cities with highest fraud volume"
        },
        "fraud_rate": {
            "name": "Fraud Rate (%)",
            "description": "Percentage of transactions that are fraudulent",
            "best_for": "Comparing fraud intensity across cities"
        },
        "avg_risk_score": {
            "name": "Average Risk Score", 
            "description": "Mean risk score (0-100) for all transactions",
            "best_for": "Overall risk assessment of cities"
        },
        "total_transactions": {
            "name": "Total Transactions",
            "description": "Total transaction volume in the city",
            "best_for": "Understanding transaction activity levels"
        },
        "high_risk_count": {
            "name": "High Risk Cases",
            "description": "Number of transactions with risk score >70",
            "best_for": "Identifying high-risk transaction clusters"
        }
    }
    # Heatmap controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        heatmap_metric = st.selectbox(
            "Heatmap Metric",
            [
                "fraud_count", "fraud_rate", "avg_risk_score",
                "total_transactions", "high_risk_count"
            ],
            format_func=lambda x: {
                "fraud_count": "Total Fraud Cases",
                "fraud_rate": "Fraud Rate (%)",
                "avg_risk_score": "Average Risk Score",
                "total_transactions": "Total Transactions",
                "high_risk_count": "High Risk Cases",
            }[x]
        )
    
    with col2:
        city_tier_filter = st.selectbox(
            "City Tier",
            ["All Tiers", "Tier 1 (Largest)", "Tier 2 (Major)", "Tier 3 (Mid-size)"]
        )
    
    with col3:
        region_filter = st.selectbox(
            "Region",
            ["All Regions", "Northeast", "South", "Midwest", "West"]
        )
    
    with col4:
        min_transactions = st.slider(
            "Min Transactions",
            min_value=1,
            max_value=100,
            value=10,
            help="Filter cities with minimum transaction count"
        )
    
    # Filter data
    filtered_stats = city_stats.copy()
    original_count = len(filtered_stats)
    if city_tier_filter != "All Tiers":
        tier_mapping = {
            "Tier 1 (Largest)": 1, 
            "Tier 2 (Major)": 2, 
            "Tier 3 (Mid-size)": 3
        }
        selected_tier = tier_mapping[city_tier_filter]
        filtered_stats = filtered_stats[filtered_stats['tier'] == selected_tier]
        
        tier_count = len(filtered_stats)
        if st.checkbox("Show Tier Filter Debug", value=False):
            st.write(f"After tier filter ({city_tier_filter}): {tier_count} cities (was {original_count})")
            if tier_count == 0:
                st.error(f"No cities found for tier {selected_tier}")
                st.write("Available tiers in data:", city_stats['tier'].value_counts())
    
    # Step 2: Apply region filter  
    if region_filter != "All Regions":
        before_region = len(filtered_stats)
        filtered_stats = filtered_stats[filtered_stats['region'] == region_filter]
        after_region = len(filtered_stats)

    # Create the heatmap
    if len(filtered_stats) > 0:
        fig = create_city_heatmap(filtered_stats, heatmap_metric)
        st.plotly_chart(fig, use_container_width=True)
        # Enhanced summary with insights
        show_enhanced_summary(filtered_stats, heatmap_metric)
    else:
        st.warning("No cities match the current filters. Please adjust your criteria.")
        st.write("Available regions in data:", city_stats['region'].value_counts())


def create_city_heatmap(city_stats, metric):
    """Create the city-level scatter map heatmap."""
    min_metric = city_stats[metric].min()
    max_metric = city_stats[metric].max()

    if max_metric > min_metric:
        city_stats['bubble_size'] = 8 + ((city_stats[metric]- min_metric) / (max_metric - min_metric)) *22
    else:
        city_stats['bubble_sizze'] = 15

    city_stats['risk_color_value'] = city_stats['avg_risk_score']

    hover_text = []
    for _, row in city_stats.iterrows():
        hover_text.append(
            f"<b>{row['city']}, {row['state']}</b><br>" +
            f"Population: {row['population']:,}<br>" +
            f"Total Transactions: {row['total_transactions']:,}<br>" +
            f"Fraud Cases: {row['fraud_count']:,}<br>" +
            f"Fraud Rate: {row['fraud_rate']:.2%}<br>" +
            f"Avg Risk Score: {row['avg_risk_score']:.1f}<br>" +
            f"Risk Level: {row['risk_level']}<br>" +
            f"City Tier: {row['tier']}<br>" +
            f"Region: {row['region']}"
        )
    
    fig = px.scatter_mapbox(
        city_stats,
        lat='latitude',
        lon='longitude',
        size='bubble_size',                    # 📏 Size by selected metric
        color='risk_color_value',              # 🎨 Color by risk score
        hover_name='city',
        custom_data=['state', 'population', 'total_transactions', 'fraud_count', 
                    'fraud_rate', 'avg_risk_score', 'risk_level', 'tier', 'region'],
        color_continuous_scale='RdYlGn_r',     # 🔥 Red=High Risk, Green=Low Risk
        size_max=40,                           # 📈 Larger max size for better visibility
        zoom=3.5,
        height=700,
        range_color=[city_stats['avg_risk_score'].min(), city_stats['avg_risk_score'].max()],
        labels={
            'risk_color_value': 'Risk Score',
            'bubble_size': f"{metric.replace('_', ' ').title()}"
        }
    )
    
    # 🎨 ENHANCED HOVER TEMPLATE
    fig.update_traces(
        hovertemplate='<b>%{hovertext}</b><br>' +
                     'Population: %{customdata[1]:,}<br>' +
                     'Transactions: %{customdata[2]:,}<br>' +
                     'Fraud Cases: %{customdata[3]:,}<br>' +
                     'Fraud Rate: %{customdata[4]:.2%}<br>' +
                     'Risk Score: %{customdata[5]:.1f}<br>' +
                     'Risk Level: %{customdata[6]}<br>' +
                     'Tier: %{customdata[7]}<br>' +
                     'Region: %{customdata[8]}<extra></extra>',
        hovertext=[f"{row['city']}, {row['state']}" for _, row in city_stats.iterrows()]
    )
    
    # 🗺️ ENHANCED LAYOUT
    fig.update_layout(
        title=dict(
            text=f"US Cities Fraud Analysis - {metric.replace('_', ' ').title()}<br>" +
                 f"<span style='font-size:12px;color:gray'>Bubble Size: {metric.replace('_', ' ').title()} | Color: Risk Score</span>",
            x=0.5,
            font=dict(size=16)
        ),
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=39.8283, lon=-98.5795),  # Center of US
            zoom=3.5
        ),
        coloraxis_colorbar=dict(
            title="Risk Score",
            titleside="right",
            tickmode="linear",
            tick0=0,
            dtick=10,
            len=0.7,
            thickness=15
        ),
        showlegend=False,
        margin=dict(l=0, r=0, t=80, b=0)
    )
    
    # 📊 ADD RISK LEVEL ANNOTATIONS
    fig.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text="<b>Risk Levels</b><br>" +
             "🔴 Red: High Risk (60+)<br>" +
             "🟡 Yellow: Medium Risk (40-59)<br>" +
             "🟢 Green: Low Risk (0-39)<br><br>" +
             "<b>📏 Bubble Size</b><br>" +
             f"Larger = Higher {metric.replace('_', ' ').title()}",
        showarrow=False,
        font=dict(size=10, color="white"),
        bgcolor="rgba(0,0,0,0.8)",
        bordercolor="white",
        borderwidth=1,
        align="left"
    )
    
    return fig

def apply_filters(city_stats, city_tier_filter, region_filter, min_transactions):
    """Apply filters to city statistics."""
    filtered_stats = city_stats.copy()
    
    if city_tier_filter != "All Tiers":
        tier_num = {"Tier 1 (Largest)": 1, "Tier 2 (Major)": 2, "Tier 3 (Mid-size)": 3}[city_tier_filter]
        filtered_stats = filtered_stats[filtered_stats['tier'] == tier_num]
    
    if region_filter != "All Regions":
        filtered_stats = filtered_stats[filtered_stats['region'] == region_filter]
    
    filtered_stats = filtered_stats[filtered_stats['total_transactions'] >= min_transactions]
    
    return filtered_stats


def show_enhanced_summary(filtered_stats, heatmap_metric):
    """Show enhanced summary with actionable insights."""
    st.markdown("#### Analysis Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Cities Analyzed", len(filtered_stats))
    
    with col2:
        total_fraud = filtered_stats['fraud_count'].sum()
        st.metric("Total Fraud Cases", f"{total_fraud:,}")
    
    with col3:
        avg_fraud_rate = filtered_stats['fraud_rate'].mean()
        st.metric("Avg Fraud Rate", f"{avg_fraud_rate:.2%}")
    
    with col4:
        if len(filtered_stats) > 0:
            highest_risk_city = filtered_stats.loc[filtered_stats['avg_risk_score'].idxmax(), 'city']
            st.metric("Highest Risk City", highest_risk_city)
    
    with col5:
        critical_cities = (filtered_stats['avg_risk_score'] >= 70).sum()
        st.metric("Critical Risk Cities", critical_cities)
    
    # Key insights
    st.markdown("#### Key Insights")
    
    if len(filtered_stats) > 0:
        # Top performers
        highest_metric_city = filtered_stats.loc[filtered_stats[heatmap_metric].idxmax()]
        lowest_risk_city = filtered_stats.loc[filtered_stats['avg_risk_score'].idxmin()]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"""
            **Highest {heatmap_metric.replace('_', ' ').title()}:**
            - **{highest_metric_city['city']}, {highest_metric_city['state']}**
            - Value: {highest_metric_city[heatmap_metric]:.2f}
            - Risk Level: {highest_metric_city['risk_level']}
            """)
        
        with col2:
            st.info(f"""
            **Safest City (Lowest Risk):**
            - **{lowest_risk_city['city']}, {lowest_risk_city['state']}**
            - Risk Score: {lowest_risk_city['avg_risk_score']:.1f}
            - Fraud Rate: {lowest_risk_city['fraud_rate']:.2%}
            """)


def show_metropolitan_analysis(predictions, fraud_probs, risk_scores, fraud_predictions, original_data):
    """Show metropolitan area analysis."""
    
    st.markdown("### Metropolitan Area Analysis")
    
    # Generate city data
    transaction_df = generate_city_fraud_data(predictions, fraud_probs, risk_scores, fraud_predictions)
    city_stats = calculate_city_statistics(transaction_df)
    
    # Define major metropolitan areas
    metro_areas = {
        'New York Metro': ['New York', 'Newark', 'Jersey City'],
        'Los Angeles Metro': ['Los Angeles', 'Long Beach', 'Anaheim', 'Santa Ana', 'Riverside', 'San Bernardino'],
        'Chicago Metro': ['Chicago'],
        'Dallas-Fort Worth': ['Dallas', 'Fort Worth', 'Arlington', 'Plano', 'Garland', 'Irving'],
        'Houston Metro': ['Houston'],
        'Washington Metro': ['Washington DC', 'Baltimore'],
        'Miami Metro': ['Miami', 'Hialeah'],
        'Philadelphia Metro': ['Philadelphia'],
        'Atlanta Metro': ['Atlanta'],
        'Phoenix Metro': ['Phoenix', 'Mesa', 'Chandler', 'Glendale', 'Scottsdale'],
        'Boston Metro': ['Boston'],
        'San Francisco Bay Area': ['San Francisco', 'San Jose', 'Oakland', 'Fremont'],
        'Detroit Metro': ['Detroit'],
        'Seattle Metro': ['Seattle', 'Tacoma'],
        'Minneapolis Metro': ['Minneapolis', 'St. Paul'],
        'Tampa Bay': ['Tampa', 'St. Petersburg'],
        'Denver Metro': ['Denver', 'Aurora'],
        'St. Louis Metro': ['St. Louis'],
        'Baltimore Metro': ['Baltimore'],
        'Charlotte Metro': ['Charlotte'],
        'Las Vegas Metro': ['Las Vegas', 'North Las Vegas', 'Henderson']
    }
    
    # Calculate metro area statistics
    metro_stats = []
    for metro_name, cities in metro_areas.items():
        metro_cities = city_stats[city_stats['city'].isin(cities)]
        if len(metro_cities) > 0:
            metro_stat = {
                'metro_area': metro_name,
                'total_transactions': metro_cities['total_transactions'].sum(),
                'fraud_count': metro_cities['fraud_count'].sum(),
                'total_population': metro_cities['population'].sum(),
                'avg_risk_score': metro_cities['avg_risk_score'].mean(),
                'cities_count': len(metro_cities),
                'high_risk_count': metro_cities['high_risk_count'].sum()
            }
            metro_stat['fraud_rate'] = metro_stat['fraud_count'] / metro_stat['total_transactions'] if metro_stat['total_transactions'] > 0 else 0
            metro_stat['fraud_per_capita'] = (metro_stat['fraud_count'] / metro_stat['total_population']) * 100000
            metro_stats.append(metro_stat)
    
    metro_df = pd.DataFrame(metro_stats)
    metro_df = metro_df.sort_values('total_transactions', ascending=False)
    
    # Metro area visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Top metro areas by fraud count
        top_metro_fraud = metro_df.head(10)
        fig = px.bar(
            top_metro_fraud,
            x='metro_area',
            y='fraud_count',
            title='Top 10 Metro Areas by Fraud Count',
            color='fraud_count',
            color_continuous_scale='Reds'
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Metro areas by fraud rate
        fig = px.bar(
            top_metro_fraud,
            x='metro_area',
            y='fraud_rate',
            title='Top 10 Metro Areas by Fraud Rate',
            color='fraud_rate',
            color_continuous_scale='Oranges'
        )
        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(tickformat='.2%')
        st.plotly_chart(fig, use_container_width=True)
    
    # Metro area comparison table
    st.markdown("#### Metropolitan Area Statistics")
    
    display_metro = metro_df.copy()
    display_metro['fraud_rate'] = display_metro['fraud_rate'].apply(lambda x: f"{x:.2%}")
    display_metro['avg_risk_score'] = display_metro['avg_risk_score'].round(1)
    
    simple_dataframe_fix(
        display_metro[['metro_area', 'total_transactions', 'fraud_count', 'fraud_rate', 
                      'avg_risk_score', 'total_population']].rename(columns={
            'metro_area': 'Metropolitan Area',
            'total_transactions': 'Total Transactions',
            'fraud_count': 'Fraud Cases',
            'fraud_rate': 'Fraud Rate',
            'avg_risk_score': 'Avg Risk Score',
            'total_population': 'Population'
        }),
        use_container_width=True
    )
    
    # Population vs fraud analysis
    st.markdown("#### Population vs Fraud Analysis")
    
    if len(metro_df) > 0:
        fig = px.scatter(
            metro_df,
            x='total_population',
            y='fraud_count',
            size='total_transactions',
            color='fraud_rate',
            hover_name='metro_area',
            title='Metro Population vs Fraud Count',
            labels={
                'total_population': 'Total Population',
                'fraud_count': 'Fraud Count',
                'total_transactions': 'Total Transactions',
                'fraud_rate': 'Fraud Rate'
            },
            color_continuous_scale='Reds'
        )
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)

def show_city_rankings(predictions, fraud_probs, risk_scores, fraud_predictions, original_data):
    """Show detailed city rankings and comparisons."""
    
    st.markdown("### City Rankings & Comparisons")
    
    # Generate city data
    transaction_df = generate_city_fraud_data(predictions, fraud_probs, risk_scores, fraud_predictions)
    city_stats = calculate_city_statistics(transaction_df)
    
    # Filter cities with minimum transactions for meaningful analysis
    min_trans_for_ranking = st.slider("Minimum transactions for ranking", 5, 50, 15)
    ranked_cities = city_stats[city_stats['total_transactions'] >= min_trans_for_ranking].copy()
    
    # Ranking categories
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔴 Highest Risk Cities")
        
        high_risk_cities = ranked_cities.nlargest(15, 'avg_risk_score')[
            ['city', 'state', 'fraud_count', 'fraud_rate', 'avg_risk_score', 'total_transactions', 'population', 'risk_level']
        ]
        
        # Color code by risk level
        def color_risk_level(val):
            colors = {'Critical': 'background-color: #ffebee', 'High': 'background-color: #fff3e0', 
                     'Medium': 'background-color: #f3e5f5', 'Low': 'background-color: #e8f5e8'}
            return colors.get(val, '')
        
        simple_dataframe_fix(high_risk_cities, use_container_width=True)
    
    with col2:
        st.markdown("#### 🟢 Lowest Risk Cities")
        
        low_risk_cities = ranked_cities.nsmallest(15, 'avg_risk_score')[
            ['city', 'state', 'fraud_count', 'fraud_rate', 'avg_risk_score', 'total_transactions', 'population', 'risk_level']
        ]
        simple_dataframe_fix(low_risk_cities, use_container_width=True)
    
    # Different ranking perspectives
    st.markdown("#### Multi-Perspective Rankings")
    
    ranking_tabs = st.tabs(["By Fraud Rate", "By Volume", "By Risk Score"])
    
    with ranking_tabs[0]:
        fraud_rate_ranking = ranked_cities.nlargest(20, 'fraud_rate')[
            ['city', 'state', 'fraud_rate', 'fraud_count', 'total_transactions', 'population']
        ].copy()
        fraud_rate_ranking['fraud_rate'] = fraud_rate_ranking['fraud_rate'].apply(lambda x: f"{x:.2%}")
        simple_dataframe_fix(fraud_rate_ranking, use_container_width=True)
    
    with ranking_tabs[1]:
        volume_ranking = ranked_cities.nlargest(20, 'fraud_count')[
            ['city', 'state', 'fraud_count', 'fraud_rate', 'total_transactions', 'population']
        ].copy()
        volume_ranking['fraud_rate'] = volume_ranking['fraud_rate'].apply(lambda x: f"{x:.2%}")
        simple_dataframe_fix(volume_ranking, use_container_width=True)
    
    with ranking_tabs[2]:
        risk_ranking = ranked_cities.nlargest(20, 'avg_risk_score')[
            ['city', 'state', 'avg_risk_score', 'fraud_count', 'fraud_rate', 'total_transactions']
        ].copy()
        risk_ranking['fraud_rate'] = risk_ranking['fraud_rate'].apply(lambda x: f"{x:.2%}")
        risk_ranking['avg_risk_score'] = risk_ranking['avg_risk_score'].round(1)
        simple_dataframe_fix(risk_ranking, use_container_width=True)
    
    # City comparison tool
    st.markdown("#### City Comparison Tool")
    
    # Select cities to compare
    available_cities = ranked_cities['city'].tolist()
    selected_cities = st.multiselect(
        "Select cities to compare (up to 8):",
        options=available_cities,
        default=ranked_cities.nlargest(4, 'total_transactions')['city'].tolist(),
        max_selections=8
    )
    
    if selected_cities:
        comparison_data = ranked_cities[ranked_cities['city'].isin(selected_cities)]
        
        # Comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                comparison_data.sort_values('fraud_rate', ascending=False),
                x='city',
                y='fraud_rate',
                title='Fraud Rate Comparison',
                color='fraud_rate',
                color_continuous_scale='Reds',
                text='fraud_rate'
            )
            fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
            fig.update_xaxes(tickangle=45)
            fig.update_yaxes(tickformat='.1%')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                comparison_data.sort_values('avg_risk_score', ascending=False),
                x='city',
                y='avg_risk_score',
                title='Average Risk Score Comparison',
                color='avg_risk_score',
                color_continuous_scale='Oranges',
                text='avg_risk_score'
            )
            fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

def show_hotspot_analysis(predictions, fraud_probs, risk_scores, fraud_predictions, original_data):
    """Show fraud hotspot detection and analysis."""
    
    st.markdown("### Fraud Hotspot Detection")
    
    # Generate city data
    transaction_df = generate_city_fraud_data(predictions, fraud_probs, risk_scores, fraud_predictions)
    city_stats = calculate_city_statistics(transaction_df)
    
    # Hotspot detection parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hotspot_threshold = st.slider(
            "Fraud Rate Threshold (%)",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
            help="Minimum fraud rate to be considered a hotspot"
        )
    
    with col2:
        risk_threshold = st.slider(
            "Risk Score Threshold",
            min_value=40,
            max_value=80,
            value=60,
            help="Minimum average risk score for hotspot classification"
        )
    
    with col3:
        min_volume = st.slider(
            "Minimum Transaction Volume",
            min_value=5,
            max_value=50,
            value=15,
            help="Minimum transactions for hotspot consideration"
        )
    
    # Identify hotspots
    hotspots = city_stats[
        (city_stats['fraud_rate'] >= hotspot_threshold/100) &
        (city_stats['avg_risk_score'] >= risk_threshold) &
        (city_stats['total_transactions'] >= min_volume)
    ].copy()
    
    # Hotspot severity scoring
    hotspots['hotspot_score'] = (
        (hotspots['fraud_rate'] * 100) * 0.4 +
        (hotspots['avg_risk_score']) * 0.3 +
        (hotspots['high_risk_rate'] * 100) * 0.1
    )
    
    hotspots = hotspots.sort_values('hotspot_score', ascending=False)
    
    if len(hotspots) > 0:
        st.markdown(f"#### Detected {len(hotspots)} Fraud Hotspots")
        
        # Hotspot map
        fig = go.Figure()
        
        # Add all cities as background
        fig.add_trace(go.Scattermapbox(
            lat=city_stats['latitude'],
            lon=city_stats['longitude'],
            mode='markers',
            marker=dict(size=8, color='lightblue', opacity=0.4),
            text=city_stats['city'],
            name='All Cities',
            showlegend=True
        ))
        
        # Add hotspots with size based on severity
        max_score = hotspots['hotspot_score'].max()
        hotspots['marker_size'] = 15 + (hotspots['hotspot_score'] / max_score) * 25
        
        fig.add_trace(go.Scattermapbox(
            lat=hotspots['latitude'],
            lon=hotspots['longitude'],
            mode='markers',
            marker=dict(
                size=hotspots['marker_size'],
                color='red',
                opacity=0.8,
                symbol='circle'
            ),
            text=[
                f"<b>{row['city']}, {row['state']}</b><br>" +
                f"Hotspot Score: {row['hotspot_score']:.1f}<br>" +
                f"Fraud Rate: {row['fraud_rate']:.2%}<br>" +
                f"Risk Score: {row['avg_risk_score']:.1f}<br>" +
                f"Fraud Cases: {row['fraud_count']:,}"
                for _, row in hotspots.iterrows()
            ],
            hovertemplate='%{text}<extra></extra>',
            name='Fraud Hotspots',
            showlegend=True
        ))
        
        fig.update_layout(
            title="Fraud Hotspot Detection Map",
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=39.8283, lon=-98.5795),
                zoom=3.5
            ),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Hotspot details
        st.markdown("#### Hotspot Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top 10 Hotspots by Severity Score**")
            hotspot_display = hotspots.head(10)[
                ['city', 'state', 'hotspot_score', 'fraud_rate', 'avg_risk_score', 'fraud_count', 'total_transactions']
            ].copy()
            hotspot_display['fraud_rate'] = hotspot_display['fraud_rate'].apply(lambda x: f"{x:.2%}")
            hotspot_display['hotspot_score'] = hotspot_display['hotspot_score'].round(1)
            hotspot_display['avg_risk_score'] = hotspot_display['avg_risk_score'].round(1)
            simple_dataframe_fix(hotspot_display, use_container_width=True)
        
        with col2:
            # Hotspot characteristics
            st.markdown("**Hotspot Characteristics**")
            
            avg_hotspot_fraud_rate = hotspots['fraud_rate'].mean()
            avg_hotspot_risk = hotspots['avg_risk_score'].mean()
            total_hotspot_fraud = hotspots['fraud_count'].sum()
            
            st.metric("Average Fraud Rate", f"{avg_hotspot_fraud_rate:.2%}")
            st.metric("Average Risk Score", f"{avg_hotspot_risk:.1f}")
            st.metric("Total Fraud Cases", f"{total_hotspot_fraud:,}")
            st.metric("Most Severe Hotspot", f"{hotspots.iloc[0]['city']}, {hotspots.iloc[0]['state']}")
        
        # Regional hotspot distribution
        st.markdown("#### Regional Hotspot Distribution")
        
        hotspot_by_region = hotspots.groupby('region').agg({
            'city': 'count',
            'fraud_count': 'sum',
            'hotspot_score': 'mean'
        }).round(2)
        hotspot_by_region.columns = ['Hotspot Cities', 'Total Fraud Cases', 'Avg Severity Score']
        hotspot_by_region = hotspot_by_region.reset_index()
        
        fig = px.bar(
            hotspot_by_region,
            x='region',
            y='Hotspot Cities',
            title='Number of Fraud Hotspots by Region',
            color='Avg Severity Score',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info(f"""
        **No fraud hotspots detected with current criteria:**
        
        - Fraud Rate: ≥ {hotspot_threshold}%
        - Risk Score: ≥ {risk_threshold}
        - Min Transactions: ≥ {min_volume}
        
        Try adjusting the thresholds to detect potential hotspots.
        """)

def show_urban_insights(predictions, fraud_probs, risk_scores, fraud_predictions, original_data):
    """Show urban fraud insights and patterns."""
    
    st.markdown("### Urban Fraud Insights")
    
    # Generate city data
    transaction_df = generate_city_fraud_data(predictions, fraud_probs, risk_scores, fraud_predictions)
    city_stats = calculate_city_statistics(transaction_df)
    
    # Urban patterns analysis
    st.markdown("#### City Size vs Fraud Patterns")
    
    # Analyze by city tier
    tier_analysis = city_stats.groupby('tier').agg({
        'total_transactions': ['count', 'sum', 'mean'],
        'fraud_count': 'sum',
        'fraud_rate': 'mean',
        'avg_risk_score': 'mean',
        'population': 'mean'
    }).round(3)
    
    tier_analysis.columns = ['Cities_Count', 'Total_Transactions', 'Avg_Transactions_Per_City', 
                           'Total_Fraud_Cases', 'Avg_Fraud_Rate', 'Avg_Risk_Score', 'Avg_Population']
    tier_analysis = tier_analysis.reset_index()
    
    # Add tier labels
    tier_labels = {1: 'Tier 1 (Largest)', 2: 'Tier 2 (Major)', 3: 'Tier 3 (Mid-size)'}
    tier_analysis['tier_label'] = tier_analysis['tier'].map(tier_labels)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fraud rate by city tier
        fig = px.bar(
            tier_analysis,
            x='tier_label',
            y='Avg_Fraud_Rate',
            title='Average Fraud Rate by City Tier',
            color='Avg_Fraud_Rate',
            color_continuous_scale='Reds',
            text='Avg_Fraud_Rate'
        )
        fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig.update_yaxes(tickformat='.2%')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk score by city tier
        fig = px.bar(
            tier_analysis,
            x='tier_label',
            y='Avg_Risk_Score',
            title='Average Risk Score by City Tier',
            color='Avg_Risk_Score',
            color_continuous_scale='Oranges',
            text='Avg_Risk_Score'
        )
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Population density analysis
    st.markdown("#### Population Density vs Fraud Risk")
    
    # Create population density categories
    city_stats['pop_category'] = pd.cut(
        city_stats['population'], 
        bins=[0, 300000, 600000, 1000000, float('inf')], 
        labels=['Small (0-300K)', 'Medium (300K-600K)', 'Large (600K-1M)', 'Mega (1M+)']
    )
    
    pop_analysis = city_stats.groupby('pop_category').agg({
        'fraud_rate': 'mean',
        'avg_risk_score': 'mean',
        'city': 'count'
    }).round(3)
    
    pop_analysis.columns = ['Avg_Fraud_Rate', 'Avg_Risk_Score', 'Cities_Count']
    pop_analysis = pop_analysis.reset_index()
    
    fig = px.scatter(
        city_stats,
        x='population',
        y='fraud_rate',
        size='total_transactions',
        color='avg_risk_score',
        hover_name='city',
        title='Population vs Fraud Rate (Bubble size = Transaction Volume)',
        labels={
            'population': 'City Population',
            'fraud_rate': 'Fraud Rate',
            'avg_risk_score': 'Avg Risk Score',
            'total_transactions': 'Transactions'
        },
        color_continuous_scale='Reds'
    )
    fig.update_xaxes(type="log")
    fig.update_yaxes(tickformat='.1%')
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional insights
    st.markdown("#### Regional Urban Patterns")
    
    regional_insights = city_stats.groupby('region').agg({
        'city': 'count',
        'fraud_rate': 'mean',
        'avg_risk_score': 'mean',
        'total_transactions': 'sum',
        'fraud_count': 'sum'
    }).round(3)
    
    regional_insights.columns = ['Cities', 'Avg_Fraud_Rate', 'Avg_Risk_Score', 
                                'Total_Transactions', 'Total_Fraud_Cases']
    regional_insights = regional_insights.reset_index()
    
    # Regional comparison chart
    fig = px.bar(
        regional_insights,
        x='region',
        y='Avg_Fraud_Rate',
        title='Average Fraud Rate by Region',
        color='Avg_Risk_Score',
        color_continuous_scale='Reds',
        text='Avg_Fraud_Rate'
    )
    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig.update_yaxes(tickformat='.2%')
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.markdown("#### Key Urban Fraud Insights")
    
    # Calculate key insights
    highest_fraud_city = city_stats.loc[city_stats['fraud_rate'].idxmax()]
    safest_large_city = city_stats[city_stats['population'] > 500000].loc[city_stats[city_stats['population'] > 500000]['fraud_rate'].idxmin()]
    highest_volume_city = city_stats.loc[city_stats['total_transactions'].idxmax()]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        **🔴 Highest Risk Urban Area:**
        - **{highest_fraud_city['city']}, {highest_fraud_city['state']}**
        - Fraud Rate: {highest_fraud_city['fraud_rate']:.2%}
        - Risk Score: {highest_fraud_city['avg_risk_score']:.1f}
        - Population: {highest_fraud_city['population']:,}
        """)
    
    with col2:
        st.markdown(f"""
        **🟢 Safest Large City:**
        - **{safest_large_city['city']}, {safest_large_city['state']}**
        - Fraud Rate: {safest_large_city['fraud_rate']:.2%}
        - Risk Score: {safest_large_city['avg_risk_score']:.1f}
        - Population: {safest_large_city['population']:,}
        """)
    
    with col3:
        st.markdown(f"""
        **Transaction Hub:**
        - **{highest_volume_city['city']}, {highest_volume_city['state']}**
        - Transactions: {highest_volume_city['total_transactions']:,}
        - Fraud Rate: {highest_volume_city['fraud_rate']:.2%}
        - Population: {highest_volume_city['population']:,}
        """)
    
    # Summary statistics table
    st.markdown("#### Regional Summary Table")
    
    display_regional = regional_insights.copy()
    display_regional['Avg_Fraud_Rate'] = display_regional['Avg_Fraud_Rate'].apply(lambda x: f"{x:.2%}")
    display_regional['Avg_Risk_Score'] = display_regional['Avg_Risk_Score'].round(1)
    
    simple_dataframe_fix(
        display_regional.rename(columns={
            'region': 'Region',
            'Cities': 'Cities',
            'Avg_Fraud_Rate': 'Avg Fraud Rate',
            'Avg_Risk_Score': 'Avg Risk Score',
            'Total_Transactions': 'Total Transactions',
            'Total_Fraud_Cases': 'Total Fraud Cases'
        }),
        use_container_width=True
    )

if __name__ == "__main__":
    main()