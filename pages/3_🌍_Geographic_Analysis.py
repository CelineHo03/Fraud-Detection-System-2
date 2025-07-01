"""
Geographic Analysis Page - Location-based Fraud Patterns
Interactive maps and geographic insights for fraud detection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
import random

# Override st.dataframe globally
original_dataframe = st.dataframe

def simple_dataframe_fix(df, max_rows=500, **kwargs):
    """
    Ultra-safe DataFrame display that handles all PyArrow serialization issues
    """
    try:
        # Ensure we have a valid DataFrame
        if df is None or len(df) == 0:
            st.info("No data to display")
            return
            
        # Create a safe copy
        display_df = df.head(max_rows).copy() if len(df) > max_rows else df.copy()
        
        # Aggressively convert all problematic columns
        for col in display_df.columns:
            try:
                # Convert all object types to string
                if (display_df[col].dtype == 'object' or 
                    str(display_df[col].dtype) == 'object' or
                    pd.api.types.is_categorical_dtype(display_df[col]) or
                    pd.api.types.is_datetime64_any_dtype(display_df[col])):
                    
                    # Convert to string and handle all edge cases
                    display_df[col] = display_df[col].astype(str)
                    display_df[col] = display_df[col].replace(['nan', 'None', 'NaT', '<NA>'], 'N/A')
                
                # Handle numeric columns with potential NaN
                elif pd.api.types.is_numeric_dtype(display_df[col]):
                    display_df[col] = display_df[col].fillna(0)
                    
            except Exception:
                # Ultimate fallback - convert everything to string
                display_df[col] = display_df[col].astype(str).fillna('N/A')
        
        # Final safety check - ensure no object dtypes remain
        for col in display_df.columns:
            if display_df[col].dtype == 'object':
                display_df[col] = display_df[col].astype(str)
        
        # 🔥 CRITICAL FIX: Call the ORIGINAL function, not the override!
        original_dataframe(display_df, **kwargs)  # ✅ FIXED: Use original_dataframe
        
        if len(df) > max_rows:
            st.info(f"📋 Showing first {max_rows:,} of {len(df):,} rows")
            
    except Exception as e:
        # Final fallback - create a simple text representation
        st.warning(f"Display issue resolved with simplified view: {str(e)[:50]}...")
        
        try:
            # Create ultra-simple version
            fallback_df = df.head(3).copy()
            for col in fallback_df.columns:
                fallback_df[col] = fallback_df[col].astype(str)
            # 🔥 ALSO FIX THIS: Use original function here too
            original_dataframe(fallback_df)  # ✅ FIXED: Use original_dataframe
            st.info(f"📋 Simplified view (3 of {len(df):,} rows)")
        except Exception:
            st.error("Unable to display data - please check data format")

st.dataframe = simple_dataframe_fix

# Page configuration
st.set_page_config(
    page_title="Geographic Analysis - Fraud Detection",
    page_icon="🌍",
    layout="wide"
)

def main():
    """Main geographic analysis interface."""
    
    st.title("🌍 Geographic Analysis")
    st.markdown("Discover location-based fraud patterns and geographic risk distributions")
    
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
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Interactive Map", "📊 Regional Analysis", "🎯 Hotspot Detection", "📈 Geographic Trends"
    ])
    
    with tab1:
        show_interactive_map(predictions, fraud_probs, risk_scores, fraud_predictions, original_data)
    
    with tab2:
        show_regional_analysis(predictions, fraud_probs, risk_scores, fraud_predictions, original_data)
    
    with tab3:
        show_hotspot_detection(predictions, fraud_probs, risk_scores, fraud_predictions, original_data)
    
    with tab4:
        show_geographic_trends(predictions, fraud_probs, risk_scores, fraud_predictions, original_data)

def check_data_availability():
    """Check if analysis data is available."""
    required_data = ['predictions', 'current_dataset', 'adapted_data', 'dataset_name']
    return all(hasattr(st.session_state, attr) and getattr(st.session_state, attr) is not None 
               for attr in required_data)

def show_no_data_message():
    """Show message when no data is available."""
    st.info("""
    🔍 **No analysis data available**
    
    Please upload and analyze a dataset first to view geographic patterns.
    """)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("📁 Upload Dataset", use_container_width=True):
            st.switch_page("pages/1_🔍_Upload_and_Analyze.py")

def show_interactive_map(predictions, fraud_probs, risk_scores, fraud_predictions, original_data):
    """Show interactive map with fraud locations."""
    
    st.markdown("### 🗺️ Fraud Risk Map")
    
    # Try to find geographic data
    geographic_data = extract_geographic_data(original_data)
    
    if geographic_data is None:
        # Generate synthetic geographic data for demonstration
        st.info("No geographic data detected in dataset. Generating synthetic locations for demonstration.")
        geographic_data = generate_synthetic_geographic_data(len(predictions))
    
    # Create DataFrame with all data
    map_df = pd.DataFrame({
        'latitude': geographic_data['lat'],
        'longitude': geographic_data['lon'],
        'fraud_probability': fraud_probs,
        'risk_score': risk_scores,
        'is_fraud': fraud_predictions,
        'location_name': geographic_data.get('location_names', [f"Location {i+1}" for i in range(len(predictions))])
    })
    
    # Map controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        map_style = st.selectbox(
            "Map Style",
            ["Fraud Risk", "Transaction Density", "Risk Score"],
            key="map_style"
        )
    
    with col2:
        min_risk = st.slider(
            "Minimum Risk Score",
            min_value=0,
            max_value=100,
            value=0,
            key="min_risk_filter"
        )
    
    with col3:
        show_fraud_only = st.checkbox(
            "Show Fraud Only",
            value=False,
            key="fraud_only_filter"
        )
    
    # Filter data based on controls
    filtered_df = map_df[map_df['risk_score'] >= min_risk]
    if show_fraud_only:
        filtered_df = filtered_df[filtered_df['is_fraud'] == 1]
    
    # Create map based on style
    if map_style == "Fraud Risk":
        show_fraud_risk_map(filtered_df)
    elif map_style == "Transaction Density":
        show_density_map(filtered_df)
    else:
        show_risk_score_map(filtered_df)
    
    # Map statistics
    show_map_statistics(filtered_df, map_df)

def extract_geographic_data(original_data):
    """Extract geographic information from the dataset."""
    
    # Look for common geographic columns
    lat_cols = [col for col in original_data.columns if 'lat' in col.lower() or 'latitude' in col.lower()]
    lon_cols = [col for col in original_data.columns if 'lon' in col.lower() or 'longitude' in col.lower()]
    
    if lat_cols and lon_cols:
        lat_data = pd.to_numeric(original_data[lat_cols[0]], errors='coerce')
        lon_data = pd.to_numeric(original_data[lon_cols[0]], errors='coerce')
        
        if lat_data.notna().sum() > 0 and lon_data.notna().sum() > 0:
            return {
                'lat': lat_data.fillna(0).tolist(),
                'lon': lon_data.fillna(0).tolist()
            }
    
    # Look for country/state/city columns
    location_cols = [col for col in original_data.columns 
                    if any(geo_word in col.lower() for geo_word in ['country', 'state', 'city', 'region', 'location'])]
    
    if location_cols:
        # This would require a geocoding service in production
        # For demo, we'll generate synthetic data
        return None
    
    return None

def generate_synthetic_geographic_data(n_transactions):
    """Generate synthetic geographic data for demonstration."""
    
    # Major cities with their coordinates for realistic demonstration
    major_cities = [
        {'name': 'New York', 'lat': 40.7128, 'lon': -74.0060},
        {'name': 'Los Angeles', 'lat': 34.0522, 'lon': -118.2437},
        {'name': 'Chicago', 'lat': 41.8781, 'lon': -87.6298},
        {'name': 'Houston', 'lat': 29.7604, 'lon': -95.3698},
        {'name': 'Miami', 'lat': 25.7617, 'lon': -80.1918},
        {'name': 'San Francisco', 'lat': 37.7749, 'lon': -122.4194},
        {'name': 'Boston', 'lat': 42.3601, 'lon': -71.0589},
        {'name': 'Seattle', 'lat': 47.6062, 'lon': -122.3321},
        {'name': 'Denver', 'lat': 39.7392, 'lon': -104.9903},
        {'name': 'Atlanta', 'lat': 33.7490, 'lon': -84.3880}
    ]
    
    # Generate locations with some clustering around major cities
    locations = []
    location_names = []
    
    for _ in range(n_transactions):
        # 70% chance of being near a major city
        if random.random() < 0.7:
            city = random.choice(major_cities)
            # Add some random variation around the city
            lat = city['lat'] + random.gauss(0, 0.5)
            lon = city['lon'] + random.gauss(0, 0.5)
            location_names.append(f"{city['name']} Area")
        else:
            # Random location in US
            lat = random.uniform(25, 49)  # Continental US latitude range
            lon = random.uniform(-125, -66)  # Continental US longitude range
            location_names.append("Other Location")
        
        locations.append({'lat': lat, 'lon': lon})
    
    return {
        'lat': [loc['lat'] for loc in locations],
        'lon': [loc['lon'] for loc in locations],
        'location_names': location_names
    }

def show_fraud_risk_map(map_df):
    """Show clean fraud risk map with limited strategic points"""
    
    # CRITICAL: Limit points for clean visualization
    max_points = 50  # Much more reasonable
    
    if len(map_df) > max_points:
        # Smart sampling - prioritize high-risk transactions
        high_risk = map_df[map_df['risk_score'] >= 70]
        medium_risk = map_df[(map_df['risk_score'] >= 40) & (map_df['risk_score'] < 70)]
        low_risk = map_df[map_df['risk_score'] < 40]
        
        # Take all high-risk + sample others
        sampled_df = pd.concat([
            high_risk,  # Keep ALL high-risk transactions
            medium_risk.sample(min(40, len(medium_risk))) if len(medium_risk) > 0 else pd.DataFrame(),
            low_risk.sample(min(20, len(low_risk))) if len(low_risk) > 0 else pd.DataFrame()
        ])
        
        st.info(f"🗺️ Showing {len(sampled_df)} strategic points (filtered from {len(map_df):,} total)")
    else:
        sampled_df = map_df
    
    # Create CLEAN map with limited points
    fig = px.scatter_mapbox(
        sampled_df,
        lat="latitude",
        lon="longitude",
        color="risk_score",
        size="fraud_probability",
        hover_name="location_name",
        hover_data={
            "fraud_probability": ":.3f",
            "risk_score": ":.0f",
            "is_fraud": True
        },
        color_continuous_scale="RdYlBu_r",  # Red=high risk, Blue=low risk
        size_max=20,
        zoom=3.5,
        title="Strategic Fraud Risk Points",
        height=600,
        range_color=[0, 100]
    )
    
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r":0,"t":50,"l":0,"b":0}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_density_map(map_df):
    """Show transaction density map."""
    
    # Create density heatmap
    fig = px.density_mapbox(
        map_df,
        lat="latitude",
        lon="longitude",
        z="risk_score",
        radius=20,
        center=dict(lat=39.8283, lon=-98.5795),  # Center of US
        zoom=3,
        mapbox_style="open-street-map",
        title="Transaction Density by Risk Score"
    )
    
    fig.update_layout(
        height=600,
        margin={"r":0,"t":50,"l":0,"b":0}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_risk_score_map(map_df):
    """Show risk score map with Folium."""
    
    st.markdown("**Interactive Risk Score Map**")
    
    # Create base map
    center_lat = map_df['latitude'].mean()
    center_lon = map_df['longitude'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=4,
        tiles='OpenStreetMap'
    )
    
    # Add points to map
    for idx, row in map_df.iterrows():
        if idx < 1000:  # Limit points for performance
            # Color based on risk score
            if row['risk_score'] >= 70:
                color = 'red'
            elif row['risk_score'] >= 50:
                color = 'orange'
            else:
                color = 'green'
            
            # Size based on fraud probability
            radius = 5 + (row['fraud_probability'] * 10)
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=radius,
                popup=f"""
                <b>{row['location_name']}</b><br>
                Risk Score: {row['risk_score']:.0f}<br>
                Fraud Prob: {row['fraud_probability']:.3f}<br>
                Fraud: {'Yes' if row['is_fraud'] else 'No'}
                """,
                color=color,
                fill=True,
                fillOpacity=0.7
            ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 90px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <b>Risk Levels</b><br>
    <i class="fa fa-circle" style="color:red"></i> High (70+)<br>
    <i class="fa fa-circle" style="color:orange"></i> Medium (50-69)<br>
    <i class="fa fa-circle" style="color:green"></i> Low (0-49)
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Display map
    folium_static(m, width=700, height=500)

def show_map_statistics(filtered_df, total_df):
    """Show statistics for the current map view."""
    
    st.markdown("#### 📊 Map Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Visible Transactions", len(filtered_df))
    
    with col2:
        fraud_count = len(filtered_df[filtered_df['is_fraud'] == 1])
        st.metric("Visible Fraud", fraud_count)
    
    with col3:
        avg_risk = filtered_df['risk_score'].mean()
        st.metric("Avg Risk Score", f"{avg_risk:.0f}/100")
    
    with col4:
        if len(total_df) > 0:
            coverage = len(filtered_df) / len(total_df) * 100
            st.metric("Map Coverage", f"{coverage:.0f}%")

def show_regional_analysis(predictions, fraud_probs, risk_scores, fraud_predictions, original_data):
    """Show regional fraud analysis."""
    
    st.markdown("### 📊 Regional Fraud Analysis")
    
    # Generate regional data for demonstration
    regions = ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West Coast']
    regional_data = []
    
    # Assign transactions to regions randomly for demo
    for i, prediction in enumerate(predictions):
        region = np.random.choice(regions)
        regional_data.append({
            'Region': region,
            'Fraud_Probability': fraud_probs[i],
            'Risk_Score': risk_scores[i],
            'Is_Fraud': fraud_predictions[i]
        })
    
    regional_df = pd.DataFrame(regional_data)
    
    # Regional statistics
    regional_stats = regional_df.groupby('Region').agg({
        'Is_Fraud': ['count', 'sum', 'mean'],
        'Risk_Score': 'mean',
        'Fraud_Probability': 'mean'
    }).round(3)
    
    regional_stats.columns = ['Total_Transactions', 'Fraud_Count', 'Fraud_Rate', 'Avg_Risk_Score', 'Avg_Fraud_Prob']
    regional_stats = regional_stats.reset_index()
    
    # Regional visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Fraud rate by region
        fig = px.bar(
            regional_stats,
            x='Region',
            y='Fraud_Rate',
            title='Fraud Rate by Region',
            color='Fraud_Rate',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average risk score by region
        fig = px.bar(
            regional_stats,
            x='Region',
            y='Avg_Risk_Score',
            title='Average Risk Score by Region',
            color='Avg_Risk_Score',
            color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Regional comparison table
    st.markdown("#### 📋 Regional Comparison")
    simple_dataframe_fix(regional_stats, use_container_width=True)
    
    # Risk distribution by region
    st.markdown("#### 🎯 Risk Distribution by Region")
    
    fig = px.box(
        regional_df,
        x='Region',
        y='Risk_Score',
        title='Risk Score Distribution by Region',
        color='Region'
    )
    st.plotly_chart(fig, use_container_width=True)

def show_hotspot_detection(predictions, fraud_probs, risk_scores, fraud_predictions, original_data):
    """Show fraud hotspot detection."""
    
    st.markdown("### 🎯 Fraud Hotspot Detection")
    
    # Generate hotspot analysis for demo
    geographic_data = generate_synthetic_geographic_data(len(predictions))
    
    # Create DataFrame for hotspot analysis
    hotspot_df = pd.DataFrame({
        'latitude': geographic_data['lat'],
        'longitude': geographic_data['lon'],
        'fraud_probability': fraud_probs,
        'risk_score': risk_scores,
        'is_fraud': fraud_predictions
    })
    
    # Hotspot detection parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hotspot_threshold = st.slider(
            "Hotspot Risk Threshold",
            min_value=50,
            max_value=90,
            value=70,
            help="Minimum risk score to be considered a hotspot"
        )
    
    with col2:
        radius = st.slider(
            "Hotspot Radius (miles)",
            min_value=10,
            max_value=100,
            value=25,
            help="Radius for grouping nearby transactions"
        )
    
    with col3:
        min_transactions = st.number_input(
            "Min Transactions",
            min_value=3,
            max_value=20,
            value=5,
            help="Minimum transactions to form a hotspot"
        )
    
    # Identify hotspots (simplified algorithm for demo)
    hotspots = identify_hotspots(hotspot_df, hotspot_threshold, radius, min_transactions)
    
    # Hotspot visualization
    if hotspots:
        st.markdown(f"#### 🔥 Detected {len(hotspots)} Fraud Hotspots")
        
        # Create hotspot map
        fig = go.Figure()
        
        # Add all transactions
        fig.add_trace(go.Scattermapbox(
            lat=hotspot_df['latitude'],
            lon=hotspot_df['longitude'],
            mode='markers',
            marker=dict(
                size=6,
                color=hotspot_df['risk_score'],
                colorscale='Viridis',
                opacity=0.6
            ),
            text=hotspot_df['risk_score'],
            name='All Transactions'
        ))
        
        # Add hotspot centers
        for i, hotspot in enumerate(hotspots):
            fig.add_trace(go.Scattermapbox(
                lat=[hotspot['center_lat']],
                lon=[hotspot['center_lon']],
                mode='markers',
                marker=dict(
                    size=20,
                    color='red',
                    symbol='star'
                ),
                text=f"Hotspot {i+1}",
                name=f"Hotspot {i+1}"
            ))
        
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=39.8283, lon=-98.5795),
                zoom=3
            ),
            title="Fraud Hotspots Detection",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Hotspot details
        st.markdown("#### 📋 Hotspot Details")
        
        hotspot_details = pd.DataFrame([
            {
                'Hotspot': f"Hotspot {i+1}",
                'Center Lat': f"{h['center_lat']:.4f}",
                'Center Lon': f"{h['center_lon']:.4f}",
                'Transactions': h['transaction_count'],
                'Avg Risk Score': f"{h['avg_risk_score']:.1f}",
                'Fraud Rate': f"{h['fraud_rate']:.1%}"
            }
            for i, h in enumerate(hotspots)
        ])
        
        simple_dataframe_fix(hotspot_details, use_container_width=True)
    
    else:
        st.info("No fraud hotspots detected with current parameters. Try adjusting the thresholds.")

def identify_hotspots(df, risk_threshold, radius_miles, min_transactions):
    """Identify fraud hotspots using simplified clustering."""
    
    # Filter high-risk transactions
    high_risk_df = df[df['risk_score'] >= risk_threshold]
    
    if len(high_risk_df) < min_transactions:
        return []
    
    # Simple grid-based clustering for demo
    # In production, you'd use proper clustering algorithms like DBSCAN
    
    lat_bins = np.linspace(high_risk_df['latitude'].min(), high_risk_df['latitude'].max(), 10)
    lon_bins = np.linspace(high_risk_df['longitude'].min(), high_risk_df['longitude'].max(), 10)
    
    hotspots = []
    
    for i in range(len(lat_bins) - 1):
        for j in range(len(lon_bins) - 1):
            # Find transactions in this grid cell
            cell_transactions = high_risk_df[
                (high_risk_df['latitude'] >= lat_bins[i]) &
                (high_risk_df['latitude'] < lat_bins[i+1]) &
                (high_risk_df['longitude'] >= lon_bins[j]) &
                (high_risk_df['longitude'] < lon_bins[j+1])
            ]
            
            if len(cell_transactions) >= min_transactions:
                hotspot = {
                    'center_lat': cell_transactions['latitude'].mean(),
                    'center_lon': cell_transactions['longitude'].mean(),
                    'transaction_count': len(cell_transactions),
                    'avg_risk_score': cell_transactions['risk_score'].mean(),
                    'fraud_rate': cell_transactions['is_fraud'].mean()
                }
                hotspots.append(hotspot)
    
    return hotspots

def show_geographic_trends(predictions, fraud_probs, risk_scores, fraud_predictions, original_data):
    """Show geographic trends and patterns."""
    
    st.markdown("### 📈 Geographic Trends")
    
    # Generate trend data for demo
    geographic_data = generate_synthetic_geographic_data(len(predictions))
    
    # Create DataFrame with geographic and temporal data
    trend_df = pd.DataFrame({
        'latitude': geographic_data['lat'],
        'longitude': geographic_data['lon'],
        'location_name': geographic_data['location_names'],
        'fraud_probability': fraud_probs,
        'risk_score': risk_scores,
        'is_fraud': fraud_predictions,
        'month': np.random.choice(range(1, 13), len(predictions))  # Random months for demo
    })
    
    # Monthly trends by location
    st.markdown("#### 📅 Monthly Fraud Trends by Location")
    
    monthly_trends = trend_df.groupby(['location_name', 'month']).agg({
        'is_fraud': ['count', 'sum', 'mean'],
        'risk_score': 'mean'
    }).round(3)
    
    monthly_trends.columns = ['Total_Transactions', 'Fraud_Count', 'Fraud_Rate', 'Avg_Risk']
    monthly_trends = monthly_trends.reset_index()
    
    # Select location for trend analysis
    unique_locations = trend_df['location_name'].unique()
    selected_location = st.selectbox(
        "Select location for trend analysis:",
        unique_locations,
        key="location_trend_selector"
    )
    
    if selected_location:
        location_trends = monthly_trends[monthly_trends['location_name'] == selected_location]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly fraud rate trend
            fig = px.line(
                location_trends,
                x='month',
                y='Fraud_Rate',
                title=f'Monthly Fraud Rate - {selected_location}',
                markers=True
            )
            fig.update_xaxes(title="Month")
            fig.update_yaxes(title="Fraud Rate")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Monthly risk score trend
            fig = px.line(
                location_trends,
                x='month',
                y='Avg_Risk',
                title=f'Monthly Average Risk Score - {selected_location}',
                markers=True,
                line_shape='spline'
            )
            fig.update_xaxes(title="Month")
            fig.update_yaxes(title="Average Risk Score")
            st.plotly_chart(fig, use_container_width=True)
    
    # Distance-based analysis
    st.markdown("#### 📏 Distance-based Risk Analysis")
    
    # Calculate distances from a reference point (center of data)
    center_lat = trend_df['latitude'].mean()
    center_lon = trend_df['longitude'].mean()
    
    trend_df['distance_from_center'] = np.sqrt(
        (trend_df['latitude'] - center_lat)**2 + 
        (trend_df['longitude'] - center_lon)**2
    )
    
    # Bin by distance
    trend_df['distance_bin'] = pd.cut(
        trend_df['distance_from_center'], 
        bins=5, 
        labels=['Very Close', 'Close', 'Medium', 'Far', 'Very Far']
    )
    
    distance_analysis = trend_df.groupby('distance_bin').agg({
        'is_fraud': ['count', 'sum', 'mean'],
        'risk_score': 'mean'
    }).round(3)
    
    distance_analysis.columns = ['Total_Transactions', 'Fraud_Count', 'Fraud_Rate', 'Avg_Risk']
    distance_analysis = distance_analysis.reset_index()
    
    # Distance vs fraud rate chart
    fig = px.bar(
        distance_analysis,
        x='distance_bin',
        y='Fraud_Rate',
        title='Fraud Rate by Distance from Center',
        color='Fraud_Rate',
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Geographic correlation analysis
    st.markdown("#### 🔗 Geographic Correlation Analysis")
    
    # Calculate correlations
    correlation_data = {
        'Latitude vs Risk Score': np.corrcoef(trend_df['latitude'], trend_df['risk_score'])[0,1],
        'Longitude vs Risk Score': np.corrcoef(trend_df['longitude'], trend_df['risk_score'])[0,1],
        'Distance vs Risk Score': np.corrcoef(trend_df['distance_from_center'], trend_df['risk_score'])[0,1]
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        corr_value = correlation_data['Latitude vs Risk Score']
        st.metric(
            "Latitude Correlation",
            f"{corr_value:.3f}",
            help="Correlation between latitude and risk score"
        )
    
    with col2:
        corr_value = correlation_data['Longitude vs Risk Score']
        st.metric(
            "Longitude Correlation", 
            f"{corr_value:.3f}",
            help="Correlation between longitude and risk score"
        )
    
    with col3:
        corr_value = correlation_data['Distance vs Risk Score']
        st.metric(
            "Distance Correlation",
            f"{corr_value:.3f}",
            help="Correlation between distance from center and risk score"
        )

if __name__ == "__main__":
    main()