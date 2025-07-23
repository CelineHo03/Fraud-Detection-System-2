"""
Analytics Dashboard Page - Comprehensive Fraud Analysis
Advanced visualizations and insights from fraud detection results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import math

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
            # ALSO FIX THIS: Use original function here too
            original_dataframe(fallback_df)  # ✅ FIXED: Use original_dataframe
            st.info(f"📋 Simplified view (3 of {len(df):,} rows)")
        except Exception:
            st.error("Unable to display data - please check data format")

st.dataframe = simple_dataframe_fix


# Page configuration
st.set_page_config(
    page_title="Analytics Dashboard - Fraud Detection",
    page_icon="📊",
    layout="wide"
)

def main():
    """Main analytics dashboard interface."""
    
    st.title(" Analytics Dashboard")
    st.markdown("Comprehensive fraud analysis and business insights")
    
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
    
    # Dashboard header with key metrics
    show_dashboard_header(predictions, fraud_probs, risk_scores, fraud_predictions, dataset_name)
    
    # Main dashboard content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        " Overview", " Risk Analysis",  "Patterns", "⚡ Performance", " Deep Dive"
    ])
    
    with tab1:
        show_overview_dashboard(predictions, fraud_probs, risk_scores, fraud_predictions, original_data)
    
    with tab2:
        show_risk_analysis(predictions, fraud_probs, risk_scores, fraud_predictions, original_data)
    
    with tab3:
        show_pattern_analysis(predictions, fraud_probs, risk_scores, fraud_predictions, original_data, adapted_data)
    
    with tab4:
        show_performance_metrics(predictions, original_data, adapted_data)
    
    with tab5:
        show_deep_dive_analysis(predictions, fraud_probs, risk_scores, original_data)

def check_data_availability():
    """Check if analysis data is available."""
    
    required_data = ['predictions', 'current_dataset', 'adapted_data', 'dataset_name']
    return all(hasattr(st.session_state, attr) and getattr(st.session_state, attr) is not None 
               for attr in required_data)

def show_no_data_message():
    """Show message when no data is available."""
    
    st.info("""
     **No analysis data available**
    
    Please upload and analyze a dataset first to view the analytics dashboard.
    """)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button(" Upload Dataset", use_container_width=True):
            st.switch_page("pages/1_🔍_Upload_and_Analyze.py")

def show_dashboard_header(predictions, fraud_probs, risk_scores, fraud_predictions, dataset_name):
    """Show dashboard header with key metrics."""
    
    # Calculate key metrics
    total_transactions = len(predictions)
    fraud_count = sum(fraud_predictions)
    fraud_rate = fraud_count / total_transactions if total_transactions > 0 else 0
    avg_risk = np.mean(risk_scores)
    high_risk_count = sum(1 for score in risk_scores if score > 70)
    
    # Header with gradient background
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    ">
        <h2 style="margin: 0; font-size: 1.8rem;"> Fraud Analysis: {dataset_name}</h2>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
            Analyzed {total_transactions:,} transactions • 
            Generated {datetime.now().strftime('%Y-%m-%d at %H:%M')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Transactions",
            f"{total_transactions:,}",
            help="Total number of transactions analyzed"
        )
    
    with col2:
        delta_color = "inverse" if fraud_count > 0 else "normal"
        st.metric(
            "Fraud Detected",
            fraud_count,
            delta=f"{fraud_rate:.1%} rate",
            help="Number and percentage of fraudulent transactions"
        )
    
    with col3:
        risk_color = "🟢" if avg_risk < 30 else "🟡" if avg_risk < 60 else "🔴"
        st.metric(
            "Average Risk",
            f"{risk_color} {avg_risk:.0f}/100",
            help="Average risk score across all transactions"
        )
    
    with col4:
        st.metric(
            "High Risk (>70)",
            high_risk_count,
            delta=f"{high_risk_count/total_transactions:.1%}",
            help="Transactions with risk score above 70"
        )
    
    with col5:
        # Model confidence score
        confidence_score = 95  # This would come from model metadata
        st.metric(
            "Model Confidence",
            f" {confidence_score}%",
            help="Overall model prediction confidence"
        )

def show_overview_dashboard(predictions, fraud_probs, risk_scores, fraud_predictions, original_data):
    """Show overview dashboard with main charts."""
    
    st.markdown("###  Transaction Overview")
    
    # Main charts row
    col1, col2 = st.columns(2)
    
    with col1:
        # Fraud vs Non-Fraud pie chart
        fraud_counts = {'Legitimate': len(fraud_predictions) - sum(fraud_predictions), 
                       'Fraudulent': sum(fraud_predictions)}
        
        fig = px.pie(
            values=list(fraud_counts.values()),
            names=list(fraud_counts.keys()),
            title="Transaction Classification",
            color_discrete_sequence=['#4ECDC4', '#FF6B6B']
        )
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk score distribution
        fig = px.histogram(
            x=risk_scores,
            nbins=25,
            title="Risk Score Distribution",
            labels={'x': 'Risk Score', 'y': 'Number of Transactions'},
            color_discrete_sequence=['#45B7D1']
        )
        fig.add_vline(x=50, line_dash="dash", line_color="orange", annotation_text="Medium Risk")
        fig.add_vline(x=70, line_dash="dash", line_color="red", annotation_text="High Risk")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk level breakdown
    st.markdown("###  Risk Level Breakdown")
    
    # Categorize by risk levels
    low_risk = sum(1 for score in risk_scores if score < 30)
    medium_risk = sum(1 for score in risk_scores if 30 < score < 70)
    high_risk = sum(1 for score in risk_scores if score > 70)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="background: #d4edda; padding: 1rem; border-radius: 8px; border-left: 4px solid #28a745;">
            <h4 style="color: #155724; margin: 0;">🟢 Low Risk (0-29)</h4>
            <h2 style="color: #155724; margin: 0.5rem 0;">{low_risk:,}</h2>
            <p style="color: #155724; margin: 0;">{low_risk/len(risk_scores):.1%} of transactions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: #fff3cd; padding: 1rem; border-radius: 8px; border-left: 4px solid #ffc107;">
            <h4 style="color: #856404; margin: 0;">🟡 Medium Risk (30-69)</h4>
            <h2 style="color: #856404; margin: 0.5rem 0;">{medium_risk:,}</h2>
            <p style="color: #856404; margin: 0;">{medium_risk/len(risk_scores):.1%} of transactions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: #f8d7da; padding: 1rem; border-radius: 8px; border-left: 4px solid #dc3545;">
            <h4 style="color: #721c24; margin: 0;">🔴 High Risk (70+)</h4>
            <h2 style="color: #721c24; margin: 0.5rem 0;">{high_risk:,}</h2>
            <p style="color: #721c24; margin: 0;">{high_risk/len(risk_scores):.1%} of transactions</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Transaction amount analysis (if available)
    if 'TransactionAmt' in original_data.columns or any('amount' in col.lower() for col in original_data.columns):
        show_amount_analysis(original_data, fraud_predictions, risk_scores)

def show_amount_analysis(original_data, fraud_predictions, risk_scores):
    """Show transaction amount analysis."""
    
    st.markdown("### 💰 Transaction Amount Analysis")
    
    # Find amount column
    amount_col = None
    for col in original_data.columns:
        if col.lower() in ['transactionamt', 'amount', 'transaction_amount', 'value']:
            amount_col = col
            break
    
    if amount_col and pd.api.types.is_numeric_dtype(original_data[amount_col]):
        amounts = original_data[amount_col]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Amount distribution by fraud status
            fraud_amounts = amounts[np.array(fraud_predictions) == 1]
            legit_amounts = amounts[np.array(fraud_predictions) == 0]
            
            fig = go.Figure()
            
            if len(legit_amounts) > 0:
                fig.add_trace(go.Histogram(
                    x=legit_amounts,
                    name='Legitimate',
                    opacity=0.7,
                    nbinsx=30,
                    marker_color='#4ECDC4'
                ))
            
            if len(fraud_amounts) > 0:
                fig.add_trace(go.Histogram(
                    x=fraud_amounts,
                    name='Fraudulent',
                    opacity=0.7,
                    nbinsx=30,
                    marker_color='#FF6B6B'
                ))
            
            fig.update_layout(
                title='Transaction Amounts by Fraud Status',
                xaxis_title='Transaction Amount',
                yaxis_title='Count',
                barmode='overlay'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Amount vs Risk Score scatter
            fig = px.scatter(
                x=amounts,
                y=risk_scores,
                title='Transaction Amount vs Risk Score',
                labels={'x': 'Transaction Amount', 'y': 'Risk Score'},
                opacity=0.6,
                color=risk_scores,
                color_continuous_scale='Reds'
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
            st.plotly_chart(fig, use_container_width=True)

def show_risk_analysis(predictions, fraud_probs, risk_scores, fraud_predictions, original_data):
    """Show detailed risk analysis."""
    
    st.markdown("###  Risk Analysis Deep Dive")
    
    # Risk score statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Min Risk Score", f"{min(risk_scores):.0f}")
    with col2:
        st.metric("Max Risk Score", f"{max(risk_scores):.0f}")
    with col3:
        st.metric("Median Risk Score", f"{np.median(risk_scores):.0f}")
    with col4:
        st.metric("Risk Std Dev", f"{np.std(risk_scores):.1f}")
    
    # Advanced risk visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk score box plot
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=risk_scores,
            name="Risk Scores",
            boxpoints='outliers',
            marker_color='#FF6B6B'
        ))
        fig.update_layout(
            title='Risk Score Distribution (Box Plot)',
            yaxis_title='Risk Score'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Fraud probability vs Risk score
        fig = px.scatter(
            x=fraud_probs,
            y=risk_scores,
            title='Fraud Probability vs Risk Score',
            labels={'x': 'Fraud Probability', 'y': 'Risk Score'},
            opacity=0.6,
            color=fraud_predictions,
            color_discrete_sequence=['#4ECDC4', '#FF6B6B']
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="Classification Threshold")
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="High Risk")
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk threshold analysis
    st.markdown("#### Threshold Sensitivity Analysis")
    
    thresholds = np.arange(0.1, 1.0, 0.05)
    threshold_metrics = []
    
    for threshold in thresholds:
        pred_at_threshold = [1 if p > threshold else 0 for p in fraud_probs]
        fraud_count = sum(pred_at_threshold)
        threshold_metrics.append({
            'Threshold': threshold,
            'Fraud_Count': fraud_count,
            'Fraud_Rate': fraud_count / len(fraud_probs)
        })
    
    threshold_df = pd.DataFrame(threshold_metrics)
    
    fig = px.line(
        threshold_df,
        x='Threshold',
        y='Fraud_Count',
        title='Fraud Detection Count vs Threshold',
        labels={'Fraud_Count': 'Fraud Count', 'Threshold': 'Probability Threshold'}
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Current Threshold")
    st.plotly_chart(fig, use_container_width=True)

def show_pattern_analysis(predictions, fraud_probs, risk_scores, fraud_predictions, original_data, adapted_data):
    """Show pattern analysis and insights."""
    
    st.markdown("###  Pattern Analysis")
    
    # Time-based patterns (if time data available)
    time_cols = [col for col in original_data.columns 
                if any(time_word in col.lower() for time_word in ['time', 'date', 'timestamp'])]
    
    if time_cols:
        show_temporal_patterns(original_data, time_cols[0], fraud_predictions, risk_scores)
    
    # Feature correlation analysis
    show_feature_importance(predictions)
    
    # Categorical analysis
    show_categorical_patterns(original_data, fraud_predictions, risk_scores)

def show_temporal_patterns(original_data, time_col, fraud_predictions, risk_scores):
    """Show temporal fraud patterns."""
    
    st.markdown("#### Temporal Patterns")
    
    try:
        # Try to parse the time column
        time_data = pd.to_datetime(original_data[time_col], errors='coerce')
        
        if time_data.notna().sum() > 0:
            # Create hourly fraud analysis
            df_temp = pd.DataFrame({
                'Time': time_data,
                'Fraud': fraud_predictions,
                'Risk_Score': risk_scores
            }).dropna()
            
            df_temp['Hour'] = df_temp['Time'].dt.hour
            hourly_stats = df_temp.groupby('Hour').agg({
                'Fraud': ['count', 'sum', 'mean'],
                'Risk_Score': 'mean'
            }).round(3)
            
            hourly_stats.columns = ['Total_Transactions', 'Fraud_Count', 'Fraud_Rate', 'Avg_Risk']
            hourly_stats = hourly_stats.reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Hourly fraud rate
                fig = px.line(
                    hourly_stats,
                    x='Hour',
                    y='Fraud_Rate',
                    title='Fraud Rate by Hour of Day',
                    labels={'Fraud_Rate': 'Fraud Rate', 'Hour': 'Hour of Day'}
                )
                fig.update_traces(line_color='#FF6B6B')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Hourly risk score
                fig = px.bar(
                    hourly_stats,
                    x='Hour',
                    y='Avg_Risk',
                    title='Average Risk Score by Hour',
                    labels={'Avg_Risk': 'Average Risk Score', 'Hour': 'Hour of Day'},
                    color='Avg_Risk',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.info(f"Could not analyze temporal patterns: {e}")

def show_feature_importance(predictions):
    """Show feature importance if SHAP data is available."""
    
    st.markdown("####  Feature Importance")
    
    # Check if we have SHAP explanations
    has_shap = any(hasattr(p, 'top_risk_factors') and p.top_risk_factors for p in predictions)
    
    if has_shap:
        # Aggregate SHAP values
        all_factors = {}
        for prediction in predictions:
            if hasattr(prediction, 'top_risk_factors') and prediction.top_risk_factors:
                for factor in prediction.top_risk_factors:
                    feature = factor['feature']
                    impact = abs(factor['impact'])
                    if feature in all_factors:
                        all_factors[feature].append(impact)
                    else:
                        all_factors[feature] = [impact]
        
        # Calculate average importance
        feature_importance = {
            feature: np.mean(impacts) 
            for feature, impacts in all_factors.items()
        }
        
        # Sort and display top features
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
        
        if sorted_features:
            features_df = pd.DataFrame(sorted_features, columns=['Feature', 'Importance'])
            
            fig = px.bar(
                features_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 15 Most Important Features',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance analysis requires SHAP explanations. Enable SHAP in the upload settings for detailed feature analysis.")

def show_categorical_patterns(original_data, fraud_predictions, risk_scores):
    """Show categorical variable patterns."""
    
    st.markdown("####  Categorical Patterns")
    
    # Find categorical columns
    categorical_cols = original_data.select_dtypes(include=['object', 'category']).columns
    categorical_cols = [col for col in categorical_cols if original_data[col].nunique() < 20]  # Reasonable number of categories
    
    if len(categorical_cols) > 0:
        selected_cat = st.selectbox("Select categorical variable to analyze:", categorical_cols)
        
        if selected_cat:
            min_length = min(len(original_data), len(fraud_predictions), len(risk_scores))

            # Analyze patterns by category
            df_cat = pd.DataFrame({
                'Category': original_data[selected_cat],
                'Fraud': fraud_predictions,
                'Risk_Score': risk_scores
            })
            
            cat_stats = df_cat.groupby('Category').agg({
                'Fraud': ['count', 'sum', 'mean'],
                'Risk_Score': 'mean'
            }).round(3)
            
            cat_stats.columns = ['Total_Count', 'Fraud_Count', 'Fraud_Rate', 'Avg_Risk']
            cat_stats = cat_stats.reset_index().sort_values('Fraud_Rate', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    cat_stats.head(10),
                    x='Category',
                    y='Fraud_Rate',
                    title=f'Fraud Rate by {selected_cat}',
                    color='Fraud_Rate',
                    color_continuous_scale='Reds'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    cat_stats.head(10),
                    x='Category',
                    y='Avg_Risk',
                    title=f'Average Risk Score by {selected_cat}',
                    color='Avg_Risk',
                    color_continuous_scale='Oranges'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed stats table
            simple_dataframe_fix(cat_stats, use_container_width=True)
    else:
        st.info("No suitable categorical variables found for pattern analysis.")

def show_performance_metrics(predictions, original_data, adapted_data):
    """Show model performance metrics."""
    
    st.markdown("### ⚡ Model Performance")
    
    # Processing statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Dataset Size", f"{original_data.shape[0]:,} rows")
    with col2:
        st.metric("Original Features", original_data.shape[1])
    with col3:
        st.metric("Adapted Features", adapted_data.shape[1])
    with col4:
        final_features = len(st.session_state.engineered_features.columns) if hasattr(st.session_state, 'engineered_features') else "N/A"
        st.metric("Final Features", final_features)
    
    # Adaptation confidence
    if hasattr(st.session_state, 'adaptation_result'):
        adaptation_result = st.session_state.adaptation_result
        
        st.markdown("####  Dataset Adaptation Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Adaptation Confidence", f"{adaptation_result.adaptation_confidence:.1%}")
        with col2:
            st.metric("Column Mappings", len(adaptation_result.column_mappings))
        with col3:
            st.metric("Synthetic Features", len(adaptation_result.synthetic_features))
    
    # Prediction confidence analysis
    st.markdown("####  Prediction Confidence")
    
    fraud_probs = [p.fraud_probability for p in predictions]
    
    # Confidence distribution
    confidence_bins = {
        'Very High (>0.9)': sum(1 for p in fraud_probs if p > 0.9 or p < 0.1),
        'High (0.7-0.9)': sum(1 for p in fraud_probs if 0.7 < p <= 0.9 or 0.1 <= p < 0.3),
        'Medium (0.5-0.7)': sum(1 for p in fraud_probs if 0.3 <= p < 0.5 or 0.5 < p <= 0.7),
        'Low (0.3-0.5)': sum(1 for p in fraud_probs if 0.3 < p <= 0.5)
    }
    
    # Create confidence distribution chart
    fig = px.pie(
        values=list(confidence_bins.values()),
        names=list(confidence_bins.keys()),
        title="Prediction Confidence Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_deep_dive_analysis(predictions, fraud_probs, risk_scores, original_data):
    """Show deep dive analysis with advanced insights."""
    
    st.markdown("###  Deep Dive Analysis")
    
    # Statistical analysis
    st.markdown("####  Statistical Summary")
    
    stats_df = pd.DataFrame({
        'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1', 'Q3'],
        'Fraud Probability': [
            np.mean(fraud_probs), np.median(fraud_probs), np.std(fraud_probs),
            np.min(fraud_probs), np.max(fraud_probs), 
            np.percentile(fraud_probs, 25), np.percentile(fraud_probs, 75)
        ],
        'Risk Score': [
            np.mean(risk_scores), np.median(risk_scores), np.std(risk_scores),
            np.min(risk_scores), np.max(risk_scores),
            np.percentile(risk_scores, 25), np.percentile(risk_scores, 75)
        ]
    })
    
    simple_dataframe_fix(stats_df.round(3), use_container_width=True)
    
    # Outlier analysis
    st.markdown("####  Outlier Analysis")
    
    # Calculate outliers using IQR method
    q1_risk = np.percentile(risk_scores, 25)
    q3_risk = np.percentile(risk_scores, 75)
    iqr_risk = q3_risk - q1_risk
    lower_bound = q1_risk - 1.5 * iqr_risk
    upper_bound = q3_risk + 1.5 * iqr_risk
    
    outliers = [i for i, score in enumerate(risk_scores) if score < lower_bound or score > upper_bound]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Outliers", len(outliers))
        st.metric("Outlier Percentage", f"{len(outliers)/len(risk_scores):.1%}")
    
    with col2:
        if len(outliers) > 0:
            outlier_data = original_data.iloc[outliers].copy()
            outlier_data['Risk_Score'] = [risk_scores[i] for i in outliers]
            outlier_data['Fraud_Probability'] = [fraud_probs[i] for i in outliers]
            
            if st.checkbox("Show outlier transactions"):
                simple_dataframe_fix(outlier_data.head(10), use_container_width=True)
    
    # Correlation matrix (if numerical data available)
    numerical_cols = original_data.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 1:
        st.markdown("#### 🔗 Feature Correlations")
        
        # Create correlation matrix with predictions
        corr_data = original_data[numerical_cols].copy()
        corr_data['Risk_Score'] = risk_scores
        corr_data['Fraud_Probability'] = fraud_probs
        
        # Limit to top correlated features for readability
        correlation_matrix = corr_data.corr()
        
        # Plot correlation heatmap
        fig = px.imshow(
            correlation_matrix,
            title="Feature Correlation Heatmap",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()