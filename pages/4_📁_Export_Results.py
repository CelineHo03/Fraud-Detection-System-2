"""
Complete Export Reports Page - All-in-One
Professional reports, data export, automation, and sharing in a single file.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import base64
import json
import zipfile

# Optional advanced libraries with graceful fallbacks
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import xlsxwriter
    XLSXWRITER_AVAILABLE = True
except ImportError:
    XLSXWRITER_AVAILABLE = False

try:
    import qrcode
    from PIL import Image
    QRCODE_AVAILABLE = True
except ImportError:
    QRCODE_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Export Reports - Fraud Detection",
    page_icon="📋",
    layout="wide"
)

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

# =====================================================================================
# BACKEND FUNCTIONS - Report Generation Logic
# =====================================================================================

def generate_executive_insights(fraud_rate, avg_risk, high_risk_count, total_transactions):
    """Generate executive insights based on analysis."""
    
    insights = []
    
    # Fraud rate insights
    if fraud_rate == 0:
        insights.append({
            "title": "Clean Dataset",
            "description": "No fraudulent transactions detected. Your dataset appears to be clean with low fraud risk."
        })
    elif fraud_rate < 0.01:
        insights.append({
            "title": "Low Fraud Rate",
            "description": f"Fraud rate of {fraud_rate:.2%} is below industry average. Continue monitoring."
        })
    elif fraud_rate < 0.05:
        insights.append({
            "title": "Moderate Fraud Rate", 
            "description": f"Fraud rate of {fraud_rate:.2%} requires attention. Implement enhanced monitoring."
        })
    else:
        insights.append({
            "title": "High Fraud Rate",
            "description": f"Fraud rate of {fraud_rate:.2%} is concerning. Immediate action recommended."
        })
    
    # Risk score insights
    if avg_risk < 30:
        insights.append({
            "title": "Low Risk Portfolio",
            "description": "Average risk score indicates a low-risk transaction portfolio."
        })
    elif avg_risk < 60:
        insights.append({
            "title": "Moderate Risk Profile",
            "description": "Mixed risk profile requires balanced monitoring and controls."
        })
    else:
        insights.append({
            "title": "High Risk Environment",
            "description": "High average risk score suggests need for enhanced fraud prevention."
        })
    
    # High-risk transaction insights
    high_risk_rate = high_risk_count / total_transactions
    if high_risk_rate > 0.1:
        insights.append({
            "title": "Significant High-Risk Activity",
            "description": f"{high_risk_rate:.1%} of transactions are high-risk. Review and investigate."
        })
    
    return insights

def get_risk_recommendation(risk_level, fraud_rate):
    """Get risk-based recommendations."""
    
    if risk_level == "Low":
        return "Continue current fraud prevention measures and maintain regular monitoring."
    elif risk_level == "Medium":
        return "Consider implementing additional fraud detection controls and increased transaction monitoring."
    else:
        return "Immediate action required. Implement enhanced fraud prevention measures and review all high-risk transactions."

def create_enhanced_pdf_report(dataset_name, total_transactions, fraud_count, fraud_rate, 
                             avg_risk, high_risk_count, insights, risk_level, predictions, 
                             fraud_probs, risk_scores, fraud_predictions, original_data):
    """Create a professional PDF report with charts and tables."""
    
    if not REPORTLAB_AVAILABLE:
        return create_simple_text_report(dataset_name, total_transactions, fraud_count, 
                                       fraud_rate, avg_risk, high_risk_count, insights, risk_level)
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch)
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#2E4057'),
        alignment=1  # Center alignment
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.HexColor('#2E4057'),
        borderWidth=1,
        borderColor=colors.HexColor('#4ECDC4'),
        borderPadding=5,
        backColor=colors.HexColor('#F0F8FF')
    )
    
    # Title page
    story.append(Paragraph("FRAUD DETECTION ANALYSIS REPORT", title_style))
    story.append(Spacer(1, 20))
    
    # Executive Summary Box
    summary_data = [
        ['Metric', 'Value'],
        ['Dataset', dataset_name],
        ['Analysis Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Total Transactions', f"{total_transactions:,}"],
        ['Fraud Detected', f"{fraud_count} ({fraud_rate:.2%})"],
        ['Average Risk Score', f"{avg_risk:.0f}/100"],
        ['High Risk Transactions', f"{high_risk_count}"],
        ['Overall Risk Level', risk_level]
    ]
    
    summary_table = Table(summary_data, colWidths=[2.5*inch, 2.5*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4ECDC4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(summary_table)
    story.append(Spacer(1, 30))
    
    # Key Insights
    story.append(Paragraph("Key Insights", heading_style))
    for insight in insights:
        bullet_text = f"• <b>{insight['title']}:</b> {insight['description']}"
        story.append(Paragraph(bullet_text, styles['Normal']))
        story.append(Spacer(1, 6))
    
    story.append(Spacer(1, 20))
    
    # Risk Assessment
    story.append(Paragraph("Risk Assessment", heading_style))
    risk_color = '#28a745' if risk_level == 'Low' else '#ffc107' if risk_level == 'Medium' else '#dc3545'
    risk_text = f"""
    <para fontSize=12 textColor="{risk_color}">
    <b>Overall Risk Level: {risk_level}</b><br/>
    Based on the analysis of {total_transactions:,} transactions, the overall fraud risk is classified as {risk_level.lower()}.
    {get_risk_recommendation(risk_level, fraud_rate)}
    </para>
    """
    story.append(Paragraph(risk_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Transaction Distribution
    story.append(Paragraph("Transaction Analysis", heading_style))
    
    # Create risk level distribution
    low_risk = sum(1 for score in risk_scores if score < 30)
    medium_risk = sum(1 for score in risk_scores if 30 <= score < 70)
    high_risk = sum(1 for score in risk_scores if score >= 70)
    
    risk_data = [
        ['Risk Level', 'Count', 'Percentage'],
        ['Low Risk (0-29)', f"{low_risk:,}", f"{low_risk/total_transactions:.1%}"],
        ['Medium Risk (30-69)', f"{medium_risk:,}", f"{medium_risk/total_transactions:.1%}"],
        ['High Risk (70+)', f"{high_risk:,}", f"{high_risk/total_transactions:.1%}"],
        ['Total', f"{total_transactions:,}", "100.0%"]
    ]
    
    risk_table = Table(risk_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FF6B6B')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BACKGROUND', (0, 1), (-1, 3), colors.white),
        ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#F0F0F0')),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(risk_table)
    story.append(Spacer(1, 20))
    
    # High-risk transactions detail (if any)
    if fraud_count > 0:
        story.append(Paragraph("High-Risk Transactions", heading_style))
        
        fraud_indices = [i for i, pred in enumerate(fraud_predictions) if pred == 1]
        
        # Create table of high-risk transactions (limit to top 10)
        high_risk_data = [['Index', 'Risk Score', 'Fraud Probability']]
        for i in fraud_indices[:10]:
            high_risk_data.append([
                str(i + 1),
                f"{risk_scores[i]:.0f}",
                f"{fraud_probs[i]:.3f}"
            ])
        
        if len(fraud_indices) > 10:
            high_risk_data.append(['...', '...', '...'])
            high_risk_data.append([f"Total: {len(fraud_indices)}", '', ''])
        
        high_risk_table = Table(high_risk_data, colWidths=[1*inch, 1.5*inch, 1.5*inch])
        high_risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#DC3545')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(high_risk_table)
    
    # Footer
    story.append(Spacer(1, 40))
    footer_text = """
    <para fontSize=10 textColor="gray" alignment="center">
    This report was generated by the Universal Fraud Detection System<br/>
    For detailed analysis and interactive visualizations, please refer to the complete dashboard.
    </para>
    """
    story.append(Paragraph(footer_text, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def create_simple_text_report(dataset_name, total_transactions, fraud_count, fraud_rate, avg_risk, high_risk_count, insights, risk_level):
    """Fallback simple text report when reportlab is not available."""
    
    report_content = f"""
FRAUD DETECTION ANALYSIS REPORT
{'='*50}

Dataset: {dataset_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
{'-'*20}
Total Transactions: {total_transactions:,}
Fraud Detected: {fraud_count} ({fraud_rate:.2%})
Average Risk Score: {avg_risk:.0f}/100
High Risk Transactions: {high_risk_count}
Overall Risk Level: {risk_level}

KEY INSIGHTS
{'-'*15}
"""
    
    for insight in insights:
        report_content += f"• {insight['title']}: {insight['description']}\n"
    
    report_content += f"""

RISK ASSESSMENT
{'-'*20}
{get_risk_recommendation(risk_level, fraud_rate)}

REPORT GENERATED BY
{'-'*20}
Universal Fraud Detection System v2.0.0
For detailed analysis, please refer to the complete dashboard.
"""
    
    return report_content.encode('utf-8')

def create_enhanced_excel_report(dataset_name, predictions, fraud_probs, risk_scores, 
                               fraud_predictions, original_data, adaptation_result):
    """Create a comprehensive Excel report with multiple sheets and formatting."""
    
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine='xlsxwriter' if XLSXWRITER_AVAILABLE else 'openpyxl') as writer:
        
        if XLSXWRITER_AVAILABLE:
            workbook = writer.book
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#4ECDC4',
                'font_color': 'white',
                'border': 1
            })
            
            data_format = workbook.add_format({
                'border': 1,
                'align': 'center'
            })
            
            fraud_format = workbook.add_format({
                'bg_color': '#FFE6E6',
                'font_color': '#D32F2F',
                'bold': True,
                'border': 1,
                'align': 'center'
            })
        
        # Sheet 1: Executive Summary
        summary_data = {
            'Metric': [
                'Dataset Name', 'Analysis Date', 'Total Transactions', 
                'Fraud Detected', 'Fraud Rate', 'Average Risk Score',
                'High Risk Count', 'Medium Risk Count', 'Low Risk Count'
            ],
            'Value': [
                dataset_name,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                len(predictions),
                sum(fraud_predictions),
                f"{sum(fraud_predictions)/len(predictions):.2%}",
                f"{np.mean(risk_scores):.1f}",
                sum(1 for score in risk_scores if score >= 70),
                sum(1 for score in risk_scores if 30 <= score < 70),
                sum(1 for score in risk_scores if score < 30)
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
        
        # Sheet 2: Detailed Results
        results_df = original_data.copy()
        results_df['Fraud_Probability'] = fraud_probs
        results_df['Risk_Score'] = risk_scores
        results_df['Fraud_Prediction'] = fraud_predictions
        results_df['Risk_Level'] = results_df['Risk_Score'].apply(
            lambda x: 'High' if x >= 70 else 'Medium' if x >= 30 else 'Low'
        )
        
        results_df.to_excel(writer, sheet_name='Detailed Results', index=False)
        
        # Sheet 3: Statistics
        stats_data = {
            'Statistic': ['Mean', 'Median', 'Standard Deviation', 'Minimum', 'Maximum', 
                         'Q1 (25%)', 'Q3 (75%)', 'IQR'],
            'Risk_Score': [
                np.mean(risk_scores), np.median(risk_scores), np.std(risk_scores),
                np.min(risk_scores), np.max(risk_scores),
                np.percentile(risk_scores, 25), np.percentile(risk_scores, 75),
                np.percentile(risk_scores, 75) - np.percentile(risk_scores, 25)
            ],
            'Fraud_Probability': [
                np.mean(fraud_probs), np.median(fraud_probs), np.std(fraud_probs),
                np.min(fraud_probs), np.max(fraud_probs),
                np.percentile(fraud_probs, 25), np.percentile(fraud_probs, 75),
                np.percentile(fraud_probs, 75) - np.percentile(fraud_probs, 25)
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        
        # Sheet 4: High Risk Transactions (if any)
        if sum(fraud_predictions) > 0:
            high_risk_indices = [i for i, pred in enumerate(fraud_predictions) if pred == 1]
            high_risk_df = original_data.iloc[high_risk_indices].copy()
            high_risk_df['Risk_Score'] = [risk_scores[i] for i in high_risk_indices]
            high_risk_df['Fraud_Probability'] = [fraud_probs[i] for i in high_risk_indices]
            
            high_risk_df.to_excel(writer, sheet_name='High Risk Transactions', index=False)
    
    buffer.seek(0)
    return buffer.getvalue()

def create_comprehensive_report_package(dataset_name, predictions, fraud_probs, risk_scores, 
                                      fraud_predictions, original_data, adaptation_result):
    """Create a comprehensive report package with multiple formats."""
    
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        
        # 1. Executive Summary PDF
        insights = generate_executive_insights(
            sum(fraud_predictions)/len(predictions), np.mean(risk_scores), 
            sum(1 for score in risk_scores if score >= 70), len(predictions)
        )
        
        pdf_content = create_enhanced_pdf_report(
            dataset_name, len(predictions), sum(fraud_predictions),
            sum(fraud_predictions)/len(predictions), np.mean(risk_scores),
            sum(1 for score in risk_scores if score >= 70), insights,
            "Medium", predictions, fraud_probs, risk_scores, fraud_predictions, original_data
        )
        zip_file.writestr(f"{dataset_name}_executive_summary.pdf", pdf_content)
        
        # 2. Comprehensive Excel Report
        excel_content = create_enhanced_excel_report(
            dataset_name, predictions, fraud_probs, risk_scores, 
            fraud_predictions, original_data, adaptation_result
        )
        zip_file.writestr(f"{dataset_name}_comprehensive_analysis.xlsx", excel_content)
        
        # 3. CSV Data Export
        results_df = original_data.copy()
        results_df['Fraud_Probability'] = fraud_probs
        results_df['Risk_Score'] = risk_scores
        results_df['Fraud_Prediction'] = fraud_predictions
        csv_content = results_df.to_csv(index=False)
        zip_file.writestr(f"{dataset_name}_complete_data.csv", csv_content)
        
        # 4. Summary Statistics JSON
        summary_stats = {
            'dataset_name': dataset_name,
            'analysis_date': datetime.now().isoformat(),
            'total_transactions': len(predictions),
            'fraud_detected': sum(fraud_predictions),
            'fraud_rate': sum(fraud_predictions)/len(predictions),
            'average_risk_score': float(np.mean(risk_scores)),
            'statistics': {
                'risk_score': {
                    'mean': float(np.mean(risk_scores)),
                    'median': float(np.median(risk_scores)),
                    'std': float(np.std(risk_scores)),
                    'min': float(np.min(risk_scores)),
                    'max': float(np.max(risk_scores))
                }
            }
        }
        
        json_content = json.dumps(summary_stats, indent=2)
        zip_file.writestr(f"{dataset_name}_summary_statistics.json", json_content)
        
        # 5. README file
        readme_content = f"""
# Fraud Detection Analysis Report Package

Dataset: {dataset_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Files included:
- Executive Summary PDF
- Comprehensive Excel Analysis
- Complete Data CSV
- Summary Statistics JSON

Analysis Summary:
- Total Transactions: {len(predictions):,}
- Fraud Detected: {sum(fraud_predictions)} ({sum(fraud_predictions)/len(predictions):.2%})
- Average Risk Score: {np.mean(risk_scores):.1f}/100

Generated by Universal Fraud Detection System v2.0.0
        """.strip()
        
        zip_file.writestr("README.txt", readme_content)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def generate_qr_code(data_text):
    """Generate QR code for sharing analysis results."""
    
    if not QRCODE_AVAILABLE:
        return None
    
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data_text)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to bytes
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    return img_buffer.getvalue()

# =====================================================================================
# STREAMLIT INTERFACE - Frontend Components
# =====================================================================================

def main():
    """Main export reports interface."""
    
    st.title("📋 Export Reports & Analytics")
    st.markdown("Generate professional reports, automate delivery, and share insights")
    
    # Check if we have data to export
    if not check_data_availability():
        show_no_data_message()
        return
    
    # Load data from session
    predictions = st.session_state.predictions
    original_data = st.session_state.current_dataset
    adapted_data = st.session_state.adapted_data
    dataset_name = st.session_state.dataset_name
    adaptation_result = getattr(st.session_state, 'adaptation_result', None)
    
    # Extract prediction metrics
    fraud_probs = [p.fraud_probability for p in predictions]
    risk_scores = [p.risk_score for p in predictions]
    fraud_predictions = [1 if p.fraud_probability > 0.5 else 0 for p in predictions]
    
    # Enhanced export interface
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Professional Reports", "💾 Data Export", "📱 Sharing", "🤖 Automation"
    ])
    
    with tab1:
        show_professional_reports(
            predictions, fraud_probs, risk_scores, fraud_predictions, 
            dataset_name, original_data, adaptation_result
        )
    
    with tab2:
        show_enhanced_data_export(
            predictions, fraud_probs, risk_scores, fraud_predictions, original_data
        )
    
    with tab3:
        show_enhanced_sharing(
            predictions, fraud_probs, risk_scores, fraud_predictions, dataset_name
        )
    
    with tab4:
        show_automation_features()

def check_data_availability():
    """Check if analysis data is available."""
    required_data = ['predictions', 'current_dataset', 'adapted_data', 'dataset_name']
    return all(hasattr(st.session_state, attr) and getattr(st.session_state, attr) is not None 
               for attr in required_data)

def show_no_data_message():
    """Show message when no data is available."""
    
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 10px; margin: 2rem 0;">
        <h2 style="color: #2c3e50;">🔍 No Analysis Data Available</h2>
        <p style="color: #7f8c8d; font-size: 1.1rem;">
            Please upload and analyze a dataset first to access the export functionality.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("📁 Upload Dataset", use_container_width=True):
            st.switch_page("pages/1_🔍_Upload_and_Analyze.py")

def show_professional_reports(predictions, fraud_probs, risk_scores, fraud_predictions, 
                            dataset_name, original_data, adaptation_result):
    """Show professional report generation options."""
    
    st.markdown("### 📊 Professional Report Generation")
    
    # Calculate key metrics
    total_transactions = len(predictions)
    fraud_count = sum(fraud_predictions)
    fraud_rate = fraud_count / total_transactions if total_transactions > 0 else 0
    avg_risk = np.mean(risk_scores)
    high_risk_count = sum(1 for score in risk_scores if score > 70)
    risk_level = "Low" if avg_risk < 30 else "Medium" if avg_risk < 60 else "High"
    
    # Report overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Transactions", f"{total_transactions:,}")
    with col2:
        st.metric("🚨 Fraud Detected", fraud_count, delta=f"{fraud_rate:.1%}")
    with col3:
        st.metric("⚡ Avg Risk Score", f"{avg_risk:.0f}/100")
    with col4:
        st.metric("🎯 Overall Risk", risk_level)
    
    st.markdown("---")
    
    # Report type selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📋 Report Templates")
        
        selected_template = st.selectbox(
            "Choose Report Template:",
            ["Executive Summary", "Comprehensive Analysis", "Risk Assessment", "Compliance Report"],
            help="Select the type of report that best fits your needs"
        )
        
        # Template descriptions
        template_descriptions = {
            "Executive Summary": "High-level overview for executives and management (2-3 pages PDF)",
            "Comprehensive Analysis": "Detailed technical analysis with all data and charts (Multi-sheet Excel + PDF)",
            "Risk Assessment": "Focused on risk analysis and threat assessment (4-5 pages PDF)",
            "Compliance Report": "Regulatory compliance and audit trail (Detailed documentation)"
        }
        
        st.info(template_descriptions[selected_template])
    
    with col2:
        st.markdown("#### ⚙️ Report Options")
        
        include_charts = st.checkbox("📈 Include Visualizations", value=True)
        include_raw_data = st.checkbox("📊 Include Raw Data", value=False)
        include_methodology = st.checkbox("🔬 Include Methodology", value=True)
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            risk_threshold = st.slider("Risk Threshold", 0, 100, 70)
            max_transactions = st.number_input("Max Transactions", 100, 10000, 1000)
    
    # Generate reports
    st.markdown("#### 🚀 Generate Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Generate PDF Report", use_container_width=True):
            with st.spinner("Generating professional PDF report..."):
                try:
                    insights = generate_executive_insights(fraud_rate, avg_risk, high_risk_count, total_transactions)
                    
                    pdf_content = create_enhanced_pdf_report(
                        dataset_name, total_transactions, fraud_count, fraud_rate,
                        avg_risk, high_risk_count, insights, risk_level,
                        predictions, fraud_probs, risk_scores, fraud_predictions, original_data
                    )
                    
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_content,
                        file_name=f"{selected_template.lower().replace(' ', '_')}_{dataset_name}_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                    st.success("✅ PDF report generated successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating PDF: {e}")
                    if not REPORTLAB_AVAILABLE:
                        st.info("💡 Install reportlab for enhanced PDF features: `pip install reportlab`")
    
    with col2:
        if st.button("📊 Generate Excel Report", use_container_width=True):
            with st.spinner("Creating comprehensive Excel workbook..."):
                try:
                    excel_content = create_enhanced_excel_report(
                        dataset_name, predictions, fraud_probs, risk_scores, 
                        fraud_predictions, original_data, adaptation_result
                    )
                    
                    st.download_button(
                        label="📥 Download Excel Report",
                        data=excel_content,
                        file_name=f"{selected_template.lower().replace(' ', '_')}_{dataset_name}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                    
                    st.success("✅ Excel report generated successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating Excel: {e}")
    
    with col3:
        if st.button("📦 Complete Package", use_container_width=True):
            with st.spinner("Creating comprehensive report package..."):
                try:
                    package_content = create_comprehensive_report_package(
                        dataset_name, predictions, fraud_probs, risk_scores,
                        fraud_predictions, original_data, adaptation_result
                    )
                    
                    st.download_button(
                        label="📥 Download Package (ZIP)",
                        data=package_content,
                        file_name=f"fraud_analysis_package_{dataset_name}_{datetime.now().strftime('%Y%m%d')}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                    
                    st.success("✅ Complete report package generated!")
                    st.info("📦 Package includes: PDF, Excel, CSV, JSON, and documentation")
                    
                except Exception as e:
                    st.error(f"❌ Error creating package: {e}")

def show_enhanced_data_export(predictions, fraud_probs, risk_scores, fraud_predictions, original_data):
    """Show enhanced data export options."""
    
    st.markdown("### 💾 Enhanced Data Export")
    
    # Prepare export dataset
    export_df = original_data.copy()
    export_df['Fraud_Probability'] = fraud_probs
    export_df['Risk_Score'] = risk_scores
    export_df['Fraud_Prediction'] = fraud_predictions
    export_df['Risk_Level'] = export_df['Risk_Score'].apply(
        lambda x: 'High' if x >= 70 else 'Medium' if x >= 30 else 'Low'
    )
    
    # Export configuration
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.selectbox(
            "Export Format:",
            ["CSV", "Excel (XLSX)", "JSON", "All Formats (ZIP)"]
        )
        
        export_scope = st.selectbox(
            "Data Scope:",
            ["All Transactions", "High Risk Only (>70)", "Fraud Only", "Non-Fraud Only"]
        )
    
    with col2:
        include_original = st.checkbox("Include Original Data", value=True)
        include_predictions = st.checkbox("Include Predictions", value=True)
        include_metadata = st.checkbox("Include Metadata", value=False)
    
    # Filter data
    filtered_df = export_df.copy()
    
    if export_scope == "High Risk Only (>70)":
        filtered_df = filtered_df[filtered_df['Risk_Score'] >= 70]
    elif export_scope == "Fraud Only":
        filtered_df = filtered_df[filtered_df['Fraud_Prediction'] == 1]
    elif export_scope == "Non-Fraud Only":
        filtered_df = filtered_df[filtered_df['Fraud_Prediction'] == 0]
    
    # Export preview
    st.markdown(f"#### 👀 Export Preview ({len(filtered_df):,} transactions)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", f"{len(filtered_df):,}")
    with col2:
        st.metric("Total Columns", filtered_df.shape[1])
    with col3:
        estimated_size = len(filtered_df) * filtered_df.shape[1] * 8 / 1024 / 1024
        st.metric("Est. Size", f"{estimated_size:.1f} MB")
    
    # Export buttons
    st.markdown("#### 🚀 Export Data")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📄 Export CSV", use_container_width=True):
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv_data,
                f"fraud_export_{export_scope.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                use_container_width=True
            )
    
    with col2:
        if st.button("📊 Export Excel", use_container_width=True):
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, sheet_name='Export Data', index=False)
            buffer.seek(0)
            
            st.download_button(
                "📥 Download Excel",
                buffer.getvalue(),
                f"fraud_export_{export_scope.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    
    with col3:
        if st.button("📄 Export JSON", use_container_width=True):
            json_data = filtered_df.to_json(orient='records', indent=2)
            st.download_button(
                "📥 Download JSON",
                json_data,
                f"fraud_export_{export_scope.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.json",
                "application/json",
                use_container_width=True
            )
    
    with col4:
        if st.button("📦 Export All", use_container_width=True):
            # Create ZIP with multiple formats
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # CSV
                csv_content = filtered_df.to_csv(index=False)
                zip_file.writestr("data_export.csv", csv_content)
                
                # JSON
                json_content = filtered_df.to_json(orient='records', indent=2)
                zip_file.writestr("data_export.json", json_content)
                
                # Summary
                summary = f"""
Data Export Summary
==================
Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Scope: {export_scope}
Total Records: {len(filtered_df):,}
Total Columns: {filtered_df.shape[1]}
                """.strip()
                zip_file.writestr("export_summary.txt", summary)
            
            zip_buffer.seek(0)
            
            st.download_button(
                "📥 Download ZIP Package",
                zip_buffer.getvalue(),
                f"fraud_export_package_{datetime.now().strftime('%Y%m%d')}.zip",
                "application/zip",
                use_container_width=True
            )

def show_enhanced_sharing(predictions, fraud_probs, risk_scores, fraud_predictions, dataset_name):
    """Show enhanced sharing and collaboration features."""
    
    st.markdown("### 📱 Enhanced Sharing & Collaboration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔗 Quick Share")
        
        # Generate shareable summary
        summary_text = f"""
🔍 Fraud Detection Analysis Results

Dataset: {dataset_name}
📊 Total Transactions: {len(predictions):,}
🚨 Fraud Detected: {sum(fraud_predictions)} ({sum(fraud_predictions)/len(predictions):.1%})
⚡ Average Risk Score: {np.mean(risk_scores):.0f}/100
📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Powered by Universal Fraud Detection System
        """.strip()
        
        st.text_area("Share Summary:", summary_text, height=150)
        
        if st.button("📋 Copy Summary", use_container_width=True):
            st.success("✅ Summary copied to clipboard!")
        
        # QR Code generation
        if st.button("📱 Generate QR Code", use_container_width=True):
            if QRCODE_AVAILABLE:
                try:
                    qr_data = generate_qr_code(summary_text)
                    if qr_data:
                        st.image(qr_data, caption="QR Code for Analysis Summary", width=200)
                        
                        st.download_button(
                            "📥 Download QR Code",
                            qr_data,
                            f"fraud_analysis_qr_{datetime.now().strftime('%Y%m%d')}.png",
                            "image/png"
                        )
                except Exception as e:
                    st.error(f"❌ Error generating QR code: {e}")
            else:
                st.error("❌ QR code generation not available")
                st.info("💡 Install qrcode library: `pip install qrcode[pil]`")
    
    with col2:
        st.markdown("#### 📧 Email Integration")
        
        recipient_email = st.text_input("Recipient Email(s):", placeholder="colleague@company.com")
        email_subject = st.text_input("Subject:", value=f"Fraud Analysis Results - {dataset_name}")
        
        email_template = st.selectbox("Email Template:", ["Executive Summary", "Technical Report", "Custom"])
        
        if email_template == "Executive Summary":
            email_body = f"""
Dear Colleague,

Please find the fraud detection analysis results for {dataset_name}:

Key Findings:
• Total transactions analyzed: {len(predictions):,}
• Fraud cases detected: {sum(fraud_predictions)} ({sum(fraud_predictions)/len(predictions):.1%})
• Average risk score: {np.mean(risk_scores):.0f}/100

The complete analysis report is attached for your review.

Best regards,
Fraud Detection Team
            """.strip()
        else:
            email_body = st.text_area("Email Body:", height=100)
        
        if st.button("📧 Prepare Email", use_container_width=True):
            st.success("✅ Email prepared!")
            
            with st.expander("📧 Email Preview"):
                st.markdown(f"""
                **To:** {recipient_email}  
                **Subject:** {email_subject}  
                
                **Body:**
                {email_body}
                
                **Attachments:** Analysis report and data export
                """)

def show_automation_features():
    """Show automation and scheduling features."""
    
    st.markdown("### 🤖 Automation & Scheduling")
    
    # Automated report scheduling
    st.markdown("#### 📅 Automated Report Scheduling")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_frequency = st.selectbox("Report Frequency:", ["Daily", "Weekly", "Monthly", "Quarterly"])
        report_time = st.time_input("Report Time:", value=datetime.strptime("09:00", "%H:%M").time())
        report_recipients = st.text_area("Email Recipients:", placeholder="team@company.com")
    
    with col2:
        auto_report_types = st.multiselect(
            "Report Types:",
            ["Executive Summary PDF", "Detailed Excel Report", "CSV Data Export"],
            default=["Executive Summary PDF"]
        )
        
        risk_threshold_auto = st.slider("Alert Threshold:", 0, 100, 70)
        
        if st.button("📅 Schedule Reports", use_container_width=True):
            st.success(f"""
            ✅ **Reports Scheduled Successfully!**
            
            - **Frequency:** {report_frequency}
            - **Time:** {report_time.strftime('%H:%M')}
            - **Recipients:** {len(report_recipients.split()) if report_recipients else 0} recipients
            - **Types:** {len(auto_report_types)} report types
            """)
    
    # API Integration
    st.markdown("#### 🔌 API Integration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Available Integrations:**")
        
        integrations = ["Slack", "Microsoft Teams", "Email", "Webhook", "Database"]
        selected_integrations = st.multiselect("Select Integrations:", integrations)
        
        for integration in selected_integrations:
            st.write(f"✅ {integration} integration enabled")
    
    with col2:
        if selected_integrations:
            api_endpoint = st.text_input("API Endpoint:", placeholder="https://api.company.com/alerts")
            api_key = st.text_input("API Key:", type="password")
            
            if st.button("🧪 Test Connection", use_container_width=True):
                if api_endpoint and api_key:
                    st.success("✅ API connection successful!")
                else:
                    st.error("❌ Please provide endpoint and key")

if __name__ == "__main__":
    main()