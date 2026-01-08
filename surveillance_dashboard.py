"""
AI-Powered Surveillance System Dashboard
Web-based dashboard for monitoring alerts and system status
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
import json
import cv2
import numpy as np
import time
import threading
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import logging

# Import our surveillance system
from netraguard_main_system import SurveillanceSystem, DatabaseManager

# Configure page
st.set_page_config(
    page_title="AI Surveillance Dashboard",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff4b4b;
    }
    .alert-high {
    background: linear-gradient(90deg, #2A0F14, #1A0A0E);
    border-left: 5px solid #EF4444;
    padding: 0.75rem;
    margin: 0.4rem 0;
    border-radius: 0.4rem;
    box-shadow: 0 0 10px rgba(239,68,68,0.25);
    color: #F9FAFB;   /* ✅ WHITE TEXT */
}

.alert-medium {
    background: linear-gradient(90deg, #1F2933, #111827);
    border-left: 5px solid #F59E0B;
    padding: 0.75rem;
    margin: 0.4rem 0;
    border-radius: 0.4rem;
    color: #E5E7EB;   /* ✅ LIGHT GREY TEXT */
}

.alert-low {
    background: linear-gradient(90deg, #111827, #0E1117);
    border-left: 5px solid #3B82F6;
    padding: 0.75rem;
    margin: 0.4rem 0;
    border-radius: 0.4rem;
    color: #D1D5DB;   /* ✅ SOFT WHITE */
}

    .status-running {
        color: #28a745;
        font-weight: bold;
    }
    .status-stopped {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class SurveillanceDashboard:
    """Main dashboard class for the surveillance system"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.surveillance_system = None
        self.is_processing = False
        
        # Initialize session state
        if 'alerts_data' not in st.session_state:
            st.session_state.alerts_data = []
        if 'system_stats' not in st.session_state:
            st.session_state.system_stats = {
                'fps': 0,
                'detection_count': 0,
                'alert_count': 0,
                'processed_frames': 0
            }
    
    def load_alerts_data(self, hours: int = 24):
        """Load alerts data from database"""
        try:
            alerts = self.db_manager.get_recent_alerts(limit=1000)
            
            # Filter by time range
            cutoff_time = datetime.now() - timedelta(hours=hours)
            filtered_alerts = []
            
            for alert in alerts:
                alert_time = datetime.fromisoformat(alert['timestamp'])
                if alert_time >= cutoff_time:
                    filtered_alerts.append(alert)
            
            return filtered_alerts
        except Exception as e:
            st.error(f"Error loading alerts: {e}")
            return []
    
    def create_alerts_dataframe(self, alerts):
        """Convert alerts to pandas DataFrame"""
        if not alerts:
            return pd.DataFrame()
        
        df_data = []
        for alert in alerts:
            df_data.append({
                'Timestamp': alert['timestamp'],
                'Type': alert['anomaly_type'],
                'Confidence': alert['confidence'],
                'Severity': alert['severity'],
                'Description': alert['description']
            })
        
        df = pd.DataFrame(df_data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        return df
    
    def render_header(self):
        """Render dashboard header"""
        st.title("NetraGuard | AI Surveillance System Dashboard")
        st.markdown(

              "<p style='color:#9CA3AF; font-size:18px;'>Real-time CCTV analytics, anomaly detection, and intelligent alerting</p>",
               unsafe_allow_html=True
        )

        
        # Status indicator
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            status = "🟢 RUNNING" if self.is_processing else "🔴 STOPPED"
            status_class = "status-running" if self.is_processing else "status-stopped"
            st.markdown(f'<p class="{status_class}">System Status: {status}</p>', 
                       unsafe_allow_html=True)
        
        with col2:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.write(f"⏰ Current Time: {current_time}")
        
        with col3:
            if st.button("🔄 Refresh Data"):
                st.rerun()
    
    def render_metrics(self, alerts_df):
        """Render system metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_alerts = len(alerts_df) if not alerts_df.empty else 0
            st.metric("Total Alerts (24h)", total_alerts, 
                     delta=f"+{total_alerts}" if total_alerts > 0 else None)
        
        with col2:
            high_severity = len(alerts_df[alerts_df['Severity'] >= 3]) if not alerts_df.empty else 0
            st.metric("High Priority Alerts", high_severity,
                     delta=f"+{high_severity}" if high_severity > 0 else None,
                     delta_color="inverse")
        
        with col3:
            avg_confidence = alerts_df['Confidence'].mean() if not alerts_df.empty else 0
            st.metric("Avg Confidence", f"{avg_confidence:.2f}",
                     delta=f"{avg_confidence:.2f}" if avg_confidence > 0.5 else None)
        
        with col4:
            fps = st.session_state.system_stats.get('fps', 0)
            st.metric("Processing FPS", f"{fps:.1f}",
                     delta=f"{fps:.1f}" if fps > 15 else None,
                     delta_color="normal" if fps > 15 else "inverse")
    
    def render_alerts_timeline(self, alerts_df):
        """Render alerts timeline chart"""
        st.subheader("📊 Alerts Timeline")
        
        if alerts_df.empty:
            st.info("No alerts in the selected time range")
            return
        
        # Create hourly aggregation
        alerts_df['Hour'] = alerts_df['Timestamp'].dt.floor('H')
        hourly_counts = alerts_df.groupby(['Hour', 'Type']).size().reset_index(name='Count')
        
        # Create timeline chart
        fig = px.bar(hourly_counts, x='Hour', y='Count', color='Type',
                    title="Alerts by Hour and Type",
                    labels={'Count': 'Number of Alerts', 'Hour': 'Time'})
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Alert Count",
            showlegend=True,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_alert_types_distribution(self, alerts_df):
        """Render alert types distribution"""
        st.subheader("🎯 Alert Types Distribution")
        
        if alerts_df.empty:
            st.info("No alerts to display")
            return
        
        # Create pie chart for alert types
        type_counts = alerts_df['Type'].value_counts()
        
        fig = px.pie(values=type_counts.values, names=type_counts.index,
                    title="Distribution of Alert Types")
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_recent_alerts(self, alerts_df):
        """Render recent alerts list"""
        st.subheader("🚨 Recent Alerts")
        
        if alerts_df.empty:
            st.info("No recent alerts")
            return
        
        # Sort by timestamp descending
        recent_alerts = alerts_df.sort_values('Timestamp', ascending=False).head(10)
        
        for _, alert in recent_alerts.iterrows():
            severity_class = "alert-high" if alert['Severity'] >= 3 else \
                           "alert-medium" if alert['Severity'] == 2 else "alert-low"
            
            alert_html = f"""
            <div class="{severity_class}">
                <strong>{alert['Type'].upper()}</strong> - Confidence: {alert['Confidence']:.2f}<br>
                <small>{alert['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</small><br>
                <em>{alert['Description']}</em>
            </div>
            """
            st.markdown(alert_html, unsafe_allow_html=True)
    
    def render_system_controls(self):
        """Render system control panel"""
        st.subheader("🎛️ System Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Video Processing**")
            
            uploaded_file = st.file_uploader("Upload Video File", 
                                           type=['mp4', 'avi', 'mov', 'mkv'])
            
            if uploaded_file is not None:
                if st.button("Process Video"):
                    with st.spinner("Processing video..."):
                        # Save uploaded file temporarily
                        temp_path = f"temp_{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.read())
                        
                        # Process video
                        system = SurveillanceSystem()
                        system.process_video(temp_path, f"output_{uploaded_file.name}")
                        
                        # Clean up
                        Path(temp_path).unlink(missing_ok=True)
                        
                        st.success("Video processed successfully!")
        
        with col2:
            st.write("**System Configuration**")
            
            confidence_threshold = st.slider("Confidence Threshold", 
                                            min_value=0.1, max_value=1.0, 
                                            value=0.5, step=0.1)
            
            alert_sensitivity = st.selectbox("Alert Sensitivity", 
                                           ["Low", "Medium", "High"], 
                                           index=1)
            
            auto_refresh = st.checkbox("Auto Refresh (10s)", value=False)
            
            if auto_refresh:
                time.sleep(10)
                st.rerun()
    
    def render_statistics_panel(self, alerts_df):
        """Render detailed statistics panel"""
        st.subheader("📈 Detailed Statistics")
        
        if alerts_df.empty:
            st.info("No data available for statistics")
            return
        
        # Create tabs for different statistics
        tab1, tab2, tab3 = st.tabs(["Hourly Analysis", "Confidence Analysis", "Severity Analysis"])
        
        with tab1:
            # Hourly analysis
            alerts_df['Hour'] = alerts_df['Timestamp'].dt.hour
            hourly_stats = alerts_df.groupby('Hour').agg({
                'Type': 'count',
                'Confidence': 'mean',
                'Severity': 'mean'
            }).rename(columns={'Type': 'Count'})
            
            fig = px.line(x=hourly_stats.index, y=hourly_stats['Count'],
                         title="Alert Count by Hour of Day",
                         labels={'x': 'Hour', 'y': 'Alert Count'})
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Confidence analysis
            fig = px.histogram(alerts_df, x='Confidence', nbins=20,
                             title="Distribution of Alert Confidence Scores")
            st.plotly_chart(fig, use_container_width=True)
            
            # Confidence statistics
            st.write("**Confidence Statistics:**")
            st.write(f"- Mean: {alerts_df['Confidence'].mean():.2f}")
            st.write(f"- Median: {alerts_df['Confidence'].median():.2f}")
            st.write(f"- Std Dev: {alerts_df['Confidence'].std():.2f}")
        
        with tab3:
            # Severity analysis
            severity_counts = alerts_df['Severity'].value_counts().sort_index()
            severity_labels = {1: 'Low', 2: 'Medium', 3: 'High'}
            
            fig = px.bar(x=[severity_labels.get(i, str(i)) for i in severity_counts.index],
                        y=severity_counts.values,
                        title="Alert Count by Severity Level")
            st.plotly_chart(fig, use_container_width=True)
    
    def render_export_options(self, alerts_df):
        """Render data export options"""
        st.subheader("📥 Export Data")
        
        if alerts_df.empty:
            st.info("No data to export")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export as CSV"):
                csv = alerts_df.to_csv(index=False)
                st.download_button("Download CSV", csv, "alerts_export.csv", "text/csv")
        
        with col2:
            if st.button("Export as JSON"):
                json_data = alerts_df.to_json(orient='records', date_format='iso')
                st.download_button("Download JSON", json_data, "alerts_export.json", "application/json")
        
        with col3:
            if st.button("Generate Report"):
                report = self.generate_report(alerts_df)
                st.download_button("Download Report", report, "surveillance_report.txt", "text/plain")
    
    def generate_report(self, alerts_df):
        """Generate a text report of system activity"""
        if alerts_df.empty:
            return "No alerts to report."
        
        report = f"""
AI SURVEILLANCE SYSTEM REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
========================================

SUMMARY STATISTICS:
- Total Alerts: {len(alerts_df)}
- Average Confidence: {alerts_df['Confidence'].mean():.2f}
- High Severity Alerts: {len(alerts_df[alerts_df['Severity'] >= 3])}

ALERT TYPES:
{alerts_df['Type'].value_counts().to_string()}

HOURLY DISTRIBUTION:
{alerts_df.groupby(alerts_df['Timestamp'].dt.hour)['Type'].count().to_string()}

RECENT ALERTS (Last 5):
"""
        
        recent = alerts_df.sort_values('Timestamp', ascending=False).head(5)
        for _, alert in recent.iterrows():
            report += f"\n- {alert['Timestamp']}: {alert['Type']} (Confidence: {alert['Confidence']:.2f})"
        
        return report
    
    def run(self):
        """Main dashboard application"""
        self.render_header()
        
        # Sidebar for controls
        with st.sidebar:
            st.header("Dashboard Controls")
            
            # Time range selector
            time_range = st.selectbox("Time Range", 
                                    ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last Week"],
                                    index=2)
            
            time_map = {
                "Last Hour": 1,
                "Last 6 Hours": 6,
                "Last 24 Hours": 24,
                "Last Week": 168
            }
            
            hours = time_map[time_range]
            
            # Load data
            alerts = self.load_alerts_data(hours)
            alerts_df = self.create_alerts_dataframe(alerts)
            
            st.write(f"**Loaded {len(alerts)} alerts**")
            
            # System controls
            self.render_system_controls()
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Metrics
            self.render_metrics(alerts_df)
            st.markdown("---")
            
            # Charts
            self.render_alerts_timeline(alerts_df)
            self.render_alert_types_distribution(alerts_df)
            
            # Statistics
            self.render_statistics_panel(alerts_df)
        
        with col2:
            # Recent alerts
            self.render_recent_alerts(alerts_df)
            st.markdown("---")
            
            # Export options
            self.render_export_options(alerts_df)

def main():
    """Main function to run the dashboard"""
    dashboard = SurveillanceDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()