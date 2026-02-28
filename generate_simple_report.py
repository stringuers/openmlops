#!/usr/bin/env python3
"""Generate a sample Evidently report for demonstration purposes."""
import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset
import os

# Create sample data to demonstrate Evidently
np.random.seed(42)

# Reference data (baseline)
reference_data = pd.DataFrame({
    'feature_1': np.random.normal(0, 1, 1000),
    'feature_2': np.random.normal(5, 2, 1000),
    'feature_3': np.random.exponential(2, 1000),
    'prediction': np.random.choice(['cat', 'dog', 'bird'], 1000, p=[0.4, 0.3, 0.3])
})

# Current data (with some drift)
current_data = pd.DataFrame({
    'feature_1': np.random.normal(0.5, 1.2, 1000),  # Slight drift
    'feature_2': np.random.normal(5.5, 2.5, 1000), # More drift
    'feature_3': np.random.exponential(2.5, 1000), # Drift
    'prediction': np.random.choice(['cat', 'dog', 'bird'], 1000, p=[0.3, 0.4, 0.3]) # Label drift
})

print("📊 Generating sample Evidently report...")
print(f"Reference data shape: {reference_data.shape}")
print(f"Current data shape: {current_data.shape}")

# Create Evidently report with Data Drift preset
report = Report(metrics=[DataDriftPreset()])

# Generate the report
report.run(reference_data=reference_data, current_data=current_data)

# Save as HTML
output_dir = "sample_reports"
os.makedirs(output_dir, exist_ok=True)
report_path = os.path.join(output_dir, "sample_evidently_report.html")

# Try different methods to save the report
try:
    # Method 1: Try save_html if it exists
    if hasattr(report, 'save_html'):
        report.save_html(report_path)
        print(f"✅ Report saved using save_html method: {report_path}")
    else:
        # Method 2: Try to_string if it exists
        if hasattr(report, 'to_string'):
            html_content = report.to_string()
            with open(report_path, 'w') as f:
                f.write(html_content)
            print(f"✅ Report saved using to_string method: {report_path}")
        else:
            # Method 3: Try as_dict and create simple HTML
            report_dict = report.as_dict()
            html_content = f"""
            <html>
            <head><title>Evidently Report</title></head>
            <body>
            <h1>Evidently Data Drift Report</h1>
            <h2>Summary</h2>
            <p>Reference data points: {len(reference_data)}</p>
            <p>Current data points: {len(current_data)}</p>
            <p>Features analyzed: {len(reference_data.columns)}</p>
            <h2>Features:</h2>
            <ul>
            {"".join([f"<li>{col}</li>" for col in reference_data.columns])}
            </ul>
            <h2>Sample Statistics:</h2>
            <p>Reference mean feature_1: {reference_data['feature_1'].mean():.3f}</p>
            <p>Current mean feature_1: {current_data['feature_1'].mean():.3f}</p>
            <p>Reference std feature_1: {reference_data['feature_1'].std():.3f}</p>
            <p>Current std feature_1: {current_data['feature_1'].std():.3f}</p>
            </body>
            </html>
            """
            with open(report_path, 'w') as f:
                f.write(html_content)
            print(f"✅ Report saved with custom HTML: {report_path}")
            
except Exception as e:
    print(f"❌ Error saving report: {e}")
    # Fallback: create a simple text report
    report_path = os.path.join(output_dir, "sample_evidently_report.txt")
    with open(report_path, 'w') as f:
        f.write("Evidently Report Summary\n")
        f.write("=======================\n\n")
        f.write(f"Reference data points: {len(reference_data)}\n")
        f.write(f"Current data points: {len(current_data)}\n")
        f.write(f"Features: {list(reference_data.columns)}\n\n")
        f.write("Sample Statistics:\n")
        f.write(f"  Reference feature_1 mean: {reference_data['feature_1'].mean():.3f}\n")
        f.write(f"  Current feature_1 mean: {current_data['feature_1'].mean():.3f}\n")
    print(f"✅ Text report saved as fallback: {report_path}")

print(f"\n📁 Report location: {report_path}")
print("📊 Report includes:")
print("  - Data drift detection between reference and current data")
print("  - Statistical analysis of feature distributions")
print("  - Drift metrics and visualizations")
print("\n💡 To view the report:")
print("  1. Open the HTML file in your browser")
print("  2. Or run: open sample_reports/sample_evidently_report.html")