#!/usr/bin/env python3
"""Generate a sample Evidently report for demonstration purposes."""
import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
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

# Create Evidently report
report = Report(metrics=[
    DataDriftPreset(),
    DataSummaryPreset(),
])

# Generate the report
report.run(reference_data=reference_data, current_data=current_data)

# Save as HTML
output_dir = "sample_reports"
os.makedirs(output_dir, exist_ok=True)
report_path = os.path.join(output_dir, "sample_evidently_report.html")

# Save the report (using the correct method)
with open(report_path, 'w') as f:
    f.write(report.get_html())

print(f"✅ Sample Evidently report generated: {report_path}")
print("📊 Report includes:")
print("  - Data drift detection")
print("  - Feature distribution comparisons")
print("  - Statistical test results")
print("  - Data summary metrics")

# Show summary
try:
    report_dict = report.as_dict()
    drift_data = report_dict['metrics'][0]['result']
    print(f"\n📈 Drift Summary:")
    print(f"  Drift Share: {drift_data.get('drift_share', 'N/A')}")
    print(f"  Drifted Features: {drift_data.get('number_of_drifted_columns', 'N/A')}")
    print(f"  Total Features: {drift_data.get('number_of_columns', 'N/A')}")
except Exception as e:
    print(f"\n⚠️ Could not extract detailed metrics: {e}")
    print("Report was generated successfully, check the HTML file for full details.")