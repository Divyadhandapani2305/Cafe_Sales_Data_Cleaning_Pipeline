import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# STEP 1: LOAD THE DATASET
# ==========================================

df_raw = pd.read_csv('dirty_cafe_sales.csv')
df = df_raw.copy()
print(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.\n")

# Dictionary to store our Data Quality Report metrics
audit_report = {
    'Total Raw Rows': df.shape[0],
    'Duplicate Rows Removed': 0,
    'Corrupt Text Errors in Numbers': {},
    'Missing Values Fixed': {},
    'Outliers Detected & Capped': {}
}

# ==========================================
# STEP 2: AUTOMATED CLEANING & TRACKING
# ==========================================
print("=== Step 2: Running Automated Data Cleaning & Auditing ===")

# 1. Track & Remove exact duplicate rows
initial_rows = df.shape[0]
df = df.drop_duplicates()
audit_report['Duplicate Rows Removed'] = initial_rows - df.shape[0]

# Identify columns dynamically
quantity_col = [c for c in df.columns if 'quant' in c.lower() or 'unit' in c.lower()]
price_col = [c for c in df.columns if 'price' in c.lower() or 'rate' in c.lower()]
numeric_cols_to_check = quantity_col + price_col

# 2. Track & Fix Corrupt Text (like 'Error' or 'Unknown') in Numeric Columns
for col in numeric_cols_to_check:
    if col in df.columns:
        # Count how many non-numeric strings exist before coercing
        corrupt_count = pd.to_numeric(df[col], errors='coerce').isna().sum() - df[col].isna().sum()
        audit_report['Corrupt Text Errors in Numbers'][col] = corrupt_count
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. Track & Fix Outliers using the IQR Method
for col in numeric_cols_to_check:
    if col in df.columns:
        # Temporarily drop NaNs to calculate accurate quantiles
        temp_series = df[col].dropna()
        if not temp_series.empty:
            Q1 = temp_series.quantile(0.25)
            Q3 = temp_series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count how many outliers exist
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            audit_report['Outliers Detected & Capped'][col] = outlier_count
            
            # Cap the outliers to the upper/lower bounds
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])

# 4. Track & Handle Remaining Missing Values (Impute with Median)
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    missing_count = df[col].isnull().sum()
    audit_report['Missing Values Fixed'][col] = missing_count
    if missing_count > 0:
        df[col] = df[col].fillna(df[col].median())

# Track text column missing values
text_cols = df.select_dtypes(include=['object']).columns
for col in text_cols:
    missing_count = df[col].isnull().sum()
    audit_report['Missing Values Fixed'][col] = missing_count
    df[col] = df[col].fillna('Unknown')

# 5. Standardize text formatting
for col in text_cols:
    df[col] = df[col].astype(str).str.strip().str.title()


# ==========================================
# STEP 3: DATA TRANSFORMATION & AGGREGATION
# ==========================================
category_col = [c for c in df.columns if 'cat' in c.lower() or 'item' in c.lower() or 'type' in c.lower()]

if quantity_col and price_col and category_col:
    q_name = quantity_col[0]
    p_name = price_col[0]
    c_name = category_col[0]
    df['Total_Value'] = df[q_name] * df[p_name]
    summary_report = df.groupby(c_name)['Total_Value'].sum().reset_index()
else:
    c_name = text_cols[0] if len(text_cols) > 0 else df.columns[0]
    summary_report = df.groupby(c_name).size().reset_index(name='Record_Count')


# ==========================================
# STEP 4: EXPORTS & GENERATE VISUALS
# ==========================================
df.to_csv('cleaned_master_data.csv', index=False)
summary_report.to_csv('automated_summary_report.csv', index=False)

plt.figure(figsize=(10, 6))
x_axis, y_axis = summary_report.columns[0], summary_report.columns[1]
plt.bar(summary_report[x_axis], summary_report[y_axis], color='#2b5c8f', edgecolor='black')
plt.title(f'{y_axis} by {x_axis}', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('summary_report_chart.png', dpi=150)
plt.close()


# ==========================================
# STEP 5: PRINT THE FINAL DATA AUDIT REPORT
# ==========================================
print("\n" + "="*50)
print("       AUTOMATED DATA QUALITY AUDIT REPORT       ")
print("="*50)
print(f"Total Rows Processed      : {audit_report['Total Raw Rows']}")
print(f"Duplicate Rows Removed    : {audit_report['Duplicate Rows Removed']}")
print(f"Final Cleaned Rows Remaining: {df.shape[0]}")

print("\n[1] Text Noise Removed from Numeric Fields:")
for col, count in audit_report['Corrupt Text Errors in Numbers'].items():
    print(f"  - '{col}': Found and cleaned {count} text strings (e.g. 'Error')")

print("\n[2] Missing Values Handled & Imputed:")
for col, count in audit_report['Missing Values Fixed'].items():
    print(f"  - '{col}': Replaced {count} blank/null values")

print("\n[3] Outliers Handled (IQR Method):")
for col, count in audit_report['Outliers Detected & Capped'].items():
    print(f"  - '{col}': Detected and capped {count} statistical outliers")
print("="*50)
print("PROJECT STATUS: COMPLETE SUCCESS. Outputs updated successfully.")
print("="*50)