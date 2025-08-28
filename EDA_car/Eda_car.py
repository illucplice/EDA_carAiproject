import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os

# Load the data from the data folder
df = pd.read_csv('data/car (1).csv', on_bad_lines='skip', delimiter=',', quotechar='"', skipinitialspace=True, header=0)

# Debug: Print initial raw data shape and first few rows
initial_row_count = df.shape[0]
print("="*60)
print(f"📊 INITIAL DATASET: {initial_row_count} rows, {df.shape[1]} columns")
print("="*60)
print("\nFirst 5 rows (raw):\n", df.head().to_string())

# Manually set column names if header is missing or incorrect
df.columns = ['name', 'price', 'used_for', 'transmission', 'colour', 'make_year', 'mileage', 'engine_cc', 'fuel', 'kilometer_run', 'warranty', 'types']

print("\nStandardized columns:", df.columns.tolist())
print(f"\nInitial missing values:\n{df.isnull().sum()}")

# ===================== BASIC CLEANING (from original) =====================
# Clean Make Year: To numeric, filter to reasonable years
df['make_year'] = pd.to_numeric(df['make_year'], errors='coerce')
rows_before_year_filter = len(df)
df = df[df['make_year'].between(1900, 2025)]
rows_after_year_filter = len(df)
rows_dropped_year = rows_before_year_filter - rows_after_year_filter

print(f"\n🔹 After filtering invalid years: {rows_after_year_filter} rows (dropped {rows_dropped_year} rows)")

# Optimized Clean Price: Extract and convert to lakhs efficiently
def clean_price(p):
    if pd.isna(p):
        return np.nan
    p_str = str(p).strip()
    match = re.search(r'रू\.\s*([\d,]+(?:\.\d+)?)', p_str)
    if match:
        price_str = match.group(1).replace(',', '')
        try:
            price = float(price_str) / 100000
            return price
        except ValueError:
            return np.nan
    return np.nan

df['price'] = df['price'].apply(clean_price)

# Clean Mileage: Extract number, cap at 50 km/l
def clean_mileage(m):
    if pd.isna(m):
        return np.nan
    m = re.sub(r'[^\d.]', '', str(m))
    try:
        m = float(m)
        return min(m, 50)
    except ValueError:
        return np.nan

df['mileage'] = df['mileage'].apply(clean_mileage)

# Clean Engine (CC): To numeric, cap at 5000 cc
df['engine_cc'] = pd.to_numeric(df['engine_cc'].str.replace(r'[Cc]c?', '', regex=True), errors='coerce')
df['engine_cc'] = df['engine_cc'].apply(lambda x: min(x, 5000) if pd.notna(x) else x)

# Clean Kilometer Run: To numeric
df['kilometer_run'] = pd.to_numeric(df['kilometer_run'], errors='coerce')

# Clean Transmission
def clean_transmission(t):
    if pd.isna(t):
        return np.nan
    t = str(t).replace('PowerSteering', 'Manual').replace('Power Steering', 'Manual').strip()
    t = re.sub(r'(\dWD)', r' \1', t)
    return t

df['transmission'] = df['transmission'].apply(clean_transmission)

# Clean Fuel
df['fuel'] = df['fuel'].str.lower().str.strip().replace({
    'petrol': 'Petrol', 'diesel': 'Diesel', 'electric': 'Electric', 'hybrid': 'Hybrid'
})

# Remove any completely empty rows
rows_before_empty_removal = len(df)
df.dropna(how='all', inplace=True)
rows_after_empty_removal = len(df)
empty_rows_dropped = rows_before_empty_removal - rows_after_empty_removal

print(f"\n🔹 After removing completely empty rows: {rows_after_empty_removal} rows (dropped {empty_rows_dropped} empty rows)")
print(f"🔹 Current dataset: {len(df)} rows × {len(df.columns)} columns")

print(f"\nAfter basic cleaning - Missing values:\n{df.isnull().sum()}")

# ===================== ADVANCED MISSING VALUE HANDLING =====================

# 1. Handle 'used_for' missing values (483 missing)
print("\n=== Handling 'used_for' missing values ===")
print("Current 'used_for' distribution:")
print(df['used_for'].value_counts())

# Fill missing 'used_for' with most common value or 'Unknown'
most_common_use = df['used_for'].mode()
if len(most_common_use) > 0:
    df['used_for'].fillna(most_common_use[0], inplace=True)
else:
    df['used_for'].fillna('Unknown', inplace=True)

print(f"After filling: {df['used_for'].isnull().sum()} missing values")

# 2. Handle 'colour' missing values (79 missing)
print("\n=== Handling 'colour' missing values ===")
print("Current 'colour' distribution (top 10):")
print(df['colour'].value_counts().head(10))

# Clean and standardize colour names
def clean_colour(c):
    if pd.isna(c):
        return 'Unknown'
    c = str(c).strip().title()
    # Standardize common variations
    colour_mapping = {
        'White': 'White', 'Black': 'Black', 'Silver': 'Silver', 'Gray': 'Gray', 'Grey': 'Gray',
        'Red': 'Red', 'Blue': 'Blue', 'Green': 'Green', 'Yellow': 'Yellow', 'Brown': 'Brown'
    }
    for standard, variations in colour_mapping.items():
        if any(var.lower() in c.lower() for var in [variations]):
            return standard
    return 'Other'

df['colour'] = df['colour'].apply(clean_colour)
print(f"After cleaning and filling: {df['colour'].isnull().sum()} missing values")

# 3. Handle 'mileage' missing values (384 missing) - Intelligent imputation
print("\n=== Handling 'mileage' missing values ===")

# Impute mileage based on fuel type and engine size
def impute_mileage(row):
    if pd.notna(row['mileage']):
        return row['mileage']
    
    fuel_type = row['fuel']
    engine_cc = row['engine_cc']
    
    # Default mileage by fuel type
    fuel_defaults = {
        'Petrol': 15.0,
        'Diesel': 20.0,
        'Electric': 100.0,  # km per charge equivalent
        'Hybrid': 25.0
    }
    
    base_mileage = fuel_defaults.get(fuel_type, 15.0)
    
    # Adjust based on engine size if available
    if pd.notna(engine_cc):
        if engine_cc < 1000:
            base_mileage *= 1.2  # Smaller engines are more efficient
        elif engine_cc > 2000:
            base_mileage *= 0.8  # Larger engines are less efficient
    
    return base_mileage

df['mileage'] = df.apply(impute_mileage, axis=1)
print(f"After intelligent imputation: {df['mileage'].isnull().sum()} missing values")

# 4. Handle 'engine_cc' missing values (109 missing)
print("\n=== Handling 'engine_cc' missing values ===")

# Impute engine_cc based on fuel type and car age
def impute_engine_cc(row):
    if pd.notna(row['engine_cc']):
        return row['engine_cc']
    
    fuel_type = row['fuel']
    make_year = row['make_year']
    
    # Default engine sizes by fuel type
    fuel_engine_defaults = {
        'Petrol': 1200,
        'Diesel': 1500,
        'Electric': 0,  # Electric cars don't have traditional engines
        'Hybrid': 1400
    }
    
    base_engine = fuel_engine_defaults.get(fuel_type, 1200)
    
    # Adjust for car age (newer cars tend to have more efficient, smaller engines)
    if pd.notna(make_year) and make_year > 2015:
        base_engine *= 0.9
    
    return int(base_engine)

df['engine_cc'] = df.apply(impute_engine_cc, axis=1)
print(f"After intelligent imputation: {df['engine_cc'].isnull().sum()} missing values")

# 5. Handle 'kilometer_run' missing values (138 missing)
print("\n=== Handling 'kilometer_run' missing values ===")

# Impute based on car age
def impute_kilometer_run(row):
    if pd.notna(row['kilometer_run']):
        return row['kilometer_run']
    
    make_year = row['make_year']
    current_year = 2024
    
    if pd.notna(make_year):
        car_age = current_year - make_year
        # Assume average 12,000 km per year
        estimated_km = car_age * 12000
        # Add some variation (±30%)
        variation = np.random.uniform(0.7, 1.3)
        return max(0, int(estimated_km * variation))
    
    return 50000  # Default for unknown age

# Set random seed for reproducibility
np.random.seed(42)
df['kilometer_run'] = df.apply(impute_kilometer_run, axis=1)
print(f"After intelligent imputation: {df['kilometer_run'].isnull().sum()} missing values")

# 6. Handle 'warranty' missing values (629 missing)
print("\n=== Handling 'warranty' missing values ===")
print("Current 'warranty' distribution:")
print(df['warranty'].value_counts())

# Most cars don't have warranty info, fill with 'No Warranty'
df['warranty'].fillna('No Warranty', inplace=True)
print(f"After filling: {df['warranty'].isnull().sum()} missing values")

# 7. Handle 'types' missing values (836 missing)
print("\n=== Handling 'types' missing values ===")
print("Current 'types' distribution:")
print(df['types'].value_counts())

# Fill missing types based on other features or use 'Sedan' as default
def impute_car_type(row):
    if pd.notna(row['types']):
        return row['types']
    
    # Simple heuristic based on engine size
    engine_cc = row['engine_cc']
    
    if pd.notna(engine_cc):
        if engine_cc < 1000:
            return 'Hatchback'
        elif engine_cc > 2500:
            return 'SUV'
        else:
            return 'Sedan'
    
    return 'Sedan'  # Default

df['types'] = df.apply(impute_car_type, axis=1)
print(f"After intelligent imputation: {df['types'].isnull().sum()} missing values")

# ===================== FINAL VALIDATION AND SUMMARY =====================

final_row_count = len(df)
rows_retained = final_row_count
rows_lost = initial_row_count - final_row_count
retention_rate = (rows_retained / initial_row_count) * 100

print(f"\n" + "="*60)
print(f"📈 FINAL ROW COUNT SUMMARY")
print(f"="*60)
print(f"🔢 Initial rows:           {initial_row_count:,}")
print(f"🔢 Rows after year filter: {rows_after_year_filter:,}")
print(f"🔢 Empty rows removed:     {empty_rows_dropped:,}")
print(f"🔢 FINAL ROWS:             {final_row_count:,}")
print(f"📊 Retention rate:         {retention_rate:.1f}%")
print(f"📉 Total rows lost:        {rows_lost:,}")
print(f"="*60)

print(f"\nFinal dataset shape: {df.shape}")
print(f"\nFinal missing values:\n{df.isnull().sum()}")

# Check for any remaining missing values
total_missing = df.isnull().sum().sum()
print(f"\nTotal missing values remaining: {total_missing}")

if total_missing == 0:
    print("✅ SUCCESS: All missing values have been handled!")
else:
    print("⚠  WARNING: Some missing values still remain.")

# Data quality checks
print(f"\n=== DATA QUALITY CHECKS ===")
print(f"Price range: {df['price'].min():.2f} - {df['price'].max():.2f} lakhs")
print(f"Year range: {df['make_year'].min():.0f} - {df['make_year'].max():.0f}")
print(f"Mileage range: {df['mileage'].min():.2f} - {df['mileage'].max():.2f} km/l")
print(f"Engine CC range: {df['engine_cc'].min():.0f} - {df['engine_cc'].max():.0f} cc")
print(f"Kilometer run range: {df['kilometer_run'].min():.0f} - {df['kilometer_run'].max():.0f} km")

# Save the cleaned dataset
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
df.to_csv(f'{output_dir}/cleaned_car_data.csv', index=False)
print(f"\n✅ Cleaned dataset saved to '{output_dir}/cleaned_car_data.csv'")

# Generate updated visualizations
print(f"\n=== GENERATING UPDATED VISUALIZATIONS ===")

# 1. Missing values before/after comparison (if you want to track the improvement)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# You'd need to store the original missing values for comparison
# For now, let's show the final distribution
current_missing = df.isnull().sum()
current_missing = current_missing[current_missing > 0]  # Only non-zero

if len(current_missing) > 0:
    current_missing.plot(kind='bar', ax=ax1, color='red', alpha=0.7)
    ax1.set_title('Remaining Missing Values (After Cleaning)')
    ax1.set_ylabel('Count of Missing Values')
    ax1.tick_params(axis='x', rotation=45)
else:
    ax1.text(0.5, 0.5, 'No Missing Values!', ha='center', va='center', 
             transform=ax1.transAxes, fontsize=16, color='green', weight='bold')
    ax1.set_title('Missing Values Status')

# Data type distribution
dtype_counts = df.dtypes.value_counts()
dtype_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
ax2.set_title('Data Types Distribution')

plt.tight_layout()
plt.savefig(f'{output_dir}/data_quality_summary.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Updated feature distributions
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Price distribution
df['price'].hist(bins=30, ax=axes[0,0], color='skyblue', alpha=0.7)
axes[0,0].set_title('Price Distribution (Lakhs)')
axes[0,0].set_xlabel('Price (Lakhs)')
axes[0,0].set_ylabel('Frequency')

# Fuel type distribution
df['fuel'].value_counts().plot(kind='bar', ax=axes[0,1], color='lightcoral')
axes[0,1].set_title('Fuel Type Distribution')
axes[0,1].tick_params(axis='x', rotation=45)

# Car age distribution
current_year = 2024
df['car_age'] = current_year - df['make_year']
df['car_age'].hist(bins=20, ax=axes[1,0], color='lightgreen', alpha=0.7)
axes[1,0].set_title('Car Age Distribution')
axes[1,0].set_xlabel('Age (Years)')
axes[1,0].set_ylabel('Frequency')

# Types distribution
df['types'].value_counts().plot(kind='bar', ax=axes[1,1], color='gold')
axes[1,1].set_title('Car Types Distribution')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f'{output_dir}/updated_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ Updated visualizations saved to '{output_dir}/' folder")

# Final insights
print(f"\n=== KEY INSIGHTS FROM CLEANED DATA ===")
print(f"📋 FINAL DATASET SUMMARY:")
print(f"   • Total records: {df.shape[0]:,} rows")
print(f"   • Total features: {df.shape[1]} columns")
print(f"   • Data retention: {retention_rate:.1f}% of original data")
print(f"   • Missing values: {total_missing} (0%)")
print(f"\n📊 DATA CHARACTERISTICS:")
print(f"   • Most common fuel type: {df['fuel'].mode().iloc[0]} ({df['fuel'].value_counts().iloc[0]:,} cars)")
print(f"   • Most common car type: {df['types'].mode().iloc[0]} ({df['types'].value_counts().iloc[0]:,} cars)")
print(f"   • Average car age: {df['car_age'].mean():.1f} years")
print(f"   • Average price: {df['price'].mean():.2f} lakhs")
print(f"   • Average mileage: {df['mileage'].mean():.1f} km/l")
print(f"\n🎯 The dataset with {final_row_count:,} rows is now ready for machine learning models and analysis!")
print(f"="*60)
