# ApexPlanet Internship - Task 1: Data Immersion & Wrangling
# Dataset familiarization, data quality assessment, cleaning, transformations,
# and export of the analysis-ready dataset.

import pandas as pd
import numpy as np

# Google Colab upload (if running in Colab)
from google.colab import files
uploaded = files.upload()

# Load dataset
input_file = "ApexPlanet_DataAnalytics_Dataset.xlsx"
df = pd.read_excel(input_file)
print("Dataset loaded successfully!")

# -------------------- Initial Exploration --------------------
print("First 5 rows:")
display(df.head())
print("Dataset dimensions:")
print(df.shape)
print("Column names:")
print(df.columns.tolist())
print("Dataset Information:")
df.info()

# Missing values and exact duplicate assessment
print("Missing values in each column:")
print(df.isnull().sum())
print("Total duplicate rows:", df.duplicated().sum())

# Unique values in categorical columns
for col in ['Gender', 'City', 'Product', 'Category']:
    print(f"\nUnique values in {col}:")
    print(df[col].value_counts())

# Check text columns for leading/trailing spaces
text_columns = df.select_dtypes(include='object').columns
for col in text_columns:
    count = (df[col] != df[col].str.strip()).sum()
    print(f"{col}: {count} values with extra spaces")

# Summary statistics
print(df.describe())

# Verify Total_Sales calculation
# (These two columns are retained in the final dataset as validation fields.)
df['Calculated_Sales'] = df['Quantity'] * df['Unit_Price']
df['Sales_Difference'] = (df['Total_Sales'] - df['Calculated_Sales']).round(2)
print("Records with mismatched sales:", (df['Sales_Difference'] != 0).sum())
display(df[['Quantity', 'Unit_Price', 'Total_Sales',
             'Calculated_Sales', 'Sales_Difference']].head(10))
print(df.describe())

# -------------------- Outlier Assessment (IQR) --------------------
numerical_columns = ['Age', 'Quantity', 'Unit_Price', 'Total_Sales']
for col in numerical_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"\nColumn: {col}")
    print(f"Q1: {Q1}")
    print(f"Q3: {Q3}")
    print(f"IQR: {IQR}")
    print(f"Lower Bound: {lower_bound}")
    print(f"Upper Bound: {upper_bound}")
    print(f"Number of Outliers: {len(outliers)}")

# Inspect potential Total_Sales outliers
Q1 = df['Total_Sales'].quantile(0.25)
Q3 = df['Total_Sales'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
sales_outliers = df[(df['Total_Sales'] < lower_bound) |
                    (df['Total_Sales'] > upper_bound)]
display(sales_outliers[['Order_ID', 'Order_Date', 'Product', 'Category',
                         'Quantity', 'Unit_Price', 'Total_Sales']]
        .sort_values(by='Total_Sales', ascending=False))

# -------------------- Date Assessment --------------------
print("Data type before conversion:")
print(df['Order_Date'].dtype)
df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
print("\nData type after conversion:")
print(df['Order_Date'].dtype)
print("\nInvalid or missing dates:", df['Order_Date'].isna().sum())
print("Earliest order date:", df['Order_Date'].min())
print("Latest order date:", df['Order_Date'].max())
print("Sample order dates:")
display(df['Order_Date'].head())

# -------------------- Logical Validation --------------------
print("Age below 18:", (df['Age'] < 18).sum())
print("Age above 65:", (df['Age'] > 65).sum())
print("Zero or negative Quantity:", (df['Quantity'] <= 0).sum())
print("Zero or negative Unit_Price:", (df['Unit_Price'] <= 0).sum())
print("Zero or negative Total_Sales:", (df['Total_Sales'] <= 0).sum())
print("Duplicate Order_IDs:", df['Order_ID'].duplicated().sum())
print("Exact duplicate rows:", df.duplicated().sum())

# -------------------- Data Cleaning & Transformation --------------------
# 1. Remove exact duplicate rows (assessment found zero; included for clean-script reproducibility)
df = df.drop_duplicates()

# 2. Handle missing values: Age -> median; City -> 'Unknown'
df['Age'] = df['Age'].fillna(df['Age'].median())
df['City'] = df['City'].fillna('Unknown')

# 3. Standardize categorical text fields by trimming spaces and title-casing
text_columns = ['Gender', 'City', 'Product', 'Category']
for col in text_columns:
    df[col] = df[col].str.strip().str.title()

# 4. Feature engineering: create age groups from existing Age column
df['Customer_Age_Group'] = pd.cut(
    df['Age'],
    bins=[17, 25, 35, 45, 55, 65],
    labels=['18-25', '26-35', '36-45', '46-55', '56-65']
)

# -------------------- Final Dataset Check & Export --------------------
print("Data cleaning and transformation completed!")
print("Final dataset shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())

output_file = 'ApexPlanet_Cleaned_Dataset.csv'
df.to_csv(output_file, index=False)
print(f"Cleaned dataset saved as {output_file}")
files.download(output_file)
