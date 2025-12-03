import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('Telco_Customer_Churn.csv')

print('='*80)
print('DATA QUALITY CHECK')
print('='*80)

print(f'\nDataset Shape: {df.shape}')
print(f'\nColumn Names:\n{df.columns.tolist()}')

print(f'\n--- Data Types ---')
print(df.dtypes)

print(f'\n--- TotalCharges Analysis ---')
print(f'TotalCharges dtype: {df["TotalCharges"].dtype}')

# Check for spaces in TotalCharges
if df["TotalCharges"].dtype == 'object':
    print(f'\n⚠️ TotalCharges is object type (should be numeric)!')
    blank_charges = df[df['TotalCharges'].str.strip() == '']
    print(f'Rows with blank TotalCharges: {len(blank_charges)}')
    
    if len(blank_charges) > 0:
        print(f'\nSample rows with blank TotalCharges:')
        print(blank_charges[['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']].head(15))

print(f'\n--- Missing Values ---')
print(df.isnull().sum())

print(f'\n--- Tenure Analysis ---')
print(f'Tenure range: {df["tenure"].min()} to {df["tenure"].max()}')
print(f'Customers with tenure = 0: {(df["tenure"] == 0).sum()}')
print(f'Customers with tenure = 1: {(df["tenure"] == 1).sum()}')

print(f'\n--- Churn Distribution ---')
print(df['Churn'].value_counts())
print(f'\nChurn percentage: {df["Churn"].value_counts(normalize=True) * 100}')

print(f'\n--- Duplicate customerIDs ---')
print(f'Duplicate IDs: {df["customerID"].duplicated().sum()}')

print('\n' + '='*80)
print('RECOMMENDATION: Drop rows where TotalCharges is blank (11 rows)')
print('='*80)
