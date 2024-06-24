import pandas as pd
import openpyxl

def UPN_counter(df):
    df['UPN Count'] = (df.iloc[:,1:17]).sum(axis = 1)
    return df



def analyze_descriptions(df):
    # Step 1: Identify unique descriptions
    unique_descriptions = df['Description New'].unique()
    
    # Step 2: Create a DataFrame to count occurrences of each description for each UPN
    upn_description_count = df.groupby(['UPN', 'Description New']).size().unstack(fill_value=0).reset_index()
    
    # Rename columns to include 'Count' in the description
    upn_description_count.columns = [col if col == 'UPN' else f"{col} Count" for col in upn_description_count.columns]
    
    # Step 3: Calculate the UPN count
    upn_count = df['UPN'].value_counts().reset_index()
    upn_count.columns = ['UPN', 'UPN Count']
    
    # Step 4: Merge UPN count with the description counts
    final_df = pd.merge(upn_count, upn_description_count, on='UPN', how='left')
    
    return final_df
