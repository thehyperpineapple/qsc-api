import pandas as pd
import openpyxl

def UPN_counter(df):
    df['UPN Count'] = (df.iloc[:,1:17]).sum(axis = 1)
    return df

def dataset_cleaner(df, manual):
    if(manual):
        return UPN_counter(df)
    
    df.drop(['Description', 'IP', 'Location', 'Date', 'Severity '], axis = 1)
    df.dropna(inplace = True)

    grouped_data = df.groupby('UPN').agg(
        Access_from_anonymous_IP_address=('Description New', lambda x: (x == 'Access from anonymous IP address').sum()),
        Access_from_infrequent_location=('Description New', lambda x: (x == 'Access from infrequent location').sum()),
        Access_from_Malware_linked_IP_address=('Description New', lambda x: (x == 'Access from Malware linked IP address').sum()),
        Anonymous_IP_address=('Description New', lambda x: (x == 'Anonymous IP address').sum()),
        Delete_messages_from_Deleted_Items_folder=('Description New', lambda x: (x == 'Delete messages from Deleted Items folder').sum()),
        Domain_impersonation=('Description New', lambda x: (x == 'Domain impersonation').sum()),
        Failed_Logon_Activity=('Description New', lambda x: (x == 'Failed Logon Activity').sum()),
        FileAccessedExtended=('Description New', lambda x: (x == 'FileAccessedExtended').sum()),
        FileModifiedExtended=('Description New', lambda x: (x == 'FileAccessedExtended').sum()),
        Impossible_travel_activity=('Description New', lambda x: (x == 'Impossible travel activity').sum()),
        Malicious_URL_reputation=('Description New', lambda x: (x == 'Malicious URL reputation').sum()),
        Mas_download_by_a_single_user=('Description New', lambda x: (x == 'Mass download by a single user').sum()),
        SearchQueryPerformed=('Description New', lambda x: (x == 'SearchQueryPerformed').sum()),
        Suspicious_email=('Description New', lambda x: (x == 'Suspicious email').sum()),
        Threats=('Description New', lambda x: (x == 'Threats').sum()),
        Unfamiliar_sign_in_properties=('Description New', lambda x: (x == 'Unfamiliar sign-in properties').sum()),
        User_impersonation=('Description New', lambda x: (x == 'User impersonation').sum())
    ).reset_index()

    grouped_data = grouped_data.rename(columns={ 
        "Access_from_anonymous_IP_address" : "Access from anonymous IP address",
        "Access_from_infrequent_location" : "Access from infrequent location",
        "Access_from_Malware_linked_IP_address" : "Access from Malware linked IP address",
        "Anonymous_IP_address" : "Anonymous IP address",
        "Delete_messages_from_Deleted_Items_folder" : "Delete messages from Deleted Items folder",
        "Domain_impersonation" : "Domain impersonation",
        "Failed_Logon_Activity" : "Failed Logon Activity",
        "FileAccessedExtended" : "FileAccessedExtended",
        "FileModifiedExtended" : "FileModifiedExtended",
        "Impossible_travel_activity" : "Impossible travel activity",
        "Malicious_URL_reputation" : "Malicious URL reputation",
        "Mas_download_by_a_single_user" : "Mass download by a single user",
        "SearchQueryPerformed" : "SearchQueryPerformed",
        "Suspicious_email" : "Suspicious email",
        "Threats" : "Threats",
        "Unfamiliar_sign_in_properties" : "Unfamiliar sign-in properties",
        "User_impersonation" : "User impersonation"
    })

    # UPN_counter(grouped_data).to_csv("C:/Users/Adit Prasad/Downloads/Sample Output_File 1.csv")
    return UPN_counter(grouped_data)

# dataset_cleaner(pd.read_csv("C:/Users/Adit Prasad/Downloads/MLInput_MCAS_AADP.csv", encoding = 'cp1252'), False)

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
