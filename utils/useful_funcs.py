import os
import re
import pandas as pd 

def extract_number(filename):
    match = re.search(r'(\d+)$', filename)
    return int(match.group(1)) if match else -1
    
def get_the_last_checkpoint(folder_path):
    files = os.listdir(folder_path)

    # Get the file with the highest number
    if files:
        sorted_files = sorted(files, key=extract_number)
        last_file = sorted_files[-1]
        return folder_path + "/" + last_file
    else:
        return None
        
def get_new_df_interval(name, n, interval): 
    df = pd.read_csv(name) 
    selected_rows = df[df['nsites'] == n].iloc[interval]

    # Creating a new DataFrame
    new_df = pd.DataFrame(selected_rows)
    return new_df
