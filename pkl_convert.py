import pandas as pd
import re
import pickle
import os

def process_llmob_data(raw_data):
    """
    Parses the specific list structure from LLMob_UC .pkl files.
    """
    # 1. Identify Data Chunks (Trajectory Data is usually at index 0 and 1)
    history_logs = []
    
    # Safety check: ensure the list is long enough before accessing indices
    if len(raw_data) > 0 and raw_data[0]: 
        history_logs.extend(raw_data[0])
    if len(raw_data) > 1 and raw_data[1]: 
        history_logs.extend(raw_data[1])
    
    # Merge category maps (Index 3 and Index 10 based on the structure provided)
    category_map = {}
    if len(raw_data) > 3 and isinstance(raw_data[3], dict):
        category_map.update(raw_data[3])
    # Some files might have the map at index 10 (as seen in your dump), check if it exists
    if len(raw_data) > 10 and isinstance(raw_data[10], dict):
        category_map.update(raw_data[10])

    parsed_records = []

    # 2. Regex Pattern Definition
    # Matches: "Activities at 2019-01-02" -> extracts Date
    date_pattern = re.compile(r"Activities at (\d{4}-\d{2}-\d{2})")
    
    # Matches: "LocationName#123 at 10:00:00" -> extracts Name, ID, Time
    activity_pattern = re.compile(r"(.*?)#(\d+)\s+at\s+(\d{1,2}:\d{2}:\d{2})")

    # 3. Iterative Parsing
    for log_entry in history_logs:
        if not isinstance(log_entry, str): continue
        
        # Split the log into the Date part and the Activities part
        parts = log_entry.split(": ", 1)
        if len(parts) != 2: continue
        
        header, activities_str = parts
        
        # Extract Date
        date_match = date_pattern.search(header)
        current_date = date_match.group(1) if date_match else "Unknown"
        
        # Extract individual visits
        visits = activities_str.split(", ")
        
        for visit in visits:
            visit = visit.strip().rstrip(".")
            
            match = activity_pattern.match(visit)
            if match:
                loc_name = match.group(1).strip()
                loc_id = match.group(2)
                time_str = match.group(3)
                
                # Retrieve category from the map
                category = category_map.get(loc_name, 
                           category_map.get(loc_name.strip(), "Unknown"))

                parsed_records.append({
                    "Date": current_date,
                    "Time": time_str,
                    "Location_Name": loc_name,
                    "Location_ID": loc_id,
                    "Category": category,
                    "Full_Datetime": f"{current_date} {time_str}"
                })

    # 4. Create DataFrame
    if not parsed_records:
        return pd.DataFrame() # Return empty if no data found
        
    df = pd.DataFrame(parsed_records)
    df['Full_Datetime'] = pd.to_datetime(df['Full_Datetime'])
    df = df.sort_values(by='Full_Datetime')
    
    return df

# --- Main Execution ---

file_path = './data/2019/13.pkl'

if os.path.exists(file_path):
    try:
        with open(file_path, 'rb') as f:
            # Load the raw list data from the pickle file
            real_data = pickle.load(f)
        
        # Run the parser on the real data
        df_result = process_llmob_data(real_data)
        
        # Display results
        print(f"\nExtracted {len(df_result)} activity records.")
        print(df_result.head(10).to_markdown(index=False))
        
        # Optional: Save to CSV so you don't have to parse it again
        # df_result.to_csv('processed_1024_trajectory.csv', index=False)
        
    except Exception as e:
        print(f"Error processing file: {e}")
else:
    print(f"File not found at: {file_path}")
    print("Please check that the file path is correct and accessible.")