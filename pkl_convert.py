import pandas as pd
import re

def process_llmob_data(raw_data):
    # Identify Data Chunks
    history_logs=[]
    
    # Ensure the list is long enough before accessing indices
    if len(raw_data)>0 and raw_data[0]: history_logs.extend(raw_data[0])
    if len(raw_data)>1 and raw_data[1]: history_logs.extend(raw_data[1])
    
    # Merge category maps
    category_map={}
    if len(raw_data)>3 and isinstance(raw_data[3],dict): category_map.update(raw_data[3])
    if len(raw_data)>10 and isinstance(raw_data[10],dict): category_map.update(raw_data[10])

    parsed_records=[]

    # Regex Pattern Definition
    date_pattern=re.compile(r'Activities at (\d{4}-\d{2}-\d{2})')
    activity_pattern=re.compile(r'(.*?)#(\d+)\s+at\s+(\d{1,2}:\d{2}:\d{2})')

    # Iterative Parsing
    for log_entry in history_logs:
        if not isinstance(log_entry,str): continue
        
        # Split the log into the Date part and the Activities part
        parts=log_entry.split(': ',1)
        if len(parts)!=2: continue
        
        header,activities_str=parts
        
        # Extract Date
        date_match=date_pattern.search(header)
        current_date=date_match.group(1) if date_match else 'Unknown'
        
        # Extract individual visits
        visits=activities_str.split(', ')
        
        for visit in visits:
            visit=visit.strip().rstrip('.')
            
            match=activity_pattern.match(visit)
            if match:
                loc_name=match.group(1).strip()
                loc_id=match.group(2)
                time_str=match.group(3)
                
                # Retrieve category from the map
                category=category_map.get(loc_name,category_map.get(loc_name.strip(),'Unknown'))

                parsed_records.append({
                    'Date': current_date,
                    'Time': time_str,
                    'Location_Name': loc_name,
                    'Location_ID': loc_id,
                    'Category': category,
                    'Full_Datetime': f'{current_date} {time_str}'})

    # Create DataFrame
    if not parsed_records: return pd.DataFrame()

    df=pd.DataFrame(parsed_records)
    df['Full_Datetime']=pd.to_datetime(df['Full_Datetime'])
    df=df.sort_values(by='Full_Datetime')
    
    return df